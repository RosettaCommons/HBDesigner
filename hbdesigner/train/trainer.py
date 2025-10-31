import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Protocol
import git
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
import wandb
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from hbdesigner.model.hbdesign_model import HBDesigner3
from hbdesigner.train.config import TrainConfig
from hbdesigner.utils import (
    NoamLR,
    create_logger,
    cycle,
    dist_cycle,
    model_grad_norm,
    set_main_process_device,
)


MODELS = {
    "HBDesigner3": HBDesigner3,
}

class SupervisedTrainer:
    def __init__(
        self, config: TrainConfig, print_config: bool = True, rank: int = 0
    ) -> None:
        """A generic supervised learning trainer that should be subclassed. Contains the main training loop in `fit`.

        Args:
            config (TrainConfig): The hyperparameters for the trainer.
            print_config (bool, optional): Whether to print the config. Defaults to True.
        """
        # There are three sources of config values
        #   - The default values specified in individual config classes
        #   - The default values specified in the `default_hps` method
        #   - The values passed in the constructor, typically what is called by the user
        # The final config is obtained by merging the three sources with the following precedence:
        #   config classes < default_hps < constructor (i.e. the constructor overrides the default_hps, and so on)
        self.default_cfg: TrainConfig = TrainConfig()
        self.set_default_hps(self.default_cfg)
        self.cfg: TrainConfig = OmegaConf.merge(self.default_cfg, config)
        self.print_config = print_config
        self.initial_step = 0
        self.to_terminate: List[Closable] = []

        # Set up device and rank
        self.rank = rank
        self.device = torch.device(self.rank)
        if not self.cfg.use_ddp:
            self.use_ddp = False
            set_main_process_device(self.device)
        else:
            self.use_ddp = True
            self.setup_ddp()

        # These objects are created by their respective setup methods.
        self.train_data: Dataset
        self.valid_data: Dataset
        self.model: nn.Module
        self.opt: torch.optim.Optimizer

        # Perform trainer setup.
        self.setup()

    def setup_ddp(self) -> None:
        """Sets up DDP for distributed training."""
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        if torch.cuda.device_count() < self.cfg.ddp_n_procs:
            raise RuntimeError(
                f"Number of GPUs detected ({torch.cuda.device_count()}) is less than the number of processes ({self.cfg.ddp_n_procs})."
            )

        # Make sure master variables are set
        os.environ["MASTER_ADDR"] = self.cfg.ddp_addr
        os.environ["MASTER_PORT"] = self.cfg.ddp_port

        # Initialize DDP
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.cfg.ddp_n_procs,
        )

    def set_default_hps(self, base: TrainConfig) -> None:
        """Sets hyperparameters that are specific for this trainer object.

        Args:
            base (TrainConfig): Default configuration specified in the class definitions.
        """
        raise NotImplementedError()

    def setup_data(self) -> None:
        """Sets up the data used by this trainer for training and validation."""
        raise NotImplementedError()

    def setup_model(self) -> None:
        """Sets up the model used by this trainer."""
        self.model = self._get_model()

        # Put model on device and wrap in DDP if necessary
        self.model.to(self.device)
        if self.use_ddp:
            self.model = DDP(
                self.model, device_ids=[self.rank], find_unused_parameters=True
            )

    def _get_model(self) -> nn.Module:
        """Returns newly constructed supervised-learning model."""
        model_name = self.cfg.model.model_name
        if model_name not in MODELS:
            raise ValueError(
                f"Currently only {list(MODELS.keys())} are supported model names."
            )
        return MODELS[model_name](self.cfg)

    def setup_opt(self) -> None:
        def opt(params, lr):
            if self.cfg.opt.opt == "adam":
                return torch.optim.Adam(
                    params,
                    lr,
                    (self.cfg.opt.momentum, 0.999),
                    eps=self.cfg.opt.adam_eps,
                    weight_decay=self.cfg.opt.weight_decay,
                )
            elif self.cfg.opt.opt == "sgd":
                return torch.optim.SGD(
                    params,
                    lr,
                    self.cfg.opt.momentum,
                    weight_decay=self.cfg.opt.weight_decay,
                )
            elif self.cfg.opt.opt == "noam":
                return torch.optim.Adam(
                    params,
                    lr=0,
                    betas=(0.9, 0.98),
                    eps=self.cfg.opt.adam_eps,
                )
            else:
                raise NotImplementedError(f"{self.cfg.opt.opt} is not implemented.")

        # Construct optimizer for model parameters.
        self.opt = opt(self.model.parameters(), self.cfg.opt.learning_rate)

        # Set up the learning rate schedule.
        if self.cfg.opt.lr_decay is not None:
            if self.cfg.opt.opt == "noam":
                raise ValueError("Noam opt is not compatible with custom LR decay.")
            self.lr_sched = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lambda steps: 2 ** (-steps / self.cfg.opt.lr_decay)
            )
        elif self.cfg.opt.opt == "noam":
            self.lr_sched = NoamLR(
                model_size=128,
                factor=self.cfg.opt.noam_factor,
                warmup=self.cfg.opt.noam_warmup,
                optimizer=self.opt,
                step=0,
            )

        # Set up the clip_grad_callback.
        self.clip_grad_callback = {
            "value": lambda params: torch.nn.utils.clip_grad_value_(
                params, self.cfg.opt.clip_grad_param
            ),
            "norm": lambda params: [
                torch.nn.utils.clip_grad_norm_(p, self.cfg.opt.clip_grad_param)
                for p in params
            ],
            "total_norm": lambda params: torch.nn.utils.clip_grad_norm_(
                params, self.cfg.opt.clip_grad_param
            ),
            "none": lambda x: None,
        }[self.cfg.opt.clip_grad_type]

    def step(self, loss: Tensor) -> None:
        # Compute gradients and clip.
        if self.cfg.mixed_precision:
            self.scaler.scale(loss).backward()
            with torch.no_grad():
                # Recommended to unscale grads before clipping
                self.scaler.unscale_(self.opt)
                g0 = model_grad_norm(self.model)
                self.clip_grad_callback(self.model.parameters())
                g1 = model_grad_norm(self.model)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            with torch.no_grad():
                g0 = model_grad_norm(self.model)
                self.clip_grad_callback(self.model.parameters())
                g1 = model_grad_norm(self.model)
            self.opt.step()

        # Zero out optimizer gradient.
        self.opt.zero_grad()

        # Perform scheduler steps.
        if hasattr(self, "lr_sched"):
            self.lr_sched.step()

        return {"grad_norm": g0, "grad_norm_clip": g1}

    def setup(self) -> None:
        # Create the log dir, if necessary.
        if self.cfg.log_dir is not None:
            if self.rank == 0:
                if os.path.exists(self.cfg.log_dir):
                    print(f"Log dir {self.cfg.log_dir} already exists.")
                # Make the weights dir which will also create the parent.
                os.makedirs(os.path.join(self.cfg.log_dir, "weights"), exist_ok=True)

        # Set up the data, model, and optimizer
        self.setup_data()
        self.setup_model()
        self.setup_opt()

        # Set up gradient scaler for mixed precision training.
        if self.cfg.mixed_precision:
            print("Automatic mixed precision training enabled.")
            self.scaler = torch.cuda.amp.GradScaler()

        # Try to get the git hash of the repo.
        try:
            self.cfg.git_hash = git.Repo(
                __file__, search_parent_directories=True
            ).head.object.hexsha[:7]
        except git.InvalidGitRepositoryError:
            self.cfg.git_hash = "unknown"

        # Save and print config, if necessary.
        yaml_cfg = OmegaConf.to_yaml(self.cfg)
        if self.rank == 0:
            if self.cfg.log_dir is not None:
                with open(
                    os.path.join(self.cfg.log_dir, "config.yaml"), "w", encoding="utf8"
                ) as f:
                    f.write(yaml_cfg)
            if self.print_config:
                print("Configuration:")
                print(yaml_cfg)

    def _make_data_loader(
        self,
        src: Dataset,
        batch_size: int,
        collate_fn: Callable,
        shuffle: Optional[bool] = None,
    ) -> DataLoader:
        if self.use_ddp:
            sampler = DistributedSampler(
                src,
                num_replicas=self.cfg.ddp_n_procs,
                rank=self.rank,
                shuffle=shuffle,
            )
        else:
            sampler = None

        return DataLoader(
            src,
            batch_size=batch_size,
            shuffle=(shuffle if sampler is None else None),
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=2 if self.cfg.num_workers else None,
        )

    def build_training_data_loader(self) -> DataLoader:
        return self._make_data_loader(
            self.train_data,
            self.cfg.algo.num_from_dataset,
            self.train_data.collate,
            True,
        )

    def build_validation_data_loader(self) -> DataLoader:
        return self._make_data_loader(
            self.valid_data,
            self.cfg.algo.valid_num_from_dataset,
            self.valid_data.collate,
        )

    def train_batch(self, batch: gd.Batch) -> Dict[str, Any]:
        tick = time.time()
        self.model.train()
        try:
            if self.cfg.mixed_precision:
                with torch.cuda.amp.autocast():
                    if not self.use_ddp:
                        loss, info = self.model.compute_losses(batch)
                    else:
                        results = self.model(batch)
                        loss, info = self.model.module._compute_losses(batch, results)
            else:
                if not self.use_ddp:
                    loss, info = self.model.compute_losses(batch)
                else:
                    results = self.model(batch)
                    loss, info = self.model.module._compute_losses(batch, results)
            if not torch.isfinite(loss):
                raise ValueError("Loss is not finite.")
            step_info = self.step(loss)
        except ValueError as e:
            if self.cfg.log_dir is not None:
                if self.rank == 0:
                    if self.use_ddp:
                        state_dict = self.model.module.state_dict()
                    else:
                        state_dict = self.model.state_dict()
                    torch.save(
                        [state_dict, batch, loss, info],
                        os.path.join(self.cfg.log_dir, "nonfinite_dump.pt"),
                    )
            raise e

        if step_info is not None:
            info.update(step_info)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        info["train_time"] = time.time() - tick

        train_info = {
            k: v.item() if hasattr(v, "item") else v
            for k, v in info.items()
            if not k.endswith("_batch")
        }
        return train_info

    def evaluate_batch(self, batch: gd.Batch) -> Dict[str, Any]:
        tick = time.time()
        self.model.eval()
        valid_info = {}

        if not self.use_ddp:
            _, loss_dict = self.model.compute_losses(batch)
        else:
            results = self.model(batch)
            _, loss_dict = self.model.module._compute_losses(batch, results)
        for k, v in loss_dict.items():
            if k.endswith("_batch"):
                valid_info[k[:-6]] = v.tolist()

        if hasattr(batch, "extra_info"):
            valid_info.update(batch.extra_info)
        valid_info["eval_time"] = time.time() - tick
        return valid_info

    @torch.no_grad()
    def validation_loop(
        self,
        dataloader: DataLoader,
        repeats: int = 1,
        steps: int = None,
    ) -> Dict[str, Any]:
        valid_info = {}
        steps = steps if steps is not None else len(dataloader)
        for _ in range(repeats):
            for _, batch in zip(range(steps), cycle(dataloader)):
                # Validate on a batch.
                info = self.evaluate_batch(batch.to(self.device))
                # Accumulate values in info
                if valid_info == {}:
                    valid_info = {
                        k: [v] if not isinstance(v, list) else v
                        for k, v in info.items()
                    }
                else:
                    for k, v in info.items():
                        if isinstance(v, list):
                            valid_info[k].extend(v)
                        else:
                            valid_info[k].append(v)

        # Take mean for each metric
        valid_info = {k: np.mean(v) for k, v in valid_info.items()}
        return valid_info

    def fit(
        self,
        logger: Optional[logging.Logger] = None,
        skip_initial_validation: bool = False,
    ) -> None:
        """Trains the model for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        # Create logger if not provided.
        if logger is None:
            logger = create_logger(
                log_file=os.path.join(self.cfg.log_dir, "train.log")
                if self.cfg.log_dir is not None
                else None
            )

        # Build dataloaders.
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()

        # Get some config values.
        print_freq = self.cfg.print_every
        valid_freq = self.cfg.validate_every
        ckpt_freq = self.cfg.checkpoint_every
        num_training_steps = self.cfg.num_training_steps

        # Create training dataloader cycler
        if not self.use_ddp:
            train_cycler = cycle(train_dl)
        else:
            train_cycler = dist_cycle(train_dl)

        # Train for num_training_steps steps.
        if self.rank == 0:
            logger.info("Starting training")
        start_time = time.time()
        for it, batch in zip(
            range(self.initial_step, num_training_steps), train_cycler
        ):
            # Train on a batch.
            info = self.train_batch(batch.to(self.device))
            info["time_spent"] = time.time() - start_time
            start_time = time.time()

            # Train logging and printing.
            if self.rank == 0:
                self.log(info, it, "train")
                if print_freq > 0 and it % print_freq == 0:
                    logger.info(
                        f"Iteration {it}: "
                        + " | ".join(f"{k}: {v:.3f}" for k, v in info.items())
                    )

            # Validation loop.
            if valid_freq > 0 and (it % valid_freq == 0):
                if not (skip_initial_validation and it == 0):
                    valid_info = self.validation_loop(
                        valid_dl, steps=self.cfg.num_validation_gen_steps
                    )
                    if self.rank == 0:
                        self.log(valid_info, it, "valid")
                        if print_freq > 0:
                            logger.info(
                                f"Validation - Iteration {it}: "
                                + " | ".join(
                                    f"{k}: {v:.3f}" for k, v in valid_info.items()
                                )
                            )

            # Checkpoint if necessary.
            if ckpt_freq > 0 and it % ckpt_freq == 0:
                self.save_state(it)

            if self.use_ddp:
                # Synchronize all processes
                dist.barrier()

        # Save a final checkpoint.
        self.save_state(num_training_steps)
        if self.use_ddp:
            # Synchronize all processes
            dist.barrier()

        # Perform final validation.
        if self.rank == 0:
            logger.info("Performing final validation...")
        final_info = self.validation_loop(valid_dl, steps=self.cfg.num_final_gen_steps)

        # Print and log final_info.
        if self.rank == 0:
            logger.info(
                "Final Validation - "
                + " | ".join(f"{k}:{v:.2f}" for k, v in final_info.items())
            )
            self.log(final_info, num_training_steps, "final")

        # Clean up dataloaders.
        del train_dl
        del valid_dl

    def terminate(self) -> None:
        # Close all logger handlers.
        logger = logging.getLogger("logger")
        for handler in logger.handlers:
            handler.close()

        for terminate in self.to_terminate:
            terminate()

        # Terminate DDP, if applicable
        if self.use_ddp:
            dist.destroy_process_group()

    def save_state(self, it: int) -> None:
        if self.rank == 0:
            if self.use_ddp:
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()

            # Create the state dictionary.
            state = {
                "model_state_dict": model_state_dict,
                "cfg": self.cfg,
                "step": it,
                "opt_state_dict": self.opt.state_dict(),
            }
            if self.cfg.opt.lr_decay is not None:
                state["lr_sched"] = self.lr_sched.state_dict()

            # Save the state.
            if self.cfg.log_dir is not None:
                fn = os.path.join(self.cfg.log_dir, "weights", f"model_ckpt_{it}.pt")
                torch.save(state, fn)

    def load_model_state(self, ckpt_path: str) -> None:
        # Load weights from saved checkpoint.
        map_location = {"cuda:0": f"cuda:{self.rank}"}
        state = torch.load(ckpt_path, map_location=map_location)
        if not self.use_ddp:
            self.model.load_state_dict(state["model_state_dict"])
            self.model.to(self.device)
        else:
            model = self._get_model()
            model.load_state_dict(state["model_state_dict"])
            model.to(self.device)
            self.model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)

        # Optimizer and scheduler info.
        self.opt.load_state_dict(state["opt_state_dict"])
        if self.cfg.opt.lr_decay is not None:
            self.lr_sched.load_state_dict(state["lr_sched"])

        # Update initial training step.
        self.initial_step = state["step"]
        if self.initial_step > self.cfg.num_training_steps:
            raise ValueError(
                f"Current initial_step ({self.initial_step}) > num_training_steps ({self.cfg.num_training_steps}!"
            )

    def log(self, info: Dict[str, float], index: int, key: str) -> None:
        # Log to wandb
        if wandb.run is not None:
            wandb.log({f"{key}_{k}": v for k, v in info.items()}, step=index)

    def __del__(self) -> None:
        self.terminate()


class Closable(Protocol):
    def close(self):
        pass
