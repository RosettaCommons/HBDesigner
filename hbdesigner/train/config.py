from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional
from omegaconf import MISSING

from hbdesigner.model.config import ModelConfig
from hbdesigner.utils import StrictDataClass


@dataclass
class OptimizerConfig(StrictDataClass):
    """Generic configuration for optimizers.

    Attributes:
    opt (str): The optimizer to use. Currently only "adam", "noam", and "sgd" are supported.
        Defaults to "adam".
    learning_rate (float): The learning rate. Defaults to 1e-3.
    lr_decay (Optional[float]): The learning rate decay, in steps. Decay follows
        f = 2 ** (-steps / lr_decay)). If None, then learning rate decay is disabled.
        Defaults to 20_000.
    weight_decay (float): The L2 weight decay. Defaults to 0.0.
    momentum (float): The momentum parameter value. Defaults to 0.9.
    clip_grad_type (str): The type of gradient clipping to use. Currently "value",
        "norm", "total_norm" or "none" are supported. Defaults to "norm".
    clip_grad_param (float): The parameter for gradient clipping. Defaults to 10.0.
    adam_eps (float): The epsilon parameter for Adam. Defaults to 1e-8.
    noam_factor (Optional[float]): Constant scaling factor for Noam lr curve. Defaults to 2.
    noam_warmup (Optional[int]): Warmup period, in optimizer steps, for Noam lr curve. Defaults to 4_000.
    """

    opt: str = "adam"
    learning_rate: float = 1e-3
    lr_decay: Optional[float] = 20_000
    weight_decay: float = 0.0
    momentum: float = 0.9
    clip_grad_type: str = "norm"
    clip_grad_param: float = 10.0
    adam_eps: float = 1e-8

    noam_factor: Optional[float] = 2
    noam_warmup: Optional[int] = 4_000


@dataclass
class TrainConfig(StrictDataClass):
    """Base configuration for training.

    Attributes:
        desc (str): A description of the experiment. Defaults to "".
        log_dir (Optional[str]): The directory that'll store logs, checkpoints, and samples.
            If None, logging is disabled. Defaults to None.
        device (str): The device to use for training. Defaults to "cuda:0".
        seed (int): The random seed. Defaults to 42.
        validate_every (int): The number of training steps after which to validate
            the model. Defaults to 1000.
        checkpoint_every (Optional[int]): The number of training steps after which
            to checkpoint the model. If None, checkpointing is disabled. Defaults to None.
        print_every (Optional[int]): The number of training steps after which to print the
            training loss. If None, loss printing is disabled. Defaults to None.
        num_training_steps (int): The number of training steps. Defaults to 10_000.
        num_validation_gen_steps (Optional[int]): The number of steps for which to generate
            graphs during validation. If None, validation is disabled. Defaults to None.
        num_final_gen_steps (Optional[int]): After training, the number of steps for which
            to generate graphs. If None, final generation is disabled. Defaults to None.
        num_workers (int): The number of workers to use for creating minibatches. If 0, then
            multiprocessing is disabled. Defaults to 0.
        mixed_precision (bool): Whether to enable automatic mixed precision. Can speed up training and reduce memory usage,
        but may be less stable. Defaults to False.
        git_hash (Optional[str]): The git hash of the current commit. Defaults to None.
        pickle_mp_messages (bool): Whether to pickle messages sent between processes (only
            relevant if num_workers > 0). Defaults to False.
        use_ddp (bool): Whether to use distributed data parallel (DDP)
            training. Defaults to False. Note this will override the `device`
            value and requires multiple GPUs.
        ddp_n_procs (int): The number of processes to use for DDP training.
            This value should equal the number of GPUs to use. Defaults to 2.
        ddp_addr (str): The address of the DDP master process. Defaults to "localhost".
        ddp_port (str): The port of the DDP master process. Defaults to "12345".
        algo (AlgoConfig): The algorithm configuration for training.
        model (ModelConfig): The model configuration for training.
        opt (OptimizerConfig): The optimizer configuration for training.
        env (EnvConfig): The environment configuration for training.
    """

    desc: str = ""
    log_dir: Optional[str] = None
    device: str = "cuda:0"
    seed: int = 42
    validate_every: int = 1000
    checkpoint_every: Optional[int] = None
    print_every: Optional[int] = None
    num_training_steps: int = 10_000
    num_validation_gen_steps: Optional[int] = None
    num_final_gen_steps: Optional[int] = None
    num_workers: int = 0
    mixed_precision: bool = False
    git_hash: Optional[str] = None
    pickle_mp_messages: bool = False
    use_ddp: bool = False
    ddp_n_procs: int = 2
    ddp_addr: str = "localhost"
    ddp_port: str = "12345"
    model: ModelConfig = field(default_factory=ModelConfig)
    opt: OptimizerConfig = field(default_factory=OptimizerConfig)


def init_empty(cfg: StrictDataClass) -> StrictDataClass:
    """Initialize a StrictDataClass with all fields set to MISSING,
    including nested dataclasses.

    This is meant to be used by a user to provide some configuration
    using default values while overwritting only the fields set by
    the user.

    Args:
        cfg (StrictDataClass): The config whose attributes should be used.

    Returns:
        StrictDataClass: A config with the same attributes as cfg, but
            all set to MISSING.
    """
    for f in fields(cfg):
        if is_dataclass(f.type):
            setattr(cfg, f.name, init_empty(f.type()))
        else:
            setattr(cfg, f.name, MISSING)

    return cfg
