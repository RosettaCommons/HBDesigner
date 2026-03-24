import logging
import os
import random
import sys
import contextlib
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from hbdesigner.data.features import Array

from dataclasses import fields, is_dataclass
from typing import Optional
from omegaconf import MISSING, OmegaConf


def get_config_from_file(filename: str) -> OmegaConf:
    """Load a config from a yaml file."""
    assert os.path.isfile(filename), f"Invalid config file {filename} specified."
    return OmegaConf.load(filename)


class StrictDataClass:
    """A dataclass that raises an error if any field is created outside of the __init__ method, while
    still enabling changing current attributes.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name) or name in self.__annotations__:
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'."
                f" '{type(self).__name__}' is a StrictDataClass object."
                f" Attributes can only be defined in the class definition."
            )


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


def create_logger(
    name: str = "logger",
    level: float = logging.INFO,
    log_file: Optional[str] = None,
    stream_handle: bool = True,
) -> logging.Logger:
    # Create logger and set its level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create logging handlers and formatter.
    handlers = []
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="a"))
    if stream_handle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - {} - %(message)s".format(name),
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    # Apply formatters to handlers and handlers to logger
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def cycle(it: Iterable) -> Any:
    while True:
        for i in it:
            yield i


def dist_cycle(loader: DataLoader) -> Any:
    while True:
        if loader.sampler is not None:
            # Set epoch for sampler to random value for shuffle
            loader.sampler.set_epoch(random.randint(0, sys.maxsize))
        for i in loader:
            yield i


def model_grad_norm(model: nn.Module) -> torch.Tensor:
    x = torch.zeros(1).to(next(model.parameters()).device)
    for i in model.parameters():
        if i.grad is not None:
            x += (i.grad * i.grad).sum()
    return torch.sqrt(x)


def detach_and_cpu(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    elif isinstance(x, dict):
        x = {k: detach_and_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        x = [detach_and_cpu(v) for v in x]
    elif isinstance(x, tuple):
        x = tuple(detach_and_cpu(v) for v in x)
    return x


_worker_rngs = {}
_worker_rng_seed = [120723]
_main_process_device = [torch.device("cpu")]


def get_worker_rng() -> int:
    worker_info = torch.utils.data.get_worker_info()
    wid = worker_info.id if worker_info is not None else 0
    if wid not in _worker_rngs:
        _worker_rngs[wid] = np.random.RandomState(_worker_rng_seed[0] + wid)
    return _worker_rngs[wid]


def set_worker_rng_seed(seed: int) -> None:
    _worker_rng_seed[0] = seed
    for wid in _worker_rngs:
        _worker_rngs[wid].seed(seed + wid)


def set_main_process_device(device: torch.device) -> None:
    _main_process_device[0] = device


def get_worker_device() -> torch.device:
    worker_info = torch.utils.data.get_worker_info()
    return _main_process_device[0] if worker_info is None else torch.device("cpu")

def is_cuda_device(device: Union[str, torch.device]) -> bool:
    dev = torch.device(device)
    return dev.type == "cuda" and torch.cuda.is_available()


def get_autocast_context(
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float16,
):
    if is_cuda_device(device):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def seed_everything(seed: int = 42) -> None:
    """Seeds everything EXCEPT the worker seeds (see above fxn)"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init(worker_id):
    # For seeding torch dataloaders
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + worker_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)


# Some functions that return bool. Used mostly for testing
def is_value(x: Array, v: Union[int, float]) -> bool:
    if isinstance(x, torch.Tensor):
        return (x == v * torch.ones_like(x)).all()
    else:
        return (x == v * np.ones_like(x)).all()


def n_nonzero(x: Array) -> int:
    if isinstance(x, torch.Tensor):
        return (x != torch.zeros_like(x)).sum()
    else:
        return (x != np.zeros_like(x)).sum()


def has_keys(x: Any, keys: List[Any]) -> bool:
    return [k in x for k in keys] == [True] * 4


def has_shape(x: Array, shape: Tuple[int]) -> bool:
    return x.shape == shape


def only_contains(x: Array, elems: List[Union[int, float]]) -> bool:
    contains = [e in x for e in elems] == [True] * len(elems)
    unique = np.unique(x)
    only = [i in elems for i in unique] == [True] * len(unique)
    return contains and only


class NoamLR(LRScheduler):
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
