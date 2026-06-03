from functools import lru_cache
from itertools import repeat

import torch

__all__ = ["make_ntuple", "make_tensor", "make_divisible"]


def make_ntuple(x: int | list[int] | tuple[int, ...] | torch.Tensor, ndim: int) -> tuple[int, ...]:
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, torch.Tensor):
        x = tuple(x.view(-1).cpu().numpy().tolist())

    assert isinstance(x, tuple) and len(x) == ndim, x
    return x


@lru_cache
def make_tensor(x: tuple[int, ...], dtype: torch.dtype, device) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device)


def make_divisible(x: int, divisor: int):
    return (x + divisor - 1) // divisor * divisor
