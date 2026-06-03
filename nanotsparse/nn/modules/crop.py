from torch import nn

from nanotsparse import SparseTensor
from nanotsparse.nn import functional as F

__all__ = ["SparseCrop"]


class SparseCrop(nn.Module):
    def __init__(
        self,
        coords_min: tuple[int, ...] | None = None,
        coords_max: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.coords_min = coords_min
        self.coords_max = coords_max

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.spcrop(input, self.coords_min, self.coords_max)
