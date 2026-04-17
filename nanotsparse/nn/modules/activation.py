from torch import nn

from nanotsparse import SparseTensor
from nanotsparse.nn.utils import fapply

__all__ = ["ReLU", "LeakyReLU", "SiLU", "GELU"]


class ReLU(nn.ReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class LeakyReLU(nn.LeakyReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class SiLU(nn.SiLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class GELU(nn.GELU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)
