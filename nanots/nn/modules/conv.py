import math
import sys
from typing import Dict, List, Tuple, Union

from functools import cached_property

import numpy as np
import torch
from torch import nn

import nanots
from nanots import SparseTensor
from nanots.nn import functional as F
from nanots.utils import make_ntuple

__all__ = ["Conv2d", "Conv3d"]


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: int = 1,
        bias: bool = False,
        transposed: bool = False,
        generative: bool = False,
        config: Dict = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=2)
        self.stride = make_ntuple(stride, ndim=2)
        self.dilation = make_ntuple(dilation, ndim=2)
        _padding = make_ntuple(padding, 2)
        self.padding = ()
        for i in range(2):
            if self.kernel_size[i] % 2 == 1 and self.stride[i] == 1:
                self.padding += ((self.kernel_size[i] - 1) // 2,)
            else:
                self.padding += (_padding[i],)
        self.transposed = transposed
        self.generative = generative

        self.kernel_volume = int(np.prod(self.kernel_size))
        if (
            self.kernel_volume > 1
            or self.kernel_volume == 1
            and self.stride != (1, 1)
        ):
            if self.transposed:
                self.kernel = nn.Parameter(
                    torch.zeros(in_channels, out_channels, *self.kernel_size)
                )
            else:
                self.kernel = nn.Parameter(
                    torch.zeros(out_channels, in_channels, *self.kernel_size)
                )
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"
        if self.stride != (1,) * len(self.stride):
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.bias is None:
            s += ", bias=False"
        if self.transposed:
            s += ", transposed=True"
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(
            (self.out_channels if self.transposed else self.in_channels)
            * self.kernel_volume
        )
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input: SparseTensor) -> SparseTensor:
        coords=input.coords
        feats = input.feats
        spatial_range = input.spatial_range

        if self.kernel_size == (1, 1) and self.stride == (1, 1) and self.dilation == (1, 1):
            tensor_stride = input.stride
            tensor_spatial_range = spatial_range
            feats = feats.matmul(self.kernel)
            if self.bias is not None:
                feats = feats + self.bias
        elif not self.transposed:
            feats = feats.unflatten(-2, spatial_range[:3]).movedim(-1, -3).contiguous()
            tensor_stride = tuple(
                input.stride[k] * self.stride[k] for k in range(2)
            ) + input.stride[2:]
            feats = torch.nn.functional.conv2d(
                feats, self.kernel, self.bias, self.stride, self.padding, self.dilation
            )
            feats = feats.movedim(-3, -1).contiguous()
            tensor_spatial_range = feats.shape[:3] + spatial_range[3:]
            feats = feats.flatten(-4, -2)
        else:
            tensor_stride = tuple(
                input.stride[k] // self.stride[k] for k in range(2)
            ) + input.stride[2:]
            coords, tensor_spatial_range = input._caches.cmaps[tensor_stride]
            output_padding = tuple(
                tensor_spatial_range[k + 1] - (
                    (spatial_range[k + 1] - 1) * self.stride[k]
                    - 2 * self.padding[k] + self.kernel_size[k]
                ) for k in range(2)
            )
            feats = feats.unflatten(-2, spatial_range[:3]).movedim(-1, -3).contiguous()
            feats = torch.nn.functional.conv_transpose2d(
                feats, self.kernel, self.bias, self.stride, self.padding, output_padding, 1, self.dilation
            )
            feats = feats.movedim(-3, -1).contiguous().flatten(-4, -2)

        output = SparseTensor(
            coords=coords,
            feats=feats,
            stride=tensor_stride,
            spatial_range=tensor_spatial_range,
        )
        output._caches = input._caches
        output._caches.cmaps.setdefault(
            output.stride, (output.coords, output.spatial_range)
        )
        return output


class Conv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: int = 1,
        bias: bool = False,
        transposed: bool = False,
        generative: bool = False,
        config: Dict = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.dilation = dilation
        _padding = make_ntuple(padding, 3)
        self.padding = ()
        for i in range(3):
            if self.kernel_size[i] % 2 == 1 and self.stride[i] == 1:
                self.padding += ((self.kernel_size[i] - 1) // 2,)
            else:
                self.padding += (_padding[i],)
        self.transposed = transposed
        self.generative = generative
        if self.generative:
            assert self.transposed

        self._config = config

        self.kernel_volume = int(np.prod(self.kernel_size))
        if (
            self.kernel_volume > 1
            or self.kernel_volume == 1
            and self.stride != (1, 1, 1)
        ):
            self.kernel = nn.Parameter(
                torch.zeros(self.kernel_volume, in_channels, out_channels)
            )
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"
        if self.stride != (1,) * len(self.stride):
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.bias is None:
            s += ", bias=False"
        if self.transposed:
            s += ", transposed=True"
        if self.generative:
            s += ", generative=True"
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(
            (self.out_channels if self.transposed else self.in_channels)
            * self.kernel_volume
        )
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.conv3d(
            input,
            weight=self.kernel,
            kernel_size=self.kernel_size,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            transposed=self.transposed,
            generative=self.generative,
            config=self._config,
            training=self.training,
        )
