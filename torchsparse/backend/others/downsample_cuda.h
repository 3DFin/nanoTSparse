#pragma once

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

at::Tensor
downsample_cuda(const at::Tensor &_in_coords, const at::Tensor &_coords_max,
                const at::Tensor &_coords_min, const at::Tensor &_kernel_sizes,
                const at::Tensor &_stride, const at::Tensor &_padding);
