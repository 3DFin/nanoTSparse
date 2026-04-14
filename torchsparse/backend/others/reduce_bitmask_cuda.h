#pragma once

#include <ATen/Operators.h>
#include <cstdint>
#include <torch/all.h>
#include <torch/library.h>

at::Tensor reduce_bitmask_cuda(const at::Tensor &_bitmask_int, int64_t M_tile);
