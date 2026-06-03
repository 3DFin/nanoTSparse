#pragma once

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

at::Tensor reduce_bitmask_cuda(const at::Tensor& _bitmask_int, int64_t M_tile);
