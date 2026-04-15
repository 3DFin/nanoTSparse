#pragma once

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

at::Tensor convert_transposed_out_in_map(const at::Tensor& out_in_map, int64_t size);

at::Tensor derive_bitmask_from_out_in_map(const at::Tensor& out_in_map,
                                          int64_t split_mask_num,
                                          int64_t valid_n);
