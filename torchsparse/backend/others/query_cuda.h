#pragma once

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

void convert_transposed_out_in_map(const at::Tensor& out_in_map,
                                   at::Tensor out_in_map_t);

at::Tensor derive_bitmask_from_out_in_map(const at::Tensor& out_in_map,
                                          const int split_mask_num,
                                          int valid_n);
