#pragma once

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

at::Tensor reorder_out_in_map_cuda(
    const at::Tensor& _out_in_map,
    const at::Tensor& _reorder_loc
);
