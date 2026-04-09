#pragma once
#include <torch/torch.h>

at::Tensor reorder_out_in_map_cuda(
    const at::Tensor& _out_in_map,
    const at::Tensor& _reorder_loc
);
