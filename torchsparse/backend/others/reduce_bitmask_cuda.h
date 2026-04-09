#pragma once
#include <torch/torch.h>

at::Tensor reduce_bitmask_cuda(
    at::Tensor _bitmask_int,
    int M_tile
);
