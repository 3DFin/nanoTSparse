#pragma once

#include <torch/torch.h>

at::Tensor exclusive_scan_quantified_wrapper(
    const int k_vol, const at::Tensor &neighbor_offset,
    const at::Tensor &neighbor_address, const at::Tensor &q_neighbor_address);
