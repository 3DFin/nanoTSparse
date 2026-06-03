#pragma once

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

at::Tensor conv_forward_gather_scatter_cuda_fallback(
    at::Tensor& in_feat, at::Tensor& kernel, const at::Tensor& neighbor_map,
    const int64_t output_size, const int8_t conv_mode,
    const at::Tensor& neighbor_offset, const bool transpose);

std::vector<at::Tensor> conv_backward_gather_scatter_cuda(
    const at::Tensor& in_feats, const at::Tensor& grad_out_feats,
    const at::Tensor& kernel, const at::Tensor& neighbor_maps,
    const at::Tensor& neighbor_offsets, bool transpose);
