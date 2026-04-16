#pragma once

#include <ATen/Operators.h>
#include <cstdint>
#include <torch/all.h>
#include <torch/library.h>

at::Tensor conv_forward_gather_scatter_cuda(
    at::Tensor &in_feat, at::Tensor &kernel, const at::Tensor &neighbor_map,
    const at::Tensor &neighbor_offset, const at::Tensor &input_mask,
    const at::Tensor &output_mask, const int64_t output_size, const double epsilon,
    const int64_t mm_thresh, const std::int8_t conv_mode, const bool transpose,
    at::Tensor buffer);

at::Tensor conv_forward_gather_scatter_cuda_latest(
    at::Tensor &in_feat, const at::Tensor &kernel,
    const at::Tensor &neighbor_map, const at::Tensor &neighbor_offset,
    const at::Tensor &input_mask, const at::Tensor &output_mask,
    const int output_size, const float epsilon, const int mm_thresh,
    const int conv_mode, const bool transpose, at::Tensor buffer);

at::Tensor conv_forward_gather_scatter_cuda_fallback(
    at::Tensor &in_feat, at::Tensor &kernel, const at::Tensor &neighbor_map,
    const int64_t output_size, const int8_t conv_mode,
    const at::Tensor &neighbor_offset, const bool transpose);

std::vector<at::Tensor> conv_backward_gather_scatter_cuda(
    const at::Tensor &in_feats, const at::Tensor &grad_out_feats,
    const at::Tensor &kernel, const at::Tensor &neighbor_maps,
    const at::Tensor &neighbor_offsets, bool transpose);
