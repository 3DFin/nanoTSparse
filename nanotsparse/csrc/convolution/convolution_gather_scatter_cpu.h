#pragma once

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

at::Tensor conv_forward_gather_scatter_cpu(const at::Tensor& in_feats,
                                           const at::Tensor& kernel,
                                           const at::Tensor& neighbor_maps,
                                           const at::Tensor& neighbor_offsets,
                                           int64_t output_size, bool transpose);

std::vector<at::Tensor> conv_backward_gather_scatter_cpu(
    const at::Tensor& in_feats, const at::Tensor& grad_out_feats,
    const at::Tensor& kernel, const at::Tensor& neighbor_maps,
    const at::Tensor& neighbor_offsets, bool transpose);
