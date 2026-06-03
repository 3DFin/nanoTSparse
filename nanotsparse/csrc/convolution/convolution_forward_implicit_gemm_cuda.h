#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

at::Tensor conv_forward_implicit_gemm_cuda(const at::Tensor& _in_feats,
                                           const at::Tensor& _kernel,
                                           const at::Tensor& _out_in_map,
                                           int64_t num_out_feats,
                                           int64_t num_out_channels,
                                           bool allow_tf32, bool allow_fp16);
