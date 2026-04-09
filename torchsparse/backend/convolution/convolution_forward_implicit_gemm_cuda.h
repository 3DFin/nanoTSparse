#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

at::Tensor conv_forward_implicit_gemm_cuda(at::Tensor _in_feats, at::Tensor _kernel,
                       at::Tensor _out_in_map, int num_out_feats, int num_out_channels,
                       bool allow_tf32, bool allow_fp16);
