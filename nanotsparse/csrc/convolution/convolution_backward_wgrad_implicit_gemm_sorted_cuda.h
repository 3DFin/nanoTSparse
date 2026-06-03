#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

at::Tensor conv_backward_wgrad_implicit_gemm_sorted_cuda(
    const at::Tensor& _in_feats, const at::Tensor& _kernel,
    const at::Tensor& _out_in_map, const at::Tensor& _reduced_mask,
    const at::Tensor& _reorder_loc, int64_t split_k_iters, bool allow_tf32,
    bool allow_fp16);
