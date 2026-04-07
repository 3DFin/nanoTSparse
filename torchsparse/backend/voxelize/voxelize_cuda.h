#pragma once

#include <torch/torch.h>

void to_dense_forward_cuda(const at::Tensor inputs, const at::Tensor idx,
                           const at::Tensor range, at::Tensor outputs);

void to_dense_backward_cuda(const at::Tensor top_grad,
                            const at::Tensor idx, const at::Tensor range,
                            const at::Tensor bottom_grad);
