#include <stdlib.h>
#include <torch/torch.h>

#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>
#include <cmath>

// to_dense: feats (N x C), coords (N x 4), output (B x H x W x D x C)
// coords: batch, x, y, z
template <typename scalar_t>
__global__ void to_dense_forward_kernel(int N, int c, const scalar_t *__restrict__ feats, const int *__restrict__ coords, const int *__restrict__ range, scalar_t *__restrict__ out)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int i = index / c;
  int j = index % c;
  if (i < N)
  {
    const int *cur_coords = coords + 4 * i;
    int pos = cur_coords[0] * range[1] * range[2] * range[3] + cur_coords[1] * range[2] * range[3] + cur_coords[2] * range[3] + cur_coords[3];
    out[pos * c + j] = feats[index];
  }
}

// to_dense: top_grad (B x H x W x D x C), coords (N x 4), bottom_grad (N x C)
template <typename scalar_t>
__global__ void to_dense_backward_kernel(int N, int c, const scalar_t *__restrict__ top_grad, const int *__restrict__ coords, const int *__restrict__ range, scalar_t *__restrict__ bottom_grad)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int i = index / c;
  int j = index % c;
  if (i < N)
  {
    const int *cur_coords = coords + 4 * i;
    int pos = cur_coords[0] * range[1] * range[2] * range[3] + cur_coords[1] * range[2] * range[3] + cur_coords[2] * range[3] + cur_coords[3];
    bottom_grad[index] = top_grad[pos * c + j];
  }
}

void to_dense_forward_cuda(const at::Tensor inputs, const at::Tensor idx,
                           const at::Tensor range, at::Tensor outputs)
{
  c10::cuda::CUDAGuard guard(inputs.device());
  int N = inputs.size(0);
  int c = inputs.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inputs.scalar_type(), "to_dense_forward_cuda", ([&]
                                               { to_dense_forward_kernel<scalar_t><<<(N * c + 255) / 256, 256>>>(
                                                     N, c, inputs.data_ptr<scalar_t>(), idx.data_ptr<int>(),
                                                     range.data_ptr<int>(), outputs.data_ptr<scalar_t>()); }));
}

void to_dense_backward_cuda(const at::Tensor top_grad,
                            const at::Tensor idx, const at::Tensor range,
                            const at::Tensor bottom_grad)
{
  c10::cuda::CUDAGuard guard(top_grad.device());
  int N = bottom_grad.size(0);
  int c = bottom_grad.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "to_dense_backward_cuda", ([&]
                                                  { to_dense_backward_kernel<scalar_t><<<(N * c + 255) / 256, 256>>>(
                                                        N, c, top_grad.data_ptr<scalar_t>(), idx.data_ptr<int>(),
                                                        range.data_ptr<int>(), bottom_grad.data_ptr<scalar_t>()); }));
}
