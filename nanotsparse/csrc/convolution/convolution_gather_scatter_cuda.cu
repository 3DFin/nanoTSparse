#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "convolution_gather_scatter_cuda.h"

#define CONVERT_FLOAT(pointer) (reinterpret_cast<float *>(&(pointer))[0])
#define CONVERT_HALF2(pointer) (reinterpret_cast<half2 *>(&(pointer))[0])
#define CONVERT_HALF2_CONST(pointer) (reinterpret_cast<const half2 *>(&(pointer))[0])
#define CONVERT_INT4(pointer) (reinterpret_cast<int4 *>(&(pointer))[0])

template <typename scalar_t>
__global__ void gather_kernel(const int n_k, const int n_in, const int c,
                              const scalar_t *__restrict__ in_feat,
                              scalar_t *__restrict__ out_feat,
                              const int *__restrict__ kmap,
                              const bool transpose) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  bool isfloat = sizeof(scalar_t) == 4;
  int i, j;
  if (isfloat) {
    i = index / c;
    j = index % c;
  } else {
    i = index / (c >> 1);
    j = index % (c >> 1);
  }
  if (i >= n_k)
    return;
  int in_pos = kmap[2 * i + transpose];
  if (in_pos < 0)
    return;
  if (isfloat) {
    out_feat[i * c + j] = in_feat[in_pos * c + j];
  } else {
    CONVERT_HALF2(out_feat[i * c + (j << 1)]) =
        CONVERT_HALF2_CONST(in_feat[in_pos * c + (j << 1)]);
  }
}

template <typename scalar_t>
__global__ void scatter_kernel(const int n_in, const int n_out, const int c,
                               const scalar_t *__restrict__ in_feat,
                               scalar_t *__restrict__ out_feat,
                               const int *__restrict__ kmap,
                               const bool transpose) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j;
  bool isfloat = sizeof(scalar_t) == 4;
  if (isfloat) {
    i = index / c;
    j = index % c;
  } else {
    i = index / (c >> 1);
    j = index % (c >> 1);
  }
  if (i >= n_in)
    return;
  int out_pos = kmap[2 * i + 1 - transpose];
  if (out_pos < 0 || out_pos >= n_out)
    return;
  if (isfloat) {
    out_feat[out_pos * c + j] += in_feat[i * c + j];
  } else {
    half2 cur_out_feat = CONVERT_HALF2(out_feat[out_pos * c + (j << 1)]);
    cur_out_feat =
        __hadd2(cur_out_feat, CONVERT_HALF2_CONST(in_feat[i * c + (j << 1)]));
    CONVERT_HALF2(out_feat[out_pos * c + (j << 1)]) = cur_out_feat;
  }
}

at::Tensor conv_forward_gather_scatter_cuda_fallback(
    at::Tensor &in_feat, at::Tensor &kernel, const at::Tensor &neighbor_map,
    const int64_t output_size, const int8_t conv_mode,
    const at::Tensor &neighbor_offset, const bool transpose) {
  c10::cuda::CUDAGuard guard(in_feat.device());
  if (in_feat.size(1) != kernel.size(1)) {
    throw std::invalid_argument("Input feature size and kernel size mismatch");
  }
  bool is_half = in_feat.scalar_type() == at::ScalarType::Half;
  auto options =
      torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
  at::Tensor out_feat = at::zeros({output_size, kernel.size(-1)}, options);

  // need to avoid misaligned memory access
  bool padded = false;
  if (is_half) {
    if (in_feat.size(1) % 2 != 0) {
      in_feat = torch::cat(
          {in_feat, torch::zeros({in_feat.size(0), 1}, options)}, -1);
      kernel = torch::cat(
          {kernel, torch::zeros({kernel.size(0), 1, kernel.size(2)}, options)},
          1);
    }
    if (out_feat.size(1) % 2 != 0) {
      out_feat = torch::cat(
          {out_feat, torch::zeros({out_feat.size(0), 1}, options)}, -1);
      kernel = torch::cat(
          {kernel, torch::zeros({kernel.size(0), kernel.size(1), 1}, options)},
          -1);
      padded = true;
    }
  }

  int n_in_feats = in_feat.size(0);
  int n_in_channels = in_feat.size(1);
  int n_out_feats = out_feat.size(0);
  int n_out_channels = out_feat.size(1);
  int kernel_volume = kernel.size(0);
  // memory optimization
  bool precompute_mid = false;
  int mid_kernel = kernel_volume / 2;
  int in_buffer_size = 1;
  // we can precompute features for w[0,0] which avoids gather/scatter
  if (kernel_volume % 2 == 1 && n_in_feats == n_out_feats) {
    precompute_mid = true;
    in_buffer_size =
        *std::max_element(neighbor_offset.data_ptr<int>(),
                          neighbor_offset.data_ptr<int>() + mid_kernel);
    in_buffer_size = std::max(
        in_buffer_size,
        *std::max_element(neighbor_offset.data_ptr<int>() + mid_kernel + 1,
                          neighbor_offset.data_ptr<int>() + kernel_volume));
    in_buffer_size = std::max(in_buffer_size, 1);
    // (N, c) X (c, o) = (N, o)
    // conv_mode == 2 indicates kernel has been reordered, in which case
    // w[0,0] is placed at the end
    int mid_kmap_idx = conv_mode != 2 ? kernel_volume / 2 : kernel_volume - 1;
    torch::mm_out(out_feat, in_feat, kernel[mid_kmap_idx]);
  } else {
    in_buffer_size =
        *std::max_element(neighbor_offset.data_ptr<int>(),
                          neighbor_offset.data_ptr<int>() + kernel_volume);
  }
  auto in_buffer = torch::zeros({in_buffer_size, n_in_channels}, options);
  auto out_buffer = torch::zeros({in_buffer_size, n_out_channels}, options);
  int cur_offset = 0;
  // gather/gemm/scatter on each weight
  for (int i = 0; i < kernel_volume; i++) {
    int n_active_feats = neighbor_offset.data_ptr<int>()[i];
    // if there's no active features for this weight, skip it
    if (n_active_feats == 0) {
      continue;
    }
    // if w[0,0] was precomputed above, skip it
    if ((i == mid_kernel) && precompute_mid) {
      cur_offset += 2 * n_active_feats;
      continue;
    }
    // in_buffer_activated (i, c) holds the dense input features from gather
    // for i = n_active_feats (# of features in the activated kernel from
    // neighbor_offset) out_buffer_activated (i, o) holds the dense output
    // features to scatter
    at::Tensor out_buffer_activated;
    at::Tensor in_buffer_activated;
    if (is_half) {
      out_buffer_activated =
          torch::from_blob(out_buffer.data_ptr<at::Half>(),
                           {n_active_feats, n_out_channels}, options);
      in_buffer_activated =
          torch::from_blob(in_buffer.data_ptr<at::Half>(),
                           {n_active_feats, n_in_channels}, options);
    } else {
      out_buffer_activated =
          torch::from_blob(out_buffer.data_ptr<float>(),
                           {n_active_feats, n_out_channels}, options);
      in_buffer_activated =
          torch::from_blob(in_buffer.data_ptr<float>(),
                           {n_active_feats, n_in_channels}, options);
    }
    // gather n_active_feats dense features from N sparse input features with c
    // feature dimensions
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        in_feat.scalar_type(), "conv_forward_gather_scatter_cuda", ([&] {
          gather_kernel<scalar_t>
              <<<ceil((double)(n_active_feats * n_in_channels) / 256), 256>>>(
                  n_active_feats, n_in_feats, n_in_channels,
                  in_feat.data_ptr<scalar_t>(),
                  in_buffer_activated.data_ptr<scalar_t>(),
                  neighbor_map.data_ptr<int>() + cur_offset, transpose);
        }));
    // gemm: (i, c) X (c, o) = (i, o)
    int kmap_idx = i;
    if (conv_mode == 2) {
      kmap_idx = i < mid_kernel ? i * 2 : (kernel_volume - i) * 2 - 1;
    }
    torch::mm_out(out_buffer_activated, in_buffer_activated, kernel[kmap_idx]);
    // scatter n_active_feats dense features into n_out_feats output features of
    // dimension n_out_channels
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        in_feat.scalar_type(), "conv_forward_gather_scatter_cuda", ([&] {
          scatter_kernel<scalar_t>
              <<<ceil((double)(n_active_feats * n_out_channels) / 256), 256>>>(
                  n_active_feats, n_out_feats, n_out_channels,
                  out_buffer_activated.data_ptr<scalar_t>(),
                  out_feat.data_ptr<scalar_t>(),
                  neighbor_map.data_ptr<int>() + cur_offset, transpose);
        }));
    cur_offset += 2 * n_active_feats;
  }

  if (padded) {
    out_feat = at::slice(out_feat, 1, 0, n_out_channels - 1).contiguous();
  }
  return out_feat;
}

std::vector<at::Tensor> conv_backward_gather_scatter_cuda(
    const at::Tensor &in_feats, const at::Tensor &grad_out_feats,
    const at::Tensor &kernel, const at::Tensor &neighbor_maps,
    const at::Tensor &neighbor_offsets, bool transpose) {
  c10::cuda::CUDAGuard guard(in_feats.device());

  auto grad_in_feats = torch::zeros_like(in_feats);
  auto grad_kernel = torch::zeros_like(kernel);

  bool is_half = in_feats.scalar_type() == at::ScalarType::Half;
  int n_in_feats = in_feats.size(0);
  int n_in_channels = in_feats.size(1);
  int n_out_feats = grad_out_feats.size(0);
  int n_out_channels = kernel.size(-1);
  int kernel_volume = kernel.size(0);
  bool flag = false;

  int in_buffer_size =
      *std::max_element(neighbor_offsets.data_ptr<int>(),
                        neighbor_offsets.data_ptr<int>() + kernel_volume);
  auto options =
      torch::TensorOptions().dtype(in_feats.dtype()).device(in_feats.device());

  auto in_buffer = at::zeros({in_buffer_size, in_feats.size(1)}, options);
  auto in_grad_buffer =
      torch::zeros({in_buffer_size, in_feats.size(1)}, options);
  auto out_grad_buffer =
      torch::zeros({in_buffer_size, kernel.size(2)}, options);

  int cur_offset = 0;
  for (int i = 0; i < kernel_volume; i++) {
    auto kernel_grad_buffer = grad_kernel[i];
    int n_active_feats = neighbor_offsets.data_ptr<int>()[i];
    if (flag && (i == kernel_volume / 2)) {
      cur_offset += 2 * n_active_feats;
      continue;
    }
    if (n_active_feats == 0) {
      continue;
    }

    // Can't figure out a cleaner way to do this
    at::Tensor out_grad_buffer_activated;
    at::Tensor in_grad_buffer_activated;
    at::Tensor in_buffer_activated;

    if (is_half) {
      out_grad_buffer_activated =
          torch::from_blob(out_grad_buffer.data_ptr<at::Half>(),
                           {n_active_feats, kernel.size(2)}, options);
      in_grad_buffer_activated =
          torch::from_blob(in_grad_buffer.data_ptr<at::Half>(),
                           {n_active_feats, in_feats.size(1)}, options);
      in_buffer_activated =
          torch::from_blob(in_buffer.data_ptr<at::Half>(),
                           {n_active_feats, in_feats.size(1)}, options);
    } else {
      out_grad_buffer_activated =
          torch::from_blob(out_grad_buffer.data_ptr<float>(),
                           {n_active_feats, kernel.size(2)}, options);
      in_grad_buffer_activated =
          torch::from_blob(in_grad_buffer.data_ptr<float>(),
                           {n_active_feats, in_feats.size(1)}, options);
      in_buffer_activated =
          torch::from_blob(in_buffer.data_ptr<float>(),
                           {n_active_feats, in_feats.size(1)}, options);
    }

    // gather
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        in_feats.scalar_type(), "conv_forward_gather_scatter_cuda", ([&] {
          gather_kernel<scalar_t>
              <<<ceil((double)(n_active_feats * n_out_channels) / 256), 256>>>(
                  n_active_feats, n_out_feats, n_out_channels,
                  grad_out_feats.data_ptr<scalar_t>(),
                  out_grad_buffer_activated.data_ptr<scalar_t>(),
                  neighbor_maps.data_ptr<int>() + cur_offset, !transpose);
        }));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        in_feats.scalar_type(), "conv_forward_gather_scatter_cuda", ([&] {
          gather_kernel<scalar_t>
              <<<ceil((double)(n_active_feats * n_in_channels) / 256), 256>>>(
                  n_active_feats, n_in_feats, n_in_channels,
                  in_feats.data_ptr<scalar_t>(),
                  in_buffer_activated.data_ptr<scalar_t>(),
                  neighbor_maps.data_ptr<int>() + cur_offset, transpose);
        }));
    // gemm
    torch::mm_out(in_grad_buffer_activated, out_grad_buffer_activated,
                  torch::transpose(kernel[i], 0, 1));
    torch::mm_out(kernel_grad_buffer,
                  torch::transpose(in_buffer_activated, 0, 1),
                  out_grad_buffer_activated);

    // scatter
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        in_feats.scalar_type(), "conv_forward_gather_scatter_cuda", ([&] {
          scatter_kernel<scalar_t>
              <<<ceil((double)(n_active_feats * n_in_channels) / 256), 256>>>(
                  n_active_feats, n_in_feats, n_in_channels,
                  in_grad_buffer_activated.data_ptr<scalar_t>(),
                  grad_in_feats.data_ptr<scalar_t>(),
                  neighbor_maps.data_ptr<int>() + cur_offset, !transpose);
        }));
    cur_offset += 2 * n_active_feats;
  }
  return {grad_in_feats, grad_kernel};
}
