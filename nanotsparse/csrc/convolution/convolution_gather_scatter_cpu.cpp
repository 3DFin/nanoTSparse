#include "convolution_gather_scatter_cpu.h"

#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/taskflow.hpp>

#include <algorithm>
#include <cassert>

void scatter_cpu(int n_in, int c, const float *in_feats, float *out_feat,
                 const int *kmap, bool transpose, tf::Executor &executor) {

  tf::Taskflow taskflow;
  taskflow.for_each_index(0, n_in, 1, [&](int i) {
    assert(out_pos >= 0);
    int out_pos = kmap[2 * i + 1 - transpose] * c;
    int in_pos = i * c;
    for (int j = 0; j < c; j++) {
      out_feat[out_pos + j] += in_feats[in_pos + j];
    }
  });
  executor.run(taskflow).get();
}

void gather_cpu(int n_k, int c, const float *in_feats, float *out_feat,
                const int *kmap, bool transpose, tf::Executor &executor) {
  tf::Taskflow taskflow;
  taskflow.for_each_index(0, n_k, 1, [&](int i) {
    assert(in_pos >= 0);
    int in_pos = kmap[2 * i + transpose] * c;
    int out_pos = i * c;
    for (int j = 0; j < c; j++) {
      out_feat[out_pos + j] = in_feats[in_pos + j];
    }
  });
  executor.run(taskflow).get();
}

at::Tensor conv_forward_gather_scatter_cpu(const at::Tensor &in_feats,
                                           const at::Tensor &kernel,
                                           const at::Tensor &neighbor_maps,
                                           const at::Tensor &neighbor_offsets,
                                           int64_t output_size,
                                           bool transpose) {
  if (in_feats.size(1) != kernel.size(1)) {
    throw std::invalid_argument("Input feature size and kernel size mismatch");
  }

  auto out_feat =
      torch::zeros({output_size, kernel.size(-1)}, in_feats.options());

  // kernel shape is volume x Cin x Cout
  int kernel_volume = kernel.size(0);
  int c_in = kernel.size(1);
  int c_out = kernel.size(2);

  tf::Executor executor;

  // buffer size, the largest number of neighbors for a given kernel offset
  int _buffer_size = 0;
  bool is_submanifold = false;

  // memory optimization.
  // Allocate buffer based of the offset with max neighbors
  // Check for submanifold configuration

  if (kernel_volume % 2 && output_size == in_feats.size(0)) {
    is_submanifold = true;

    torch::mm_out(out_feat, in_feats, kernel[kernel_volume / 2]);

    // check for max in the part "before" the central voxel
    _buffer_size =
        *std::max_element(neighbor_offsets.data_ptr<int>(),
                          neighbor_offsets.data_ptr<int>() + kernel_volume / 2);
    // compare it with the part "after" the central voxel
    _buffer_size =
        std::max(_buffer_size,
                 *std::max_element(
                     neighbor_offsets.data_ptr<int>() + kernel_volume / 2 + 1,
                     neighbor_offsets.data_ptr<int>() + kernel_volume));

  } else {
    _buffer_size =
        *std::max_element(neighbor_offsets.data_ptr<int>(),
                          neighbor_offsets.data_ptr<int>() + kernel_volume);
  }

  const auto options =
      at::TensorOptions().dtype(in_feats.dtype()).device(in_feats.device());

  auto in_buffer = at::zeros({_buffer_size, c_in}, options);
  auto out_buffer = torch::zeros({_buffer_size, c_out}, options);

  auto *in_buffer_ptr = in_buffer.data_ptr<float>();
  auto *out_buffer_ptr = out_buffer.data_ptr<float>();
  const auto *neighbor_offsets_ptr = neighbor_offsets.data_ptr<int>();
  const auto *in_feats_ptr = in_feats.data_ptr<float>();
  auto *out_feat_ptr = out_feat.data_ptr<float>();

  int cur_offset = 0;
  int center_voxel_id = kernel_volume / 2;
  for (int k = 0; k < kernel_volume; ++k) {
    // no neighbor for this kernel offset, so no computation
    int num_neighbors = neighbor_offsets_ptr[k];

    if (num_neighbors == 0) {
      continue;
    }

    if (is_submanifold && (k == center_voxel_id)) {
      cur_offset += 2 * num_neighbors;
      continue;
    }

    auto out_buffer_activated = torch::from_blob(
        static_cast<void *>(out_buffer_ptr), {num_neighbors, c_out}, options);
    auto in_buffer_activated = torch::from_blob(
        static_cast<void *>(in_buffer_ptr), {num_neighbors, c_in}, options);

    // gather
    gather_cpu(num_neighbors, c_in, in_feats_ptr,
               in_buffer_activated.data_ptr<float>(),
               neighbor_maps.data_ptr<int>() + cur_offset, transpose, executor);

    // matmul => out_buffer = in_buffer x kernel
    torch::mm_out(out_buffer_activated, in_buffer_activated, kernel[k]);

    // scatter_add
    scatter_cpu(neighbor_offsets_ptr[k], c_out,
                out_buffer_activated.data_ptr<float>(), out_feat_ptr,
                neighbor_maps.data_ptr<int>() + cur_offset, transpose, executor);

    cur_offset += 2 * num_neighbors;
  }
  return out_feat;
}

std::vector<at::Tensor> conv_backward_gather_scatter_cpu(
    const at::Tensor &in_feats, const at::Tensor &grad_out_feats,
    const at::Tensor &kernel, const at::Tensor &neighbor_maps,
    const at::Tensor &neighbor_offsets, bool transpose) {

  auto grad_in_feats = torch::zeros_like(in_feats);
  auto grad_kernel = torch::zeros_like(kernel);

  int kernel_volume = kernel.size(0);
  int c_in = kernel.size(1);
  int c_out = kernel.size(2);

  bool is_submanifold = false;

  const auto *neighbor_offsets_ptr = neighbor_offsets.data_ptr<int>();

  int _buffer_size = *std::max_element(neighbor_offsets_ptr,
                                       neighbor_offsets_ptr + kernel_volume);

  const auto options =
      torch::TensorOptions().dtype(in_feats.dtype()).device(in_feats.device());
  auto in_buffer = at::zeros({_buffer_size, c_in}, options);
  auto in_grad_buffer = at::zeros({_buffer_size, c_in}, options);
  auto out_grad_buffer = torch::zeros({_buffer_size, c_out}, options);

  tf::Executor executor;

  int cur_offset = 0;
  for (int k = 0; k < kernel_volume; k++) {
    auto kernel_grad_buffer = grad_kernel[k];
    if (is_submanifold && (k == kernel_volume / 2)) {
      cur_offset += 2 * neighbor_offsets_ptr[k];
      continue;
    }

    if (neighbor_offsets.data_ptr<int>()[k] == 0) {
      continue;
    }

    auto out_grad_buffer_activated =
        torch::from_blob(out_grad_buffer.data_ptr<float>(),
                         {neighbor_offsets_ptr[k], c_out}, options);
    auto in_grad_buffer_activated =
        torch::from_blob(in_grad_buffer.data_ptr<float>(),
                         {neighbor_offsets_ptr[k], c_in}, options);
    auto in_buffer_activated = torch::from_blob(
        in_buffer.data_ptr<float>(), {neighbor_offsets_ptr[k], c_in}, options);

    // gather
    gather_cpu(out_grad_buffer_activated.size(0), c_out,
               grad_out_feats.data_ptr<float>(),
               out_grad_buffer_activated.data_ptr<float>(),
               neighbor_maps.data_ptr<int>() + cur_offset, !transpose, executor);

    gather_cpu(in_buffer_activated.size(0), c_in, in_feats.data_ptr<float>(),
               in_buffer_activated.data_ptr<float>(),
               neighbor_maps.data_ptr<int>() + cur_offset, transpose, executor);

    // matmul
    torch::mm_out(in_grad_buffer_activated, out_grad_buffer_activated,
                  torch::transpose(kernel[k], 0, 1));
    torch::mm_out(kernel_grad_buffer,
                  torch::transpose(in_buffer_activated, 0, 1),
                  out_grad_buffer_activated);

    // scatter
    scatter_cpu(neighbor_offsets_ptr[k], c_in,
                in_grad_buffer_activated.data_ptr<float>(),
                grad_in_feats.data_ptr<float>(),
                neighbor_maps.data_ptr<int>() + cur_offset, !transpose,
                executor);

    cur_offset += 2 * neighbor_offsets_ptr[k];
  }
  return { grad_in_feats, grad_kernel };
}
