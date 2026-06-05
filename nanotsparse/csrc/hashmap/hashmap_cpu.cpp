#include "hashmap_cpu.h"

#include <c10/util/Exception.h>

#include <cstdint>

std::vector<at::Tensor> build_mask_from_kmap_native(
    int64_t n_points, int64_t n_out_points, const at::Tensor& neighbor_maps,
    const at::Tensor& kmap_sizes) {
  // Input validation
  TORCH_CHECK(n_points > 0, "n_points must be positive");
  TORCH_CHECK(n_out_points > 0, "n_out_points must be positive");

  TORCH_CHECK(neighbor_maps.dim() == 2, "neighbor_maps must be a 2D tensor");
  TORCH_CHECK(neighbor_maps.scalar_type() == at::ScalarType::Int,
              "neighbor_maps must be an Int tensor");
  TORCH_CHECK(neighbor_maps.numel() > 0,
              "neighbor_maps tensor must not be empty");

  TORCH_CHECK(kmap_sizes.dim() == 1, "kmap_sizes must be a 1D tensor");
  TORCH_CHECK(kmap_sizes.scalar_type() == at::ScalarType::Int,
              "kmap_sizes must be an Int tensor");
  TORCH_CHECK(kmap_sizes.numel() > 0, "kmap_sizes tensor must not be empty");

  int kernel_volume = kmap_sizes.size(0);
  const auto options = torch::TensorOptions()
                           .dtype(at::ScalarType::Int)
                           .device(neighbor_maps.device());
  at::Tensor input_mask = at::full({kernel_volume * n_points}, -1, options);
  at::Tensor output_mask =
      torch::full({kernel_volume * n_out_points}, -1, options);
  at::Tensor cum_kmap_sizes =
      torch::cumsum(kmap_sizes, 0).to(at::ScalarType::Int);

  auto* kmap_sizes_ptr = kmap_sizes.data_ptr<int>();
  auto* cum_kmap_sizes_ptr = cum_kmap_sizes.data_ptr<int>();
  auto* kmap_ptr = neighbor_maps.data_ptr<int>();

  auto* input_mask_ptr = input_mask.data_ptr<int>();
  auto* output_mask_ptr = output_mask.data_ptr<int>();

  tf::Executor executor;
  tf::Taskflow taskflow;

  taskflow.for_each_index(0, kernel_volume, 1, [&](int k) {
    int n_neighbors = kmap_sizes_ptr[k];  // num of offsets with this K
    // submanifold test.
    if ((n_points == n_out_points) && (kernel_volume % 2) &&
        (k == kernel_volume / 2)) {
      return;
    }

    int offset = k == 0 ? 0 : 2 * cum_kmap_sizes_ptr[k - 1];
    const auto* curr_in_kmap = &kmap_ptr[offset];
    int in_offset = k * n_points;
    int out_offset = k * n_out_points;
    for (int i = 0; i < n_neighbors; ++i) {
      input_mask_ptr[in_offset + curr_in_kmap[2 * i]] = i;
      output_mask_ptr[out_offset + curr_in_kmap[2 * i + 1]] = i;
    }
  });

  return {input_mask, output_mask};
}
