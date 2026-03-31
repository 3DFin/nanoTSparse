#include "hashmap_cpu.h"
#include <iostream>

std::vector<at::Tensor> build_mask_from_kmap_native(int n_points, int n_out_points,
                                             at::Tensor _kmap,
                                             at::Tensor _kmap_sizes) {
  int kernel_volume = _kmap_sizes.size(0);
  auto options =
      torch::TensorOptions().dtype(at::ScalarType::Int).device(_kmap.device());
  at::Tensor _input_mask = torch::full({kernel_volume * n_points}, -1, options);
  at::Tensor _output_mask =
      torch::full({kernel_volume * n_out_points}, -1, options);
  at::Tensor _cum_kmap_sizes =
      torch::cumsum(_kmap_sizes, 0).to(at::ScalarType::Int);

  int *kmap_sizes = _kmap_sizes.data_ptr<int>();
  int *cum_kmap_sizes = _cum_kmap_sizes.data_ptr<int>();
  int *kmap = _kmap.data_ptr<int>();
  int *input_mask = _input_mask.data_ptr<int>();
  int *output_mask = _output_mask.data_ptr<int>();

  #pragma omp parallel for
  for (int k = 0; k < kernel_volume; ++k) {
    int n_neighbors = kmap_sizes[k]; // num of offsets with this K
    if ((n_points == n_out_points) && (kernel_volume % 2) &&
        (k == kernel_volume / 2)) {
      continue;
    }
    int offset = k == 0 ? 0 : 2 * cum_kmap_sizes[k - 1];
    int *curr_in_kmap = &kmap[offset];
    int in_offset = k * n_points;
    int out_offset = k * n_out_points;
    for (int i = 0; i < n_neighbors; ++i) {
      input_mask[in_offset + curr_in_kmap[2 * i]] = i;
      output_mask[out_offset + curr_in_kmap[2 * i + 1]] = i;
    }
  }

  return {_input_mask, _output_mask};
}
