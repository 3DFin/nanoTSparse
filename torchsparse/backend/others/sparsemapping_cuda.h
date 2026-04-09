#pragma once
#include "../hashmap/hashmap_cuda.h"

#include <torch/torch.h>

std::vector<at::Tensor> build_mask_from_kmap(int n_points, int n_out_points,
                                             const at::Tensor &_kmap,
                                             const at::Tensor &_kmap_sizes);

std::vector<at::Tensor> build_kernel_map_subm_hashmap(
    GPUHashMap &table, const at::Tensor &_in_coords,
    const at::Tensor &_coords_min, const at::Tensor &_coords_max,
    const at::Tensor &_kernel_sizes, const at::Tensor &_stride,
    const at::Tensor &padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_downsample_hashmap(
    GPUHashMap &table, const at::Tensor &_in_coords,
    const at::Tensor &_coords_min, const at::Tensor &_coords_max,
    const at::Tensor &_kernel_sizes, const at::Tensor &_stride,
    const at::Tensor &_padding, bool to_insert);
