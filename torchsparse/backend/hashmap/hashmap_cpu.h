#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include <iostream>
#include <torch/extension.h>

template <typename key_type, typename val_type> class HashTableCPU {
private:
  struct VoxelKey {
    std::array<key_type, 4> coords;

    VoxelKey() = default;
    explicit VoxelKey(const key_type *ptr) {
      for (int j = 0; j < 4; ++j) {
        coords[j] = ptr[j];
      }
    }

    bool operator==(const VoxelKey &other) const {
      return coords[0] == other.coords[0] && coords[1] == other.coords[1] &&
             coords[2] == other.coords[2] && coords[3] == other.coords[3];
    }
  };

  struct VoxelKeyHash {
    // USE FNV hashing
    size_t operator()(const VoxelKey &key) const {
      size_t hash = 14695981039346656037UL;
      for (int j = 0; j < 4; j++) {
        hash ^= (unsigned int)key.coords[j];
        hash *= 1099511628211UL;
      }
      // hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
      return hash;
    }
  };

  std::unordered_map<VoxelKey, val_type, VoxelKeyHash> hashmap;

public:
  HashTableCPU() = default;

  HashTableCPU(torch::Tensor table_keys, torch::Tensor table_vals) {
    assert(table_keys.is_same_size(table_vals));
    hashmap.reserve(table_keys.size(0));
    auto *key_ptr = table_keys.data_ptr<key_type>();
    auto *val_ptr = table_vals.data_ptr<val_type>();
    for (size_t i = 0; i < table_keys.size(0); ++i) {
      VoxelKey key(key_ptr + i * 4);
      hashmap[key] = val_ptr[i];
    }
  }

  ~HashTableCPU() = default;

  void insert_vals(torch::Tensor keys) {
    // TODO: Implement if needed but it seems to be used nowhere in the code.
  }

  torch::Tensor lookup_vals(torch::Tensor keys) {
    auto options =
        torch::TensorOptions().dtype(at::ScalarType::Int).device(keys.device());
    auto results = torch::zeros({keys.size(0)}, options);
    auto *key_ptr = keys.data_ptr<key_type>();
    auto *results_ptr = results.data_ptr<int>();
    for (size_t i = 0; i < keys.size(0); ++i) {
      VoxelKey key(key_ptr + i * 4);
      auto it = hashmap.find(key);
      results_ptr[i] = (it != hashmap.end()) ? it->second : 0;
    }
    return results;
  }

  void insert_coords(torch::Tensor coords) {
    auto *key_raw_ptr = coords.data_ptr<key_type>();
    hashmap.reserve(coords.size(0));
    for (size_t id = 0; id < coords.size(0); ++id) {
      VoxelKey key(&key_raw_ptr[id * 4]);
      hashmap[key] = id + 1;
    }
  }

  std::vector<std::array<int, 3>> compute_kernel_offsets(int kx, int ky,
                                                         int kz) {
    int kernel_volume = kx * ky * kz;
    std::vector<std::array<int, 3>> result;
    result.reserve(kernel_volume);

    for (int k = 0; k < kernel_volume; ++k) {
      int kernel_idx = k;
      std::array<int, 3> offset;
      for (int i = 0; i < 3; i++) {
        int cur_offset = kernel_idx % kx;
        cur_offset -= (kx - 1) / 2;
        offset[i] = cur_offset;
        kernel_idx /= kx;
      }
      result.emplace_back(offset);
    }
    return result;
  }

  torch::Tensor lookup_coords(at::Tensor coords, at::Tensor kernel_sizes,
                              at::Tensor strides, int kernel_volume) {
    auto options = torch::TensorOptions()
                       .dtype(at::ScalarType::Int)
                       .device(coords.device());
    auto results =
        torch::zeros({coords.size(0), kernel_volume}, options).contiguous();
    auto *key_raw_ptr = coords.data_ptr<key_type>();
    auto *strides_raw = strides.data_ptr<int>();
    auto *results_raw = results.data_ptr<int>();
    auto *kernel_sizes_ptr = kernel_sizes.data_ptr<int>();

    auto offsets = compute_kernel_offsets(
        kernel_sizes_ptr[0], kernel_sizes_ptr[1], kernel_sizes_ptr[2]);

#pragma omp parallel for
    for (size_t id = 0; id < coords.size(0); ++id) {
      const auto *in_coords = key_raw_ptr + id * 4;
      for (size_t k = 0; k < kernel_volume; ++k) {
        key_type out_coords[4];
        out_coords[3] = in_coords[3]; // batch is the same
        const auto &k_offsets = offsets[k];
        for (size_t dim = 0; dim < 3; ++dim) {
          out_coords[dim] = in_coords[dim] * strides_raw[dim] + k_offsets[dim];
        }
        VoxelKey key(&out_coords[0]);
        auto maybe_id = hashmap.find(key);
        if (maybe_id != hashmap.end())
          results_raw[id * kernel_volume + k] =
              static_cast<int>(maybe_id->second);
      }
    }
    return results;
  }
};

std::vector<at::Tensor> build_mask_from_kmap_native(int n_points, int n_out_points,
                                             at::Tensor _kmap,
                                             at::Tensor _kmap_sizes);

using CPUHashMap = HashTableCPU<int, int>;
using CPUHashMap32 = HashTableCPU<int64_t, int>;
