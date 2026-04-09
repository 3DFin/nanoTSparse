#pragma once

#include <torch/all.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

#include <tsl/robin_map.h>

#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/taskflow.hpp>

template <typename coord_type, typename index_type> class HashTableCPU {
private:
  struct VoxelKey {
    std::array<coord_type, 4> coords;

    VoxelKey() = default;
    explicit VoxelKey(const coord_type *ptr) {
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
        hash ^= static_cast<size_t>(key.coords[j]);
        hash *= 1099511628211UL;
      }
      // hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
      return hash;
    }
  };

  tsl::robin_map<VoxelKey, index_type, VoxelKeyHash> hashmap;

public:
  HashTableCPU() = default;
  HashTableCPU(size_t size) { hashmap.reserve(size); }

  HashTableCPU(const at::Tensor &table_keys, const at::Tensor &table_vals) {
    assert(table_keys.is_same_size(table_vals));
    hashmap.reserve(table_keys.size(0));
    auto *key_ptr = table_keys.data_ptr<coord_type>();
    auto *val_ptr = table_vals.data_ptr<index_type>();
    for (size_t i = 0; i < table_keys.size(0); ++i) {
      VoxelKey key(key_ptr + i * 4);
      hashmap[key] = val_ptr[i];
    }
  }

  ~HashTableCPU() = default;

  void insert_coords(const at::Tensor &coords) {
    const auto *key_raw_ptr = coords.data_ptr<coord_type>();
    for (size_t id = 0; id < coords.size(0); ++id) {
      hashmap.emplace(&key_raw_ptr[id * 4], id + 1);
    }
  }

  std::vector<std::array<int, 3>> compute_kernel_offsets(int kx, int ky,
                                                         int kz) {
    std::vector<std::array<int, 3>> result;
    result.reserve(kx * ky * kz);

    const int ox = (kx - 1) / 2;
    const int oy = (ky - 1) / 2;
    const int oz = (kz - 1) / 2;

    for (int z = 0; z < kz; ++z) {
      for (int y = 0; y < ky; ++y) {
        for (int x = 0; x < kx; ++x) {
          result.emplace_back(std::array<int, 3>{{x - ox, y - oy, z - oz}});
        }
      }
    }
    return result;
  }

  at::Tensor lookup_coords(const at::Tensor &coords,
                           const at::Tensor &kernel_sizes,
                           const at::Tensor &strides, int kernel_volume) {
    const auto options =
        at::TensorOptions().dtype(at::ScalarType::Int).device(coords.device());
    auto results = torch::zeros({coords.size(0), kernel_volume}, options);

    auto *results_raw = results.data_ptr<int>();

    const auto *key_raw_ptr = coords.data_ptr<coord_type>();
    const auto *strides_raw = strides.data_ptr<int>();
    const auto *kernel_sizes_ptr = kernel_sizes.data_ptr<int>();

    const auto offsets = compute_kernel_offsets(
        kernel_sizes_ptr[0], kernel_sizes_ptr[1], kernel_sizes_ptr[2]);

    tf::Executor executor;
    tf::Taskflow taskflow;
    taskflow.for_each_index(
        size_t(0), size_t(coords.size(0)), size_t(1), [&](size_t id) {
          const auto *in_coords = &key_raw_ptr[id * 4];
          for (size_t k = 0; k < kernel_volume; ++k) {
            const auto &k_offsets = offsets[k];
            coord_type out_coords[4];
            out_coords[3] = in_coords[3]; // batch is the same
            for (size_t dim = 0; dim < 3; ++dim) {
              out_coords[dim] =
                  in_coords[dim] * strides_raw[dim] + k_offsets[dim];
            }
            const VoxelKey key(&out_coords[0]);
            auto maybe_id = hashmap.find(key);
            if (maybe_id != hashmap.end())
              results_raw[id * kernel_volume + k] =
                  static_cast<int>(maybe_id->second);
          }
        });
    executor.run(taskflow).get();
    return results;
  }
};

using CPUHashMap = HashTableCPU<int, int>;

std::vector<at::Tensor>
build_mask_from_kmap_native(int n_points, int n_out_points,
                            const at::Tensor& kmap, const at::Tensor& kmap_sizes);
