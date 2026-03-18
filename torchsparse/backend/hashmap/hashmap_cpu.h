#pragma once

#include <unordered_map>
#include <vector>

#include <torch/extension.h>

#include <iostream>

template <typename key_type, typename val_type> class HashTableCPU {
private:
  std::unordered_map<key_type, val_type> hashmap;

public:

  HashTableCPU() {}

  HashTableCPU(torch::Tensor table_keys, torch::Tensor table_vals) {
    assert(table_keys.is_same_size(table_vals));
    hashmap.reserve(table_keys.size(0));
    for (size_t i = 0; i < table_keys.size(0); ++i) {
      hashmap[*table_keys[i].data_ptr<key_type>()] =
          *table_vals[i].data_ptr<val_type>();
    }
  }

  ~HashTableCPU() = default;

  // FNV HASH of 4uples (b,x,y,z)
  static key_type hash_func_64b(key_type *data) {
    uint64_t hash = 14695981039346656037UL;
    for (int j = 0; j < 4; j++) {
      hash ^= (unsigned int)data[j];
      hash *= 1099511628211UL;
    }
    // hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
    return hash;
  }

  // virtually used nowhere
  void insert_vals(torch::Tensor keys) {
    val_type id = 0;
    for (size_t i = 0; i < keys.size(0); ++i) {
      hashmap[*keys[i].data_ptr<key_type>()] = id + 1;
    }
  }

  torch::Tensor lookup_vals(torch::Tensor keys) {
    auto options =
        torch::TensorOptions().dtype(at::ScalarType::Int).device(keys.device());
    at::Tensor results = torch::zeros({(keys.size(0) + padding_divisor - 1) /
                                       padding_divisor * padding_divisor},
                                      options);
    for (size_t i = 0; i < keys.size(0); ++i) {
      results[i] = hashmap[*keys[i].data_ptr<key_type>()];
    }
    return results;
  }

  // coords are 4uples keys (b,x,y,z)
  void insert_coords(torch::Tensor coords) {
    auto *key_raw_ptr = coords.data_ptr<key_type>();
    for (size_t id = 0; id < coords.size(0); ++id) {
      key_type key = static_cast<key_type>(hash_func_64b(key_raw_ptr + id * 4));
      //std::cout << key <<  " " << id << " " << key_raw_ptr[id * 4] << " " << key_raw_ptr[id * 4 + 1] << " " << key_raw_ptr[id * 4 + 2] << " " << key_raw_ptr[id * 4 +3] << std::endl;
      hashmap[key] = id + 1;
    }
  }

  std::vector<std::array<int, 3>> compute_kernel_offsets(int kx, int ky,
                                                         int kz) {
    int kernel_volume = kx * ky * kz;

    std::vector<std::array<int, 3>> result(kernel_volume);

    int id = 0;
    for (int x = 0; x < kx; x++) {
      for (int y = 0; y < ky; y++) {
        for (int z = 0; z < kz; z++) {

          result[id][0] = x - (kx - 1) / 2;
          result[id][1] = y - (ky - 1) / 2;
          result[id][2] = z - (kz - 1) / 2;

          id++;
        }
      }
    }

    return result;
  }

  // coords are keys
  torch::Tensor lookup_coords(at::Tensor coords, at::Tensor kernel_sizes,
                              at::Tensor strides, int kernel_volume) {
    auto options = torch::TensorOptions()
                       .dtype(at::ScalarType::Int)
                       .device(coords.device());
    at::Tensor results = torch::zeros({coords.size(0),
                                       kernel_volume},
                                      options);

    auto *key_raw_ptr = coords.data_ptr<key_type>();
    auto *strides_raw = strides.data_ptr<int>();
    auto *results_raw = results.data_ptr<int>();

    auto offsets = compute_kernel_offsets(*kernel_sizes[0].data_ptr<int>(),
                                          *kernel_sizes[1].data_ptr<int>(),
                                          *kernel_sizes[2].data_ptr<int>());

    for (size_t id = 0; id < coords.size(0); ++id) {
      const auto *in_coords = key_raw_ptr + id * 4;

      for (size_t k = 0; k < kernel_volume; k++) {
        key_type out_coords[4];

        // batch is the same
        out_coords[3] = in_coords[3];

        const auto &k_offsets = offsets[k];
        for (size_t dim = 0; dim < 3; ++dim) {
          out_coords[dim] = in_coords[dim] * strides_raw[dim] + k_offsets[dim];
        }
        key_type key = static_cast<key_type>(hash_func_64b(out_coords));

        auto maybe_id = hashmap.find(key);
        if (maybe_id != hashmap.end()) {
          results_raw[id * kernel_volume + k] = maybe_id->second;
        } else {
          results_raw[id * kernel_volume + k] = 0;
        }
      }
    }
    return results;
  }

private:
  int padding_divisor = 128;
};

using CPUHashMap = HashTableCPU<int64_t, int>;
using CPUHashMap32 = HashTableCPU<int, int>;
