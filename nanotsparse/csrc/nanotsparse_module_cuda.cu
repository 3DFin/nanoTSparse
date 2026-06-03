#include <Python.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

// CUDA
#include "convolution/convolution_backward_wgrad_implicit_gemm_cuda.h"
#include "convolution/convolution_backward_wgrad_implicit_gemm_sorted_cuda.h"
#include "convolution/convolution_forward_implicit_gemm_cuda.h"
#include "convolution/convolution_forward_implicit_gemm_sorted_cuda.h"
#include "convolution/convolution_gather_scatter_cuda.h"
#include "hashmap/hashmap_cuda.h"
#include "others/downsample_cuda.h"
#include "others/query_cuda.h"
#include "others/reduce_bitmask_cuda.h"
#include "others/reorder_map_cuda.h"
#include "others/sparsemapping_cuda.h"

// CPU
#include "convolution/convolution_gather_scatter_cpu.h"
#include "hashmap/hashmap_cpu.h"

namespace nanotsparse {

PyMODINIT_FUNC PyInit__nanotsparse(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_nanotsparse", NULL, -1, NULL,
  };
  return PyModule_Create(&module_def);
}

struct CPUHashTableHolder : torch::CustomClassHolder {
  CPUHashMap map_instance;

  explicit CPUHashTableHolder(int64_t size)
      : map_instance(static_cast<size_t>(size)) {}

  void insert_coords(at::Tensor coords) {
    TORCH_CHECK(coords.dim() == 2, "coords must be a 2D tensor");
    TORCH_CHECK(coords.size(1) == 4,
                "coords must have 4 columns (x,y,z,batch)");
    TORCH_CHECK(coords.scalar_type() == at::ScalarType::Int,
                "coords must be an Int tensor");
    TORCH_CHECK(coords.numel() > 0, "coords tensor must not be empty");
    map_instance.insert_coords(coords);
  }

  at::Tensor lookup_coords(at::Tensor coords, at::Tensor kernel_sizes,
                           at::Tensor strides, int64_t kernel_volume) {
    TORCH_CHECK(coords.dim() == 2, "coords must be a 2D tensor");
    TORCH_CHECK(coords.size(1) == 4,
                "coords must have 4 columns (x,y,z,batch)");
    TORCH_CHECK(coords.scalar_type() == at::ScalarType::Int,
                "coords must be an Int tensor");
    TORCH_CHECK(coords.numel() > 0, "coords tensor must not be empty");

    TORCH_CHECK(kernel_sizes.dim() == 1, "kernel_sizes must be a 1D tensor");
    TORCH_CHECK(kernel_sizes.size(0) == 3,
                "kernel_sizes must have 3 elements (x,y,z)");
    TORCH_CHECK(kernel_sizes.scalar_type() == at::ScalarType::Int,
                "kernel_sizes must be an Int tensor");

    TORCH_CHECK(strides.dim() == 1, "strides must be a 1D tensor");
    TORCH_CHECK(strides.size(0) == 3, "strides must have 3 elements (x,y,z)");
    TORCH_CHECK(strides.scalar_type() == at::ScalarType::Int,
                "strides must be an Int tensor");

    TORCH_CHECK(kernel_volume > 0, "kernel_volume must be positive");

    return map_instance.lookup_coords(coords, kernel_sizes, strides,
                                      static_cast<int>(kernel_volume));
  }
};

// TODO try replace by CUCO
// https://github.com/NVIDIA/cuCollections
struct GPUHashTableHolder : torch::CustomClassHolder {
  GPUHashMap map_instance;

  // beware Size is the size of the table, not the number of entries
  explicit GPUHashTableHolder(int64_t capacity)
      : map_instance(static_cast<int>(capacity)) {}

  void insert_coords(at::Tensor coords) {
    TORCH_CHECK(coords.dim() == 2, "coords must be a 2D tensor");
    TORCH_CHECK(coords.size(1) == 4,
                "coords must have 4 columns (x,y,z,batch)");
    TORCH_CHECK(coords.scalar_type() == at::ScalarType::Int,
                "coords must be an Int tensor");
    TORCH_CHECK(coords.numel() > 0, "coords tensor must not be empty");
    TORCH_CHECK(coords.device().is_cuda(), "coords must be a CUDA tensor");
    map_instance.insert_coords(coords);
  }

  at::Tensor lookup_coords(at::Tensor coords, at::Tensor kernel_sizes,
                           at::Tensor strides, int64_t kernel_volume) {
    TORCH_CHECK(coords.dim() == 2, "coords must be a 2D tensor");
    TORCH_CHECK(coords.size(1) == 4,
                "coords must have 4 columns (x,y,z,batch)");
    TORCH_CHECK(coords.scalar_type() == at::ScalarType::Int,
                "coords must be an Int tensor");
    TORCH_CHECK(coords.numel() > 0, "coords tensor must not be empty");
    TORCH_CHECK(coords.device().is_cuda(), "coords must be a CUDA tensor");

    TORCH_CHECK(kernel_sizes.dim() == 1, "kernel_sizes must be a 1D tensor");
    TORCH_CHECK(kernel_sizes.size(0) == 3,
                "kernel_sizes must have 3 elements (x,y,z)");
    TORCH_CHECK(kernel_sizes.scalar_type() == at::ScalarType::Int,
                "kernel_sizes must be an Int tensor");
    TORCH_CHECK(kernel_sizes.device().is_cuda(),
                "kernel_sizes must be a CUDA tensor");

    TORCH_CHECK(strides.dim() == 1, "strides must be a 1D tensor");
    TORCH_CHECK(strides.size(0) == 3, "strides must have 3 elements (x,y,z)");
    TORCH_CHECK(strides.scalar_type() == at::ScalarType::Int,
                "strides must be an Int tensor");
    TORCH_CHECK(strides.device().is_cuda(), "strides must be a CUDA tensor");

    TORCH_CHECK(kernel_volume > 0, "kernel_volume must be positive");

    return map_instance.lookup_coords(coords, kernel_sizes, strides,
                                      static_cast<int>(kernel_volume));
  }
};

static std::vector<at::Tensor> build_kernel_map_subm_impl(
    const c10::intrusive_ptr<GPUHashTableHolder>& hash_table,
    at::Tensor in_coords, at::Tensor coords_min, at::Tensor coords_max,
    at::Tensor kernel_sizes, at::Tensor stride, at::Tensor padding,
    bool to_insert) {
  return build_kernel_map_subm_hashmap(hash_table->map_instance, in_coords,
                                       coords_min, coords_max, kernel_sizes,
                                       stride, padding, to_insert);
}

static std::vector<at::Tensor> build_kernel_map_downsample_impl(
    const c10::intrusive_ptr<GPUHashTableHolder>& hash_table,
    at::Tensor in_coords, at::Tensor coords_min, at::Tensor coords_max,
    at::Tensor kernel_sizes, at::Tensor stride, at::Tensor padding,
    bool to_insert) {
  return build_kernel_map_downsample_hashmap(
      hash_table->map_instance, in_coords, coords_min, coords_max, kernel_sizes,
      stride, padding, to_insert);
}

TORCH_LIBRARY(nanotsparse, m) {
  m.class_<GPUHashTableHolder>("GPUHashTable")
      .def(torch::init<int64_t>())  // can't have overloaded init != pybind
      .def("insert_coords", &GPUHashTableHolder::insert_coords)
      .def("lookup_coords", &GPUHashTableHolder::lookup_coords);

  m.class_<CPUHashTableHolder>("CPUHashTable")
      .def(torch::init<int64_t>())  // can't have overloaded init != pybind
      .def("insert_coords", &CPUHashTableHolder::insert_coords)
      .def("lookup_coords", &CPUHashTableHolder::lookup_coords);

  m.def(
      "conv_forward_gather_scatter_cpu("
      "Tensor in_feats, Tensor kernel, Tensor neighbor_maps, "
      "Tensor neighbor_offsets, int output_size, bool transposed) "
      "-> Tensor");

  m.def(
      "conv_backward_gather_scatter_cpu("
      "Tensor in_feats, Tensor grad_out_feats, Tensor kernel, "
      "Tensor neighbor_maps, Tensor neighbor_offsets, bool transposed) -> "
      "Tensor[]");

  m.def(
      "conv_forward_gather_scatter_cuda("
      "Tensor in_feats, Tensor kernel, Tensor neighbor_maps, "
      "int output_size, int conv_mode, Tensor neighbor_offsets,  bool "
      "transposed) "
      "-> Tensor");

  m.def(
      "conv_backward_gather_scatter_cuda("
      "Tensor in_feats, Tensor grad_out_feats, Tensor kernel, "
      "Tensor neighbor_maps, Tensor neighbor_offsets, bool transposed) -> "
      "Tensor[]");

  m.def(
      "conv_forward_implicit_gemm_cuda("
      "Tensor _in_feats, Tensor _kernel, Tensor _out_in_map, "
      "int num_out_feats, int num_out_channels, "
      "bool allow_tf32=False, bool allow_fp16=True) -> Tensor");

  m.def(
      "conv_backward_wgrad_implicit_gemm_cuda("
      "Tensor _in_feats, Tensor _kernel, "
      "Tensor _out_in_map, int split_k_iters, "
      "bool allow_tf32=False, bool allow_fp16=True) -> Tensor");

  m.def(
      "conv_forward_implicit_gemm_sorted_cuda("
      "Tensor _in_feats, Tensor _kernel, Tensor _out_in_map, "
      "Tensor _reduced_mask, Tensor _reorder_loc, "
      "int num_out_feats, int num_out_channels, "
      "bool allow_tf32=False, bool allow_fp16=True) -> Tensor");

  m.def(
      "conv_backward_wgrad_implicit_gemm_sorted_cuda("
      "Tensor _in_feats, Tensor _kernel, Tensor _out_in_map, "
      "Tensor _reduced_mask, Tensor _reorder_loc, int split_k_iters, "
      "bool allow_tf32=False, bool allow_fp16=True) -> Tensor");

  m.def(
      "build_mask_from_kmap("
      "int n_points, int n_out_points, "
      "Tensor neighbor_maps, Tensor kmap_sizes) -> Tensor[]");

  m.def(
      "build_kernel_map_subm_hashmap("
      "__torch__.torch.classes.nanotsparse.GPUHashTable table, "
      "Tensor in_coords, Tensor coords_min, Tensor coords_max, "
      "Tensor kernel_sizes, Tensor stride, Tensor padding, "
      "bool to_insert) -> Tensor[]");

  m.def(
      "build_kernel_map_downsample_hashmap("
      "__torch__.torch.classes.nanotsparse.GPUHashTable table, "
      "Tensor in_coords, Tensor coords_min, Tensor coords_max, "
      "Tensor kernel_sizes, Tensor stride, Tensor padding, "
      "bool to_insert) -> Tensor[]");

  m.def(
      "derive_bitmask_from_out_in_map("
      "Tensor out_in_map, int split_mask_num, int valid_n) -> Tensor");
  m.def(
      "reorder_out_in_map_cuda("
      "Tensor out_in_map, Tensor reorder_loc) -> Tensor");
  m.def("reduce_bitmask_cuda(Tensor bitmask, int M_tile) -> Tensor");
  m.def("convert_transposed_out_in_map(Tensor out_in_map, int size) -> Tensor");
  m.def(
      "downsample_cuda("
      "Tensor in_coords, Tensor coords_max, Tensor coords_min, "
      "Tensor kernel_sizes, Tensor stride, Tensor padding) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanotsparse, CPU, m) {
  m.impl("conv_forward_gather_scatter_cpu", &conv_forward_gather_scatter_cpu);
  m.impl("conv_backward_gather_scatter_cpu", &conv_backward_gather_scatter_cpu);
  m.impl("build_mask_from_kmap", &build_mask_from_kmap_native);
}

TORCH_LIBRARY_IMPL(nanotsparse, CUDA, m) {
  m.impl("conv_forward_gather_scatter_cuda",
         &conv_forward_gather_scatter_cuda_fallback);
  m.impl("conv_backward_gather_scatter_cuda",
         &conv_backward_gather_scatter_cuda);
  m.impl("conv_forward_implicit_gemm_cuda", &conv_forward_implicit_gemm_cuda);
  m.impl("conv_backward_wgrad_implicit_gemm_cuda",
         &conv_backward_wgrad_implicit_gemm_cuda);
  m.impl("conv_forward_implicit_gemm_sorted_cuda",
         &conv_forward_implicit_gemm_sorted_cuda);
  m.impl("conv_backward_wgrad_implicit_gemm_sorted_cuda",
         &conv_backward_wgrad_implicit_gemm_sorted_cuda);

  m.impl("build_mask_from_kmap", &build_mask_from_kmap);
  m.impl("build_kernel_map_subm_hashmap", &build_kernel_map_subm_impl);
  m.impl("build_kernel_map_downsample_hashmap",
         &build_kernel_map_downsample_impl);
  m.impl("derive_bitmask_from_out_in_map", &derive_bitmask_from_out_in_map);
  m.impl("reorder_out_in_map_cuda", &reorder_out_in_map_cuda);
  m.impl("reduce_bitmask_cuda", &reduce_bitmask_cuda);
  m.impl("convert_transposed_out_in_map", &convert_transposed_out_in_map);

  m.impl("downsample_cuda", &downsample_cuda);
}

}  // namespace nanotsparse
