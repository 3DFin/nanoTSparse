#include <Python.h>
#include <torch/library.h>

// CUDA
#include "convolution/convolution_gather_scatter_cuda.h"
#include "hashmap/hashmap_cuda.h"
#include "others/sparsemapping_cuda.h"

// CPU
#include "convolution/convolution_gather_scatter_cpu.h"
#include "hashmap/hashmap_cpu.h"

namespace torchsparse {

extern "C" {
PyObject *PyInit__nanots(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_nanots", NULL, -1, NULL,
  };
  return PyModule_Create(&module_def);
}
}

struct CPUHashTableHolder : torch::CustomClassHolder {
  CPUHashMap map_instance;

  explicit CPUHashTableHolder(int64_t size)
      : map_instance(static_cast<size_t>(size)) {}

  void insert_coords(at::Tensor coords) { map_instance.insert_coords(coords); }

  at::Tensor lookup_coords(at::Tensor coords, at::Tensor kernel_sizes,
                           at::Tensor strides, int64_t kernel_volume) {
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

  void insert_coords(at::Tensor coords) { map_instance.insert_coords(coords); }

  at::Tensor lookup_coords(at::Tensor coords, at::Tensor kernel_sizes,
                           at::Tensor strides, int64_t kernel_volume) {
    return map_instance.lookup_coords(coords, kernel_sizes, strides,
                                      static_cast<int>(kernel_volume));
  }
};

TORCH_LIBRARY(nanots, m) {

  m.class_<GPUHashTableHolder>("GPUHashTable")
      .def(torch::init<int64_t>()) // can't have overloaded init != pybind
      .def("insert_coords", &GPUHashTableHolder::insert_coords)
      .def("lookup_coords", &GPUHashTableHolder::lookup_coords);

  m.class_<CPUHashTableHolder>("CPUHashTable")
      .def(torch::init<int64_t>()) // can't have overloaded init != pybind
      .def("insert_coords", &CPUHashTableHolder::insert_coords)
      .def("lookup_coords", &CPUHashTableHolder::lookup_coords);

  m.def("conv_forward_gather_scatter_cpu("
        "Tensor in_feats, Tensor kernel, Tensor neighbor_maps, "
        "Tensor neighbor_offsets, int output_size, bool transposed) "
        "-> Tensor");

  m.def("conv_backward_gather_scatter_cpu("
        "Tensor in_feats, Tensor grad_out_feats, Tensor kernel, "
        "Tensor neighbor_maps, Tensor neighbor_offsets, bool transposed) -> "
        "Tensor[]");

  m.def("conv_forward_gather_scatter_cuda("
        "Tensor in_feats, Tensor kernel, Tensor neighbor_maps, "
        "int output_size, int conv_mode, Tensor neighbor_offsets,  bool "
        "transposed) "
        "-> Tensor");

  m.def("conv_backward_gather_scatter_cuda("
        "Tensor in_feats, Tensor grad_out_feats, Tensor kernel, "
        "Tensor neighbor_maps, Tensor neighbor_offsets, bool transposed) -> "
        "Tensor[]");

  m.def("build_mask_from_kmap("
        "int n_points, int n_out_points, "
        "Tensor neighbor_maps, Tensor kmap_sizes) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(nanots, CPU, m) {
  m.impl("conv_forward_gather_scatter_cpu", &conv_forward_gather_scatter_cpu);
  m.impl("conv_backward_gather_scatter_cpu", &conv_backward_gather_scatter_cpu);
  m.impl("build_mask_from_kmap", &build_mask_from_kmap_native);
}

TORCH_LIBRARY_IMPL(nanots, CUDA, m) {
  m.impl("conv_forward_gather_scatter_cuda",
         &conv_forward_gather_scatter_cuda_fallback);
  m.impl("conv_backward_gather_scatter_cuda",
         &conv_backward_gather_scatter_cuda);
  m.impl("build_mask_from_kmap", &build_mask_from_kmap);
}

} // namespace torchsparse
