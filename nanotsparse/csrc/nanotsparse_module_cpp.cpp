#include <Python.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "convolution/convolution_gather_scatter_cpu.h"
#include "hashmap/hashmap_cpu.h"

PyMODINIT_FUNC PyInit__nanotsparse(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_nanotsparse", NULL, -1, NULL,
  };
  return PyModule_Create(&module_def);
}

namespace nanotsparse {

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

TORCH_LIBRARY(nanotsparse, m) {
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
      "build_mask_from_kmap("
      "int n_points, int n_out_points, "
      "Tensor neighbor_maps, Tensor kmap_sizes) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(nanotsparse, CPU, m) {
  m.impl("conv_forward_gather_scatter_cpu", &conv_forward_gather_scatter_cpu);
  m.impl("conv_backward_gather_scatter_cpu", &conv_backward_gather_scatter_cpu);
  m.impl("build_mask_from_kmap", &build_mask_from_kmap_native);
}

}  // namespace nanotsparse
