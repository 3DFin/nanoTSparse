#include <Python.h>
#include <torch/library.h>

#include "convolution/convolution_gather_scatter_cpu.h"
#include "hashmap/hashmap_cpu.h"

extern "C" {
PyObject *PyInit__nanots(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_nanots", NULL, -1, NULL,
  };
  return PyModule_Create(&module_def);
}
}

namespace nanots {

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

TORCH_LIBRARY(nanots, m) {

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

  m.def("build_mask_from_kmap("
        "int n_points, int n_out_points, "
        "Tensor neighbor_maps, Tensor kmap_sizes) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(nanots, CPU, m) {
  m.impl("conv_forward_gather_scatter_cpu", &conv_forward_gather_scatter_cpu);
  m.impl("conv_backward_gather_scatter_cpu", &conv_backward_gather_scatter_cpu);
  m.impl("build_mask_from_kmap", &build_mask_from_kmap_native);
}

} // namespace nanots
