#include <torch/serialize/tensor.h>

#include "convolution/convolution_gather_scatter_cpu.h"

#include "hashmap/hashmap_cpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  py::class_<CPUHashMap>(m, "CPUHashTable")
      .def(py::init<>())
      .def(py::init<size_t>())
      .def("insert_coords", &CPUHashMap::insert_coords)
      .def("lookup_coords", &CPUHashMap::lookup_coords);
  m.def("build_mask_from_kmap", &build_mask_from_kmap_native);
  m.def("conv_forward_gather_scatter_cpu", &conv_forward_gather_scatter_cpu);
  m.def("conv_backward_gather_scatter_cpu", &conv_backward_gather_scatter_cpu);
}
