#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "convolution/convolution_gather_scatter_cpu.h"
#include "devoxelize/devoxelize_cpu.h"
#include "hashmap/hashmap_cpu.h"
#include "others/count_cpu.h"
#include "voxelize/voxelize_cpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  py::class_<CPUHashMap>(m, "CPUHashTable")
      .def(py::init<>())
      .def(py::init<size_t>())
      .def("insert_coords", &CPUHashMap::insert_coords)
      .def("lookup_coords", &CPUHashMap::lookup_coords);
  m.def("build_mask_from_kmap", &build_mask_from_kmap_native);
  m.def("conv_forward_gather_scatter_cpu", &conv_forward_gather_scatter_cpu);
  m.def("conv_backward_gather_scatter_cpu", &conv_backward_gather_scatter_cpu);
  m.def("voxelize_forward_cpu", &voxelize_forward_cpu);
  m.def("voxelize_backward_cpu", &voxelize_backward_cpu);
  m.def("devoxelize_forward_cpu", &devoxelize_forward_cpu);
  m.def("devoxelize_backward_cpu", &devoxelize_backward_cpu);
  m.def("count_cpu", &count_cpu);
}
