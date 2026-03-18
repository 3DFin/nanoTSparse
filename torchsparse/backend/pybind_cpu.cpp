#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "convolution/convolution_gather_scatter_cpu.h"
#include "devoxelize/devoxelize_cpu.h"
#include "hash/hash_cpu.h"
#include "hashmap/hashmap_cpu.h"
#include "others/count_cpu.h"
#include "others/query_cpu.h"
#include "voxelize/voxelize_cpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  py::class_<CPUHashMap>(m, "CPUHashTable")
      .def(py::init<torch::Tensor, torch::Tensor>())
      .def(py::init<>())
      .def("insert_vals", &CPUHashMap::insert_vals)
      .def("lookup_vals", &CPUHashMap::lookup_vals)
      .def("insert_coords", &CPUHashMap::insert_coords)
      .def("lookup_coords", &CPUHashMap::lookup_coords);
  py::class_<CPUHashMap32>(m, "CPUHashTable32")
      .def(py::init<torch::Tensor, torch::Tensor>())
      .def(py::init<>())
      .def("insert_vals", &CPUHashMap32::insert_vals)
      .def("lookup_vals", &CPUHashMap32::lookup_vals)
      .def("insert_coords", &CPUHashMap32::insert_coords)
      .def("lookup_coords", &CPUHashMap32::lookup_coords);
  m.def("conv_forward_gather_scatter_cpu", &conv_forward_gather_scatter_cpu);
  m.def("conv_backward_gather_scatter_cpu", &conv_backward_gather_scatter_cpu);
  m.def("voxelize_forward_cpu", &voxelize_forward_cpu);
  m.def("voxelize_backward_cpu", &voxelize_backward_cpu);
  m.def("devoxelize_forward_cpu", &devoxelize_forward_cpu);
  m.def("devoxelize_backward_cpu", &devoxelize_backward_cpu);
  m.def("hash_cpu", &hash_cpu);
  m.def("kernel_hash_cpu", &kernel_hash_cpu);
  m.def("hash_query_cpu", &hash_query_cpu);
  m.def("count_cpu", &count_cpu);
}
