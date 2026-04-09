#include <torch/serialize/tensor.h>

#include "convolution/convolution_gather_scatter_cpu.h"
#include "convolution/convolution_gather_scatter_cuda.h"

#include "convolution/convolution_forward_implicit_gemm_cuda.h"
#include "convolution/convolution_forward_implicit_gemm_sorted_cuda.h"
#include "convolution/convolution_backward_wgrad_implicit_gemm_cuda.h"
#include "convolution/convolution_backward_wgrad_implicit_gemm_sorted_cuda.h"

#include "others/downsample_cuda.h"
#include "others/exclusive_scan_cuda.h"
#include "others/reduce_bitmask_cuda.h"
#include "others/reorder_map_cuda.h"
#include "others/sparsemapping_cuda.h"
#include "others/query_cuda.h"
#include "hashmap/hashmap_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<GPUHashMap>(m, "GPUHashTable")
        .def(py::init<const int>())
        .def(py::init<at::Tensor, at::Tensor>())
        .def("insert_vals", &GPUHashMap::insert_vals)
        .def("lookup_vals", &GPUHashMap::lookup_vals)
        .def("insert_coords", &GPUHashMap::insert_coords)
        .def("lookup_coords", &GPUHashMap::lookup_coords);

  // TODO: add CPUHashmap

  // Gather Scatter Sum
  m.def("conv_forward_gather_scatter_cpu", &conv_forward_gather_scatter_cpu);
  m.def("conv_backward_gather_scatter_cpu", &conv_backward_gather_scatter_cpu);
  m.def("conv_forward_gather_scatter_cuda", &conv_forward_gather_scatter_cuda);
  m.def("conv_backward_gather_scatter_cuda", &conv_backward_gather_scatter_cuda);

  // for Gather Scatter
  m.def("build_mask_from_kmap", &build_mask_from_kmap);
  // TODO: add CPUHash

  // ImplicitGEMM
  m.def("conv_forward_implicit_gemm_cuda", &conv_forward_implicit_gemm_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("num_out_feats"),py::arg("num_out_channels"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);
  m.def("conv_forward_implicit_gemm_sorted_cuda", &conv_forward_implicit_gemm_sorted_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("_reduced_mask"), py::arg("_reorder_loc"), py::arg("num_out_feats"), py::arg("num_out_channels"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);
  m.def("conv_backward_wgrad_implicit_gemm_cuda", &conv_backward_wgrad_implicit_gemm_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("split_k_iters"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);
  m.def("conv_backward_wgrad_implicit_gemm_sorted_cuda", &conv_backward_wgrad_implicit_gemm_sorted_cuda, py::arg("_in_feats"), py::arg("_kernel"), py::arg("_out_in_map"), py::arg("_reduced_mask"), py::arg("_reorder_loc"), py::arg("split_k_iters"), py::arg("allow_tf32") = false, py::arg("allow_fp16") = true);

  m.def("convert_transposed_out_in_map", &convert_transposed_out_in_map);
  m.def("derive_bitmask_from_out_in_map", &derive_bitmask_from_out_in_map);

  m.def("reduce_bitmask_cuda", &reduce_bitmask_cuda);
  m.def("reorder_out_in_map_cuda", &reorder_out_in_map_cuda);

  m.def("build_kernel_map_subm_hashmap", &build_kernel_map_subm_hashmap);
  m.def("build_kernel_map_downsample_hashmap", &build_kernel_map_downsample_hashmap);

  m.def("exclusive_scan_quantified_wrapper", &exclusive_scan_quantified_wrapper);
  m.def("downsample_cuda", &downsample_cuda); // used in implicit GEMM
}
