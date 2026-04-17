import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

with open("nanotsparse/version.py") as f:
    __version__ = f.read().split("'")[1]

print("nanotsparse version:", __version__)

build_ext = BuildExtension.with_options(use_ninja=True)

if (torch.cuda.is_available() and CUDA_HOME is not None) or (
    os.getenv("FORCE_CUDA", "0") == "1"
):
    device = "cuda"
    module_code = f"nanotsparse_module_cuda.cu"
else:
    device = "cpu"
    module_code = f"nanotsparse_module_cpp.cpp"


base_dir = os.path.join("nanotsparse", "csrc")

sources = [os.path.join(base_dir, module_code)]

for fpath in glob.glob(os.path.join(base_dir, "**", "*")):
    if (fpath.endswith("_cpu.cpp") and device in ["cpu", "cuda"]) or (
        fpath.endswith("_cuda.cu") and device == "cuda"
    ):
        sources.append(fpath)


# set all dir as include dir
include_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]

# Robin map integration
include_dirs += ["third_party/robin-map/include/"]

# Taskflow integration
taskflow_base_dir = "third_party/taskflow"
include_dirs += [taskflow_base_dir]

extension_type = CUDAExtension if device == "cuda" else CppExtension

# https://en.wikipedia.org/wiki/CUDA
def get_cuda_arch_list():
    if not torch.cuda.is_available():
        return None

    arch_list = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        # Convert major.minor to string format
        arch = f"{props.major}.{props.minor}"
        if arch not in arch_list:
            arch_list.append(arch)

    return ";".join(arch_list)


if device == "cuda" and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    cuda_archs_list = get_cuda_arch_list()
    if cuda_archs_list is not None:
        cuda_archs_list += "+PTX"
        print(f"computed TORCH_CUDA_ARCH_LIST={cuda_archs_list}")
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_archs_list
    else:
        print("Using default CUDA architecture list for build")

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": ["-O3"],
}

setup(
    name="nanotsparse",
    version=__version__,
    packages=find_packages(),
    ext_modules=[
        extension_type(
            "nanotsparse._nanotsparse",
            sources,
            extra_compile_args=extra_compile_args,
            py_limited_api=True
        )
    ],
    exclude_package_data={"": ["csrc/*"],},
    url="https://github.com/3DFin/nanoTorchSparse",
    include_dirs=include_dirs,
    include_package_data=True,
    install_requires=["numpy", "tqdm", "torch"],
    cmdclass={"build_ext": build_ext},
    options={"bdist_wheel": {"py_limited_api": "cp39"}}
)
