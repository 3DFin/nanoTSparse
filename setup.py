import glob
import os
import subprocess
import sys

import torch
from packaging.version import Version, parse
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


# see https://github.com/Dao-AILab/flash-attention/blob/main/setup.py
def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


with open("nanotsparse/version.py") as f:
    __version__ = f.read().split('"')[1]

print("nanotsparse version:", __version__)

build_ext = BuildExtension.with_options(use_ninja=True)

if (torch.cuda.is_available() and CUDA_HOME is not None) or (os.getenv("FORCE_CUDA", "0") == "1"):
    device = "cuda"
    module_code = "nanotsparse_module_cuda.cu"
else:
    device = "cpu"
    module_code = "nanotsparse_module_cpp.cpp"


base_dir = os.path.join("nanotsparse", "csrc")

sources = [os.path.join(base_dir, module_code)]

for fpath in glob.glob(os.path.join(base_dir, "**", "*")):
    if (fpath.endswith("_cpu.cpp") and device in ["cpu", "cuda"]) or (fpath.endswith("_cuda.cu") and device == "cuda"):
        sources.append(fpath)


# set all dir as include dir
include_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]

# Robin map integration
include_dirs += ["third_party/robin-map/include/"]

# Taskflow integration
taskflow_base_dir = "third_party/taskflow"
include_dirs += [taskflow_base_dir]

extension_type = CUDAExtension if device == "cuda" else CppExtension

cxx_compile_flags = ["-O3", "-std=c++17"]
nvcc_compile_flags = ["-O3", "-std=c++17"]

if sys.platform == "win32" and os.getenv("DISTUTILS_USE_SDK") == "1":
    nvcc_compile_flags += ["-Xcompiler", "/Zc:__cplusplus"]
    cxx_compile_flags = ["/O2", "/std:c++17", "/Zc:__cplusplus"]

    print(CUDA_HOME)
    _, cuda_version = get_cuda_bare_metal_version(CUDA_HOME)

    # ref: torch #148317 and flash-attn #2403, fixes for CUDA 13/CUDA 13+
    if cuda_version is not None and cuda_version >= Version("13.0"):
        nvcc_compile_flags += [
            "-Xcompiler",
            "/Zc:preprocessor",
            "-D_WIN32=1",
            "-DUSE_CUDA=1",
        ]
        cxx_compile_flags += [
            "/Zc:preprocessor",
            "-D_WIN32=1",
            "-DUSE_CUDA=1",
        ]


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

setup(
    name="nanotsparse",
    version=__version__,
    packages=find_packages(),
    ext_modules=[
        extension_type(
            "nanotsparse._nanotsparse",
            sources,
            extra_compile_args={"cxx": cxx_compile_flags, "nvcc": nvcc_compile_flags},
            py_limited_api=True,
        )
    ],
    exclude_package_data={
        "": ["csrc/*"],
    },
    url="https://github.com/3DFin/nanoTorchSparse",
    include_dirs=include_dirs,
    include_package_data=True,
    install_requires=["numpy", "torch"],
    cmdclass={"build_ext": build_ext},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
