import glob
import os

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

with open('torchsparse/version.py') as f:
    __version__ = f.read().split("'")[1]

print("torchsparse version:", __version__)

build_ext = BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)

if (torch.cuda.is_available() and CUDA_HOME is not None) or (
    os.getenv("FORCE_CUDA", "0") == "1"
):
    device = "cuda"
    pybind_fn = f"pybind_{device}.cu"
else:
    device = "cpu"
    pybind_fn = f"pybind_{device}.cpp"

device = "cpu"
pybind_fn = f"pybind_{device}.cpp"

base_dir = os.path.join("torchsparse", "backend")

sources = [os.path.join(base_dir, pybind_fn)]

for fpath in glob.glob(os.path.join(base_dir, "**", "*")):
    if (fpath.endswith("_cpu.cpp") and device in ["cpu", "cuda"]) or (
        fpath.endswith("_cuda.cu") and device == "cuda"
    ):
        sources.append(fpath)

# collect header files to include them in sdist
header_files = [file for file in glob.glob(os.path.join(base_dir, "**", "*")) if file.endswith("h")]

# set all dir as include dir
include_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]

extension_type = CUDAExtension if device == "cuda" else CppExtension

extra_compile_args = {
    "cxx": ["-O3", "-fopenmp", "-lgomp"],
    "nvcc": ["-O3"],
}

setup(
    name="torchsparse",
    version=__version__,
    packages=find_packages(),
    ext_modules=[
        extension_type(
            "torchsparse.backend", sources, extra_compile_args=extra_compile_args
        )
    ],
    url="https://github.com/mit-han-lab/torchsparse",
    include_package_data=True,
    include_dirs=include_dirs,
    data_files=header_files,
    install_requires=[
        "ninja",
        "numpy",
        "backports.cached_property",
        "tqdm",
        "typing-extensions",
        "wheel",
        "torch",
        "torchvision"
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False
)
