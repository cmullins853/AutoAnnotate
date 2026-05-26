import glob
import os
import subprocess
import sys

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_macos_sdk_flags():
    """Return compiler flags needed on macOS when CommandLineTools C++ headers are incomplete."""
    if sys.platform != "darwin":
        return []
    try:
        sdk = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if sdk:
            cxx_include = os.path.join(sdk, "usr", "include", "c++", "v1")
            flags = [f"-I{cxx_include}"]
            return flags
    except Exception:
        pass
    return []


def get_extensions():
    extensions_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "groundingdino", "models", "GroundingDINO", "csrc",
    )

    source_cpu = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)

    sources = source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": get_macos_sdk_flags()}
    define_macros = []

    if CUDA_HOME is not None and torch.cuda.is_available():
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        print("Warning: CUDA not available. Building CPU-only version.")

    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "groundingdino._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="groundingdino",
    version="0.1.0",
    author="IDEA-Research",
    url="https://github.com/IDEA-Research/GroundingDINO",
    description="open-set object detector",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "addict",
        "yapf",
        "timm",
        "numpy",
        "opencv-python",
        "supervision",
        "pycocotools",
    ],
    python_requires=">=3.8",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
