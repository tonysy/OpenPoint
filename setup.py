#
# @Author: Songyang Zhang 
# @Date: 2019-01-24 21:48:09 
# @Last Modified by:   Songyang Zhang 
# @Last Modified time: 2019-01-24 21:48:09 
#

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extension_dir = os.path.join(this_dir, 'openpoint', 'csrc')

    main_file = glob.glob(os.path.join(extension_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extension_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extension_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx":[]}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extension_dir, s) for s in sources]

    include_dirs = [extension_dir]

    ext_modules = [
        extension(
            "openpoint._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name='openpoint',
    varsion='0.1',
    author='Songyang Zhang',
    url='https://github.com/tonysy/openpoint',
    description='point classification and semantic segmentation in PyTorch',
    packages=find_packages(exclude=("configs", "tests")),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension}
)

