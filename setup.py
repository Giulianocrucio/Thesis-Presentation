# setup.py
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        name="slg2lib",
        sources=["src/data/slg2.cpp"],
        # Standard optimization flags
        extra_compile_args=["-O3", "-Wall", "-std=c++17"],
    ),
]

setup(
    name="slg2lib",
    version="0.2.0",
    author="Thesis Project",
    description="PyTorch C++ extension for SLG2 topology",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)