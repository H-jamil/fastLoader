from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        "fastloader.core._fastloader",  # Changed module path
        ["src/python/bindings.cpp",
         "src/FileSystemBackend.cpp",
         "src/Prefetcher.cpp"],
        include_dirs=[
            "include",
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++',
        extra_compile_args=['-std=c++17']
    ),
]

setup(
    name="fastloader",
    version="0.1",
    author="Jamil Hasibul",
    description="A fast data loading library",
    packages=['fastloader', 'fastloader.core'],  # Add core package
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4.3', 'numpy'],
    setup_requires=['pybind11>=2.4.3'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.6'
)