import platform
import os
from os.path import join
from setuptools import Extension, setup

import numpy as np


source_files = [
    "polyagamma/_polyagamma.c",
    "src/pgm_random.c",
    "src/pgm_alternate.c",
    "src/pgm_devroye.c",
    "src/pgm_common.c",
    "src/pgm_saddle.c",
    "src/pgm_density.c",
]

macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
if os.getenv("BUILD_WITH_COVERAGE", None):
    macros.append(('CYTHON_TRACE_NOGIL', 1))

if platform.system() == 'Windows':
    compile_args = ['/O2']
else:
    compile_args = ['-O2', '-std=c99']

# https://numpy.org/devdocs/reference/random/examples/cython/setup.py.html
npy_include_path = np.get_include()
lib_dirs = [
    join(npy_include_path, '..', '..', 'random', 'lib'),
    join(npy_include_path, '..', 'lib')
]
extensions = [
    Extension(
        "polyagamma._polyagamma",
        sources=source_files,
        include_dirs=[npy_include_path, "./include"],
        library_dirs=lib_dirs,
        libraries=['npyrandom', 'npymath'],
        define_macros=macros,
        extra_compile_args=compile_args,
    ),
]


setup(ext_modules=extensions)
