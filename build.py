from distutils.core import Extension
import platform
import os
from os.path import join

import numpy as np
from numpy.distutils.misc_util import get_info


source_files = [
    "polyagamma/_polyagamma.c",
    "src/pgm_random.c",
    "src/pgm_alternate.c",
    "src/pgm_devroye.c",
    "src/pgm_common.c",
    "src/pgm_saddle.c",
    "src/pgm_density.c",
]


macros = [('NPY_NO_DEPRECATED_API', 0)]
if os.getenv("BUILD_WITH_COVERAGE", None):
    macros.append(('CYTHON_TRACE_NOGIL', 1))

if platform.system() == 'Windows':
    compile_args = ['/O2']
else:
    compile_args = ['-O2', '-std=c99']

# https://numpy.org/devdocs/reference/random/examples/cython/setup.py.html
include_path = np.get_include()
lib_dirs = [
    join(include_path, '..', '..', 'random', 'lib'),
    *get_info('npymath')['library_dirs']
]
extensions = [
    Extension(
        "_polyagamma",
        source_files,
        include_dirs=[include_path, "./include"],
        library_dirs=lib_dirs,
        libraries=['npyrandom', 'npymath'],
        define_macros=macros,
        extra_compile_args=compile_args,
    ),
]


def build(setup_kwargs):
    """Build extension modules."""
    setup_kwargs.update(ext_modules=extensions, zip_safe=False)
