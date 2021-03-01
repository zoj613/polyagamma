from distutils.core import Extension
import os
from os.path import join

import numpy as np


source_files = [
    "polyagamma/_polyagamma.c",
    "src/pgm_random.c",
    "src/pgm_alternate.c",
    "src/pgm_devroye.c",
    "src/pgm_common.c",
    "src/pgm_saddle.c",
]


macros = [('NPY_NO_DEPRECATED_API', 0)]
if os.getenv("BUILD_WITH_COVERAGE", None):
    macros.append(('CYTHON_TRACE_NOGIL', 1))

# https://numpy.org/devdocs/reference/random/examples/cython/setup.py.html
include_path = np.get_include()
extensions = [
    Extension(
        "_polyagamma",
        source_files,
        include_dirs=[include_path, "./include"],
        library_dirs=[join(include_path, '..', '..', 'random', 'lib')],
        libraries=['npyrandom'],
        define_macros=macros,
        extra_compile_args=['-std=c99']
    ),
]

#os.system("cp -r include polyagamma/")
def build(setup_kwargs):
    """Build extension modules."""
    setup_kwargs.update(ext_modules=extensions)
