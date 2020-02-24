# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize("dtw.pyx")
)
