# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("dtw.pyx")
)
