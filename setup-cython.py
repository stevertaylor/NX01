# setup script to compile jitter module; 
# written by M. Vallisneri (2015)
#
# use python setup.py build_ext --inplace to test

import distutils.core as D
import numpy as N

from distutils.core import setup, Extension
from distutils      import sysconfig

try:
    numpy_include = N.get_include()
except AttributeError:
    numpy_include = N.get_numpy_include()

# need to replace build_ext to build cython extension
import Cython.Distutils

extension = D.Extension('NX01_jitter',
                        sources = ['NX01_jitter.pyx'],
                        include_dirs = [numpy_include],
                        extra_compile_args = ['-std=c99']
                       )

D.setup(name = 'NX01_jitter',
        ext_modules = [extension],
        cmdclass = {"build_ext": Cython.Distutils.build_ext}
        )
