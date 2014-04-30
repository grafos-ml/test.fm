__author__ = 'joaonrb'
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

library_dirs = ['/usr/lib']
include_dirs = ['/usr/include', np.get_include()]

ext_modules = [
    Extension("testfm.models.cutil.interface", ["testfm/models/cutil/interface.pyx"],
              libraries=['blas'],
              library_dirs=library_dirs,
              include_dirs=include_dirs),
    Extension("testfm.models.cutil.float_matrix", ["testfm/models/cutil/float_matrix.pyx"],
              libraries=['blas'],
              library_dirs=library_dirs,
              include_dirs=include_dirs),
    Extension("testfm.evaluation.measures", ["testfm/evaluation/cutil/measures.pyx"]),
]

setup(
  name = 'testfm',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)