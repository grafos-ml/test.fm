__author__ = 'joaonrb'
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

library_dirs = ["/usr/lib"]
include_dirs = ["/usr/include/atlas", np.get_include()]

ext_modules = [
    Extension("testfm.models.cutil.interface", ["testfm/models/cutil/interface.pyx"],
              include_dirs=include_dirs),
    Extension("testfm.models.cutil.float_matrix", ["testfm/models/cutil/float_matrix.pyx"],
              libraries=["lapack", "cblas"],
              library_dirs=["/usr/lib/atlas-base/atlas"],
              include_dirs=include_dirs),
    Extension("testfm.models.cutil.int_array", ["testfm/models/cutil/int_array.pyx"]),
    Extension("testfm.evaluation.cutil.measures", ["testfm/evaluation/cutil/measures.pyx"]),
    Extension("testfm.models.cutil.tensorcofi", ["testfm/models/cutil/tensorcofi.pyx"],
              libraries=["cblas"],
              library_dirs=["/usr/lib/atlas-base/atlas"],
              include_dirs=include_dirs),
    Extension("testfm.evaluation.cutil.evaluator", ["testfm/evaluation/cutil/evaluator.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'])
]

setup(
  name = 'testfm',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)