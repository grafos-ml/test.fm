__author__ = 'joaonrb'
import sys
import os
from distutils.core import setup
from distutils.extension import Extension
import pip
try:
    from Cython.Distutils import build_ext
except ImportError:
    pip.main(["install", "cython"])
    from Cython.Distutils import build_ext
try:
    import numpy as np
except ImportError:
    pip.main(["install", "numpy"])
    import numpy as np
from numpy.distutils.system_info import get_info

LIBS = ["/usr/lib/atlas-base/atlas", "/usr/local", "/opt/local", "/usr/lib"]
MAC_GCC_LIB = ["/opt/local/lib/gcc48", "/opt/local/lib/gcc47", "/opt/local/lib/gcc46"]

def find_gcc():
    """
    Find GCC lib
    """
    if sys.platform in ("linux", "linux2"):
        # GCC is the standard library
        return ""
    elif sys.platform == "darwin":
        for path in MAC_GCC_LIB:
            if os.path.isdir(path):
                return path
    raise EnvironmentError("Cannot find gcc library in the system.")

atlas_info = get_info("atlas")
if len(atlas_info) != 0:
    bl_lib = atlas_info["libraries"]
    bl_lib_path = atlas_info["library_dirs"]
    bl_lib_include = atlas_info["include_dirs"]
else:
    blas_info = get_info("blas")
    lapack_info = get_info("lapack")
    if len(blas_info) == 0:
        raise EnvironmentError("Blas library is not detected in the system")
    if len(lapack_info) == 0:
        raise EnvironmentError("Lapack library is not detected in the system")
    bl_lib = set(blas_info["libraries"] + lapack_info["libraries"])
    bl_lib_path = set(blas_info["library_dirs"] + lapack_info["library_dirs"])
    bl_lib_include = set(blas_info["include_dirs"] + lapack_info["include_dirs"])


src = "src/%s"
GCCLIB = os.environ.get("GCCLIB", find_gcc())

ext_modules = [
    Extension("testfm.evaluation.cutil.measures", [src % "testfm/evaluation/cutil/measures.pyx"]),
    Extension("testfm.evaluation.cutil.evaluator", [src % "testfm/evaluation/cutil/evaluator.pyx"],
              extra_compile_args=["-fopenmp"],
              extra_link_args=["-fopenmp"],
              library_dirs=[GCCLIB, "/usr/lib"]),
    Extension("testfm.models.cutil.interface", [src % "testfm/models/cutil/interface.pyx"],
              include_dirs=[np.get_include()]),
    Extension("testfm.models.cutil.float_matrix", [src % "testfm/models/cutil/float_matrix.pyx"],
              libraries=bl_lib,
              library_dirs=bl_lib_path,
              include_dirs=bl_lib_include),
    Extension("testfm.models.cutil.int_array", [src % "testfm/models/cutil/int_array.pyx"]),
    Extension("testfm.models.cutil.tensorcofi", [src % "testfm/models/cutil/tensorcofi.pyx"],
              libraries=bl_lib,
              library_dirs=bl_lib_path,
              include_dirs=bl_lib_include+np.get_include()),
    Extension("testfm.models.cutil.baseline_model", [src % "testfm/models/cutil/baseline_model.pyx"]),
]

setup(
  name = 'testfm',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)