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
from Cython.Build import cythonize

LIBS = ["/usr/lib/atlas-base/atlas", "/usr/local", "/opt/local", "/usr/lib"]
GOMPLIB = os.environ.get("GOMPLIB", "")

def search_for_in_all(name, lib_gen):
    """
    Iterate all over lib_gen for the name.
    """
    try:
        return search_for(name, lib_gen.next()) or search_for_in_all(name, lib_gen)
    except StopIteration:
        return None


def search_for(file_name, location):
    """
    Search for the path of file
    """
    result = None
    for root, dirs, files in os.walk(location):
        print "searching at %s" % root
        if file_name in files:
            print " found it"
            result = root
            break
    return result


def find_blas():
    """
    Try to find the blas library file in the standard libraries
    """
    if sys.platform == "linux" or sys.platform == "linux2":
        blas = "libblas.so"
    elif sys.platform == "darwin":
        blas = "libcblas.a"
    else:
        raise OSError("OS not supported yet")
    result = search_for_in_all(blas, (lib for lib in LIBS))
    if result is None:
        raise EnvironmentError("Cannot find %s" % blas)
    return result


def find_lapack():
    """
    Try to find the lapack library file in the standard libraries
    """
    if sys.platform == "linux" or sys.platform == "linux2":
        lapack = "liblapack.so"
    elif sys.platform == "darwin":
        lapack = "liblapack.a"
    else:
        raise OSError("OS not supported yet")
    result = search_for_in_all(lapack, (lib for lib in LIBS))
    if result is None:
        raise EnvironmentError("Cannot find %s" % lapack)
    return result


BLASLIB = os.environ.get("BLASLIB", find_blas())
LAPACKLIB = os.environ.get("LAPACKLIB", find_lapack())

src = "src/%s"
if sys.platform == "darwin":
    pass

ext_modules = [
    Extension("testfm.evaluation.cutil.measures", [src % "testfm/evaluation/cutil/measures.pyx"]),
    Extension("testfm.evaluation.cutil.evaluator", [src % "testfm/evaluation/cutil/evaluator.pyx"],
              extra_compile_args=["-fopenmp"],
              extra_link_args=["-fopenmp"],
              library_dirs=[GOMPLIB, "/usr/lib"]),
    Extension("testfm.models.cutil.interface", [src % "testfm/models/cutil/interface.pyx"],
              include_dirs=[np.get_include()]),
    Extension("testfm.models.cutil.float_matrix", [src % "testfm/models/cutil/float_matrix.pyx"],
              libraries=["lapack", "cblas"],
              library_dirs=[BLASLIB, LAPACKLIB],
              include_dirs=["./include"]),
    Extension("testfm.models.cutil.int_array", [src % "testfm/models/cutil/int_array.pyx"]),
    Extension("testfm.models.cutil.tensorcofi", [src % "testfm/models/cutil/tensorcofi.pyx"],
              libraries=["cblas"],
              library_dirs=[BLASLIB, LAPACKLIB],
              include_dirs=["./include", np.get_include()]),
    Extension("testfm.models.cutil.baseline_model", [src % "testfm/models/cutil/baseline_model.pyx"]),
]

setup(
  name = 'testfm',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)