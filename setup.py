__author__ = "linas"

import sys
import os
import pip
from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension
from pkg_resources import WorkingSet, DistributionNotFound
working_set = WorkingSet()
try:
    dep = working_set.require("cython")
except DistributionNotFound:
    pip.main(["install", "cython"])
try:
    dep = working_set.require("numpy")
except DistributionNotFound:
    pip.main(["install", "numpy"])
from Cython.Distutils import build_ext
import numpy as np

STD_ATLAS_LIB = "/usr/lib/atlas-base/atlas"
STD_LIB = "/usr/lib"


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
        blas = "libblas.a"
    else:
        raise OSError("OS not supported yet")
    result = search_for(blas, STD_ATLAS_LIB) or search_for(blas, STD_LIB)
    print result
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
    result = search_for(lapack, STD_ATLAS_LIB) or search_for(lapack, STD_LIB)

    if result is None:
        raise EnvironmentError("Cannot find %s" % lapack)
    return result


BLASLIB = os.environ.get("BLASLIB", find_blas())
LAPACKLIB = os.environ.get("LAPACKLIB", find_lapack())


ext_modules = [
    Extension("testfm.evaluation.cutil.measures", ["src/testfm/evaluation/cutil/measures.pyx"],
              libraries=["python2.7"]),
    Extension("testfm.evaluation.cutil.evaluator", ["src/testfm/evaluation/cutil/evaluator.pyx"],
              extra_compile_args=["-fopenmp"],
              extra_link_args=["-fopenmp"]),
    Extension("testfm.models.cutil.interface", ["src/testfm/models/cutil/interface.pyx"],
              include_dirs=[np.get_include()],
              libraries=["python2.7"]),
    Extension("testfm.models.cutil.float_matrix", ["src/testfm/models/cutil/float_matrix.pyx"],
              libraries=["lapack", "cblas", "python2.7"],
              library_dirs=[BLASLIB, LAPACKLIB],
              include_dirs=["./include"]),
    Extension("testfm.models.cutil.int_array", ["src/testfm/models/cutil/int_array.pyx"]),
    Extension("testfm.models.cutil.tensorcofi", ["src/testfm/models/cutil/tensorcofi.pyx"],
              libraries=["cblas", "python2.7"],
              library_dirs=[BLASLIB, LAPACKLIB],
              include_dirs=["./include", np.get_include()]),
    Extension("testfm.models.cutil.baseline_model", ["src/testfm/models/cutil/baseline_model.pyx"],
              libraries=["python2.7"]),
]


def get_requirements():
    with open("conf/requirements.txt") as reqs_file:
        reqs = [x.replace("\n", "").strip() for x in reqs_file if not x.startswith("#")]
        return reqs

setup(
    name="testfm",
    version="1.1",
    description="Experimentation library for Recommender Systems",
    author="L. Baltrunas and J. Baptista",
    author_email="linas.baltrunas@gmail.com",
    url="http://grafos.ml",
    package_dir={"": "src"},
    packages=find_packages("src"),
    test_suite="tests",
    package_data={
        "testfm/lib": ["*.jar"],
        "testfm/data": ["*"],
        "testfm/models/cutil": ["*.pyx", "*.pxd"],
        "testfm/evaluation/cutil": ["*.pyx", "*.pxd"]
    },
    exclude_package_data={"": ["*.pyc", "*.pyo", "*.o", "*.c", "*.h"]},
    license="Apache2",
    include_package_data=True,
    install_requires=get_requirements(),
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)

