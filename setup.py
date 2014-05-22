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


ATLAS_LIB = None
ATLAS_INCLUDE = None
if sys.platform == "linux" or sys.platform == "linux2":
    ATLAS_LIB = "/usr/lib/atlas-base/atlas"
    ATLAS_INCLUDE = "/usr/include/atlas"
elif sys.platform == "darwin":
    ATLAS_LIB = "/usr/lib/atlas"
    ATLAS_INCLUDE = "/usr/include/atlas"
#elif sys.platform == "win32":
#    ATLAS_LIB = "/usr/lib/atlas-base/atlas"
#    ATLAS_INCLUDE = "/usr/include/atlas"

# Any value in env for ATLAS_LIB or ATLAS_INCLUDE will substitute the previous default python
ATLAS_LIB = os.environ.get("ATLAS_LIB", None) or ATLAS_LIB
ATLAS_INCLUDE = os.environ.get("ATLAS_INCLUDE", None) or ATLAS_INCLUDE
if not ATLAS_INCLUDE or not ATLAS_LIB:
    raise OSError("OS not supported yet")


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
              library_dirs=[ATLAS_LIB],
              include_dirs=[ATLAS_INCLUDE]),
    Extension("testfm.models.cutil.int_array", ["src/testfm/models/cutil/int_array.pyx"]),
    Extension("testfm.models.cutil.tensorcofi", ["src/testfm/models/cutil/tensorcofi.pyx"],
              libraries=["cblas", "python2.7"],
              library_dirs=[ATLAS_LIB],
              include_dirs=[ATLAS_INCLUDE, np.get_include()]),
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

