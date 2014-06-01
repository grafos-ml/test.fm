__author__ = "linas"

import pip
import os
import sys
import platform
from distutils.core import setup
from setuptools import find_packages
from compile_c_modules import ext_modules, build_ext
try:
    from Cython.Build import cythonize
except ImportError:
    if sys.platform == "darwin" and platform.mac_ver()[0] >= 10.9:
        os.environ["CFLAGS"] = "-Wunused-command-line-argument-hard-error-in-future"
    pip.main(["install", "cython"])
    from Cython.Build import cythonize


def get_requirements():
    with open("conf/requirements.txt") as reqs_file:
        reqs = [x.replace("\n", "").strip() for x in reqs_file if not x.startswith("#")]
        return reqs

packages = find_packages("src")
packages.remove("tests")


if sys.platform == "darwin":
    os.environ["CFLAGS"] = "-arch i386 -arch x86_64"
    os.environ["FFLAGS"] = "-m32 -m64"
    os.environ["LDFLAGS"] = "-Wall -undefined dynamic_lookup -bundle -arch i386 -arch x86_64"

setup(
    name="testfm",
    version="1.1.2",
    description="Experimentation library for Recommender Systems",
    author="L. Baltrunas and J. Baptista",
    author_email="linas.baltrunas@gmail.com",
    url="http://grafos.ml",
    package_dir={"": "src"},
    packages=packages,
    test_suite="tests",
    package_data={
        "testfm/lib": ["*.jar"],
        "testfm/data": ["*"],
        "testfm/models/cutil": ["*.pyx", "*.pxd"],
        "testfm/evaluation/cutil": ["*.pyx", "*.pxd"]
    },
    exclude_package_data={"": ["*.pyc", "*.pyo", "*.o", "*.so", "*.c", "*.h"]},
    license="Apache2",
    include_package_data=True,
    install_requires=get_requirements(),
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(ext_modules)
)

