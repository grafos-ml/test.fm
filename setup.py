__author__ = "linas"

#import os
#import sys
#import platform
#if sys.platform == "darwin" and platform.mac_ver()[0] >= 10.9:
#    os.environ["ARCHFLAGS"] = "-Wno-error=unused-command-line-argument-hard-error-in-future"
#import pip
from setuptools import setup
from setuptools import find_packages
from compile import ext_modules, build_ext


def get_requirements():
    with open("conf/requirements.txt") as reqs_file:
        reqs = [x.replace("\n", "").strip() for x in reqs_file if not x.startswith("#")]
        return reqs

packages = find_packages("src")


setup(
    name="testfm",
    version="1.2",
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
    ext_modules=ext_modules
)

