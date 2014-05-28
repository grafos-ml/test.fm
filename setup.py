__author__ = "linas"

import pip
from distutils.core import setup
from setuptools import find_packages
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
from compile_c_modules import ext_modules


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
    exclude_package_data={"": ["*.pyc", "*.pyo", "*.o", "*.so", "*.c", "*.h"]},
    license="Apache2",
    include_package_data=True,
    install_requires=get_requirements(),
    #cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)

