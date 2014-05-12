__author__ = 'linas'

from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

library_dirs = ["/usr/lib"]
include_dirs = ["/usr/include/atlas", np.get_include(), "/usr/local/atlas/include"]

ext_modules = [
    Extension("testfm.models.cutil.interface", ["src/testfm/models/cutil/interface.pyx"],
              include_dirs=include_dirs),
    Extension("testfm.models.cutil.float_matrix", ["src/testfm/models/cutil/float_matrix.pyx"],
              libraries=["lapack", "cblas"],
              library_dirs=["/usr/lib/atlas-base/atlas"],
              include_dirs=include_dirs),
    Extension("testfm.models.cutil.int_array", ["src/testfm/models/cutil/int_array.pyx"]),
    Extension("testfm.evaluation.cutil.measures", ["src/testfm/evaluation/cutil/measures.pyx"]),
    Extension("testfm.models.cutil.tensorcofi", ["src/testfm/models/cutil/tensorcofi.pyx"],
              libraries=["cblas"],
              library_dirs=["/usr/lib/atlas-base/atlas"],
              include_dirs=include_dirs),
]


def get_requirements():
    with open('conf/requirements.txt') as reqs_file:
        reqs = filter(None, (x.replace('\n', '').strip() for x in reqs_file if not x.startswith("#")))
        return reqs

setup(name='testfm',
      version='1.1',
      description='Experimentation library for Recommender Systems',
      author='L. Baltrunas and J. Baptista',
      author_email='linas.baltrunas@gmail.com',
      url='http://grafos.ml',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      test_suite='tests',
      package_data={
          'testfm/lib': ['*.jar'],
          'testfm/data': ['*'],
          'testfm/models/cutil': ['*.pyx', '*.pxd'],
          'testfm/evaluation/cutil': ['*.pyx', '*.pxd']
      },
      exclude_package_data={'': ['*.pyc', '*.pyo', '*.o', '*.c', '*.h']},
      license="Apache2",
      include_package_data=True,
      install_requires=get_requirements(),
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules
     )

