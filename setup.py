__author__ = 'linas'

from distutils.core import setup
from setuptools import find_packages

def get_requirements():
    with open('conf/requirements.txt') as reqs_file:
        reqs = filter(None, (x.replace('\n', '').strip() for x in reqs_file
            if not x.startswith("#")))
        return reqs

print get_requirements()

setup(name='testfm',
      version='1.0',
      description='Experimentation library for Recommender Systems',
      author='L. Baltrunas and J. Baptista',
      author_email='linas.baltrunas@gmail.com',
      url='http://grafos.ml',
      package_dir = {'': 'src'},
      packages=find_packages('src'),
      test_suite = 'tests',
      package_data= {
          'testfm/lib': ['*.jar'],
          'testfm/data': ['*']
      },
      exclude_package_data = { '': ['*.pyc','*.pyo'] },
      license="Apache2",
      include_package_data=True,
      install_requires=get_requirements(),
     )

