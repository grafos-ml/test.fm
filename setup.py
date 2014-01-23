__author__ = 'linas'

from distutils.core import setup
from setuptools import find_packages

def get_requirements():
    with open('conf/requirements.txt') as reqs_file:
        reqs = filter(None, (x.replace('\n', '').strip() for x in reqs_file if not x.startswith("#")))
        return reqs

setup(name='testfm',
      version='1.0',
      description='Experimentation library for Recommender Systems',
      author='L. Baltrunas and J. Baptista',
      author_email='linas.baltrunas@gmail.com',
      url='http://grafos.ml',
      packages=find_packages(),
      license="Apache2",
      include_package_data=True,
     )

