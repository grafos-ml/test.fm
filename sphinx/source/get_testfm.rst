.. _get-testfm:

Get Test.fm
***********

Test.fm is an open-source project under `Apache2 licence <https://github.com/grafos-ml/test.fm/blob/master/LICENSE>`_.
It is currently supported for Python 2.7 or higher but don't offer support on Python 3 series. It has dependencies
on ATLAS and LAPACK libraries.

How to get Test.fm
==================

Test.fm is currently in heavy development stage. To get it just access to the project page at
`GitHub <https://github.com/grafos-ml/test.fm>`_ and download the zip version. If you want to contribute feel free to
fork it.

Install on Ubuntu
-----------------

It is farly easy to install it on Ubuntu. First make sure that you install the Python developer tools, ATLAS and LAPACK
library. If you don't have it just run::

    $ sudo apt-get install gfortran libatlas-base-dev liblapack-dev python-dev build-essential

After this you may want to install a Python `virtual environment <http://virtualenv.readthedocs.org/en/latest/>`_. Also,
you can use the `virtualenvwrapper <http://virtualenvwrapper.readthedocs.org/en/latest/>`_, a pretty neat tool if you
need to work with a lot of virtual environments. Now, to install it there are 2 ways.

1 - Install it with setup.py
____________________________

#. Download the zip file from `GitHub <https://github.com/grafos-ml/test.fm/archive/version-2.zip>`_.

#. Unzip the file content to some folder.

#. In that folder run::

    $ python setup.py install

2 - Install it with pip
_______________________

This project is not yet in the pip repo. So in order to make it happen just follow the steps.

#. Just run::

    $ pip install https://github.com/grafos-ml/test.fm/archive/version-2.zip

#. Done.

Install on Mac OS
-----------------

To set Test.fm up and running in Mac OS as the same requisites. Install BLAS and LAPACK libraries. ATLAS is recommended.
To install ATLAS you will need to have a fortran 77 compiler installed.
If you don't have it use the MacPort::

    $ port install atlas +gcc47+gfortran

The gcc library is a specific requisite to install Test.fm, so if you use the binary BLAS you will still need to install
gcc on mac.

If you have pip installed it should do just by running::

    $ pip install https://github.com/grafos-ml/test.fm/archive/version-2.zip

If you have some compilation errors try download the zip unpack it build the binaries and than install::

    $ python package_dir/setup.py build_ext --inplace
    $ python package_dir/setup.py install
    $ rm -r package_dir

Known Issues
____________

#. Mavericks PIP issue:
    Some python libraries with dependencies of binary modules could have to be build from the source. The problem is the
    binaries in the repositories are not supported for the new versions.


Install on Windows
------------------

Test.fm don't have support for Windows.

Uninstall Test.fm
-----------------

If you think this package is to awesome for you can just remove it from your python environment. We recommend to use the
pip::

    $ pip uninstall testfm

Or you can go old school and remove the directory by yourself.
