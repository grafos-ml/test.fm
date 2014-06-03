.. _get-testfm:

Installing Test.fm
******************

Test.fm is an open-source recommender system testing toolkit under `Apache2 licence <https://github.com/grafos-ml/test.fm/blob/master/LICENSE>`_.
It is currently supported for Python 2.7 or higher but don't offer support on Python 3 series. It has dependencies
on CBLAS, LAPACK libraries, Java RE and some python modules.

How to install Test.fm
======================

Test.fm is currently in heavy development stage. To get it just access to the project page at
`GitHub <https://github.com/grafos-ml/test.fm>`_ and download the zip version. If you want to contribute feel free to
fork it.

Install on Ubuntu
-----------------
It is fairly easy to install test.fm on Ubuntu. First make sure that you install the Python developer tools, ATLAS and LAPACK
library. If you don't have it just run::

    $ sudo apt-get install gfortran libatlas-base-dev liblapack-dev python-dev build-essential

After this you may want to install a Python `virtual environment <http://virtualenv.readthedocs.org/en/latest/>`_. Also,
you can use the `virtualenvwrapper <http://virtualenvwrapper.readthedocs.org/en/latest/>`_, a pretty neat tool if you
need to work with a lot of virtual environments. 

There are 2 ways to install test.fm::

1 - Install it with pip
_______________________

This project is not yet in the pip repo. So in order to make it happen just type::

    $ pip install https://github.com/grafos-ml/test.fm/archive/v1.0.tar.gz

This should install all the dependencies and the test.fm itself.

2 - Install it with setup.py
____________________________

#. Download the zip file from `GitHub <https://github.com/grafos-ml/test.fm/archive/v1.0.tar.gz>`_.

#. Unzip the file content to some folder.

#. In that folder run::

    $ python setup.py install



Install on Mac OS
-----------------

To install test.fm  on Mac OS you need to install CBLAS and LAPACK libraries. ATLAS library in Mac
ships with them both so it is the recommended way of installing. To install ATLAS you will need to have a fortran 77 compiler installed.
The gcc library is a specific requisite to install Test.fm, so if you use the binary CBLAS you will still need to
install gcc on mac.

Please install the dependencies with::

    $ port install atlas +gcc47+gfortran

In case this fails, try first running::
    
    $ port clean atlas ; port install atlas +gcc47+gfortran

Now install test.fm using::

    $ pip install https://github.com/grafos-ml/test.fm/archive/v1.0.tar.gz

.. warning::

    This is the standard way to install test.fm and works pretty well on linux based systems and OS X 10.9.2. 
    However this method raised some complications on other versions of Mac OS. 
    Is you cannot use this in a different release of the Mac OS please post on the issue page with the exact version of your OS.

If you have some compilation errors, try download the zip unpack it, build the binaries and than install using::

    $ python package_dir/compile.py build_ext
    $ python package_dir/setup.py install
    $ rm -r package_dir

Known Issues
____________

#. Mavericks PIP issue:
    Some python libraries (pandas) fails to compile and have to be build from the source. The problem is the
    binaries in the repositories are not supported for some specific OS versions.


Install on Windows
------------------

At this moment Test.fm doesn't have support for Windows.

Uninstall Test.fm
-----------------

If you think this package is too awesome for you can just remove it from your python environment. We recommend to use the
pip::

    $ pip uninstall testfm

Or you can go old school and remove the directory by yourself.

Developing Test.fm
------------------

In case you are contributing to the test.fm, please fork the code and use::

    $ python setup.py develop

