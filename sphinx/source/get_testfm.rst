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

Install it on Ubuntu
--------------------

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

Install it on Mac OS
--------------------

It is more complicated to install Test.fm with Mac OS that with Ubuntu. First make sure to have installed Atlas library.
If you don't have it use the MacPort::

    $ port install atlas +gcc47



.. note::

    If you have ATLAS installed in a different place other than the standard place you have to pass them in a set of
    environment variables. BLASLIB variable with the path for the BLAS library and LAPACKLIB with the path for
    LAPACK library.