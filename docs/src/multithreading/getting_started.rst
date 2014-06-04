Multi-Threading Support
***********************

If you are reading this and you didn't come to this page by chance you are using (or at least trying) the test.fm framework.
So this framework is developed on top of `Python <https://www.python.org>`_ technology. The beauty
and grace of the Python language is imprinted in the paradigm of the framework making it easy to use and understand.
However, a big limitation in this technology is the **Global Interpreter Lock** or **GIL**. As many should know, the GIL
keeps the Python VM to execute more than one thread at once but makes it hard to take advantage of the multi-core
architecture.

A pure Python solution for multiprocessing problem is through c-fork. This is a simple solution and has a thread-like
interface that ships with the modern versions of Python. The problem, in our case, is that a process based multiprocessing
library replicate the processing data when a change is made. This can n-plicate the input data making it infeasible
for bigger data sets.

Our solution for this problem uses a C-Threads and openMP. We rewrite some of the computation intensive parts
of the evaluation framework in `Cython <http://cython.org>`_. Cython is a compiler that works with Python and add some extra features to use data structures
independent from the architecture.

How It Works?
=============

To be able to execute the in multi-threading the **GIL** must be unlock by the thread. Cython allow this feature but in
order to accomplish it no Python data structure must be touched. So all the **NOGIL** execution code is made with
C-data. To accomplish the multi-threading and still be able to use pure Python structures and code a set of *"built-in"*
interfaces were made. With this interfaces the :ref:`testfm-evaluation-evaluator` is able to decide if it can be
executed multi-threading or it has to be single threaded.

The interfaces used by the evaluator are the :ref:`imodel-nogil` and the :ref:`imeasure-nogil`. These two *"interfaces"* have
a nogil method that needs implementation and compilation with Cython tools. For convenience, I also developed 
the :ref:`model-factor` - an base class implementation for most of factor models that use dot product
as the prediction function. This one already have the *nogil_get_score* implemented and
just need Python side implementation.

Getting Started
===============

The first step is :ref:`installing the Test.fm framework <get-testfm>`. After that just run:

.. code-block:: python
   :linenos:

    import pandas as pd
    import testfm
    from pkg_resources import resource_filename
    from testfm.evaluation.evaluator import Evaluator
    from testfm.models.tensorcofi import TensorCoFi

    # Load data from the testfm example data
    file_path = resource_filename(testfm.__name__, "data/movielenshead.dat")
    df = pd.read_csv(file_path, sep="::", header=None, names=["user", "item", "rating", "date", "title"])

    # Divide the data into training and testing
    training, testing = testfm.split.holdoutByRandom(df, 0.5)
    items = training.item.unique()

    # Create a NOGILModel model and fit it
    model = TensorCoFi(n_factors=20, n_iterations=5, c_lambda=0.05, c_alpha=40)
    model.fit(training)

    evaluator = Evaluator()

    # Print the model name and the result from the Evaluation
    print model.get_name().ljust(50),
    print evaluator.evaluate_model(m, testing, all_items=items,)  # <<< Multi-Threading Evaluation

In this example, the :ref:`model-tensorcofi` inherit the NOGIL model, therefore, the computation can be divided among multiple threads.
Actually, TensorCoFi model inherit a more sophisticated *"abstract class"* called :ref:`imodel-factor`. This class
connects a set of C float arrays to the model after the fit method execute. It has a nogil_get_score already implemented
that calculate the score using low level implementation. So the IFactorModel can be implemented by any new model using **pure** Python and
it still will have nogil functionality.

However there are a certain protocol to follow when implementing the train method.

#. The model attribute for the factors **must be** a list of `numpy.array
   <http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_.

#. The data type must be **numpy.float32**.

#. The rows should be the "objects" and the columns must be the factors, objects = users, items, contexts, etc...


NOGIL Model from Scratch
------------------------

To make a new **nogil** model from scratch two things are needed. The Cython with the low level routine and the setup.py
to compile the Cython part. In this example we will make a third structure, a Python wrapper. The reason for this is so
we can have only the low level part in the binary and the rest of the features in the python side. This way it only
compile when a change in the nogil part is made and no more than that.

.. note::

     Keep in mind that compiling the Python code using Cython increases the performance of the execution even if you don't
     use any other optimization. However, you lose the simplicity of pure python.


We will build a random model with the nogil interface. First create a file named crandom.pyx. This file will have the low
level code for our plugin. Make the necessary imports. For this we need some Python libraries and some C libraries.
Notice that the python libraries are called the same way but the C libraries are called by using the keyword cimport.

.. code-block:: cython
    :linenos:

    # crandom.pyx

    cimport cython
    from libc.stdlib cimport rand, RAND_MAX
    from testfm.models.cutil.interface cimport NOGILModel

    cdef class NOGILRandomModel(NOGILModel):

        @cython.cdivision(False)
        cdef float nogil_get_score(NOGILRandomModel self, int user, int item, int extra_context, int *context) nogil:
            return rand() / <float>RAND_MAX

NOGILModel is imported by the c library in testfm.models.cutil.interface. We also import rand and RAND_MAX to generate the
low-level pseudo-random numbers. Notice how the class is defined as cdef. This make it hybrid between C and Python and able
to run pure C methods. Other difference from pure python is that the method and each parameter has as a defined type. 
Last but not least, the decorator on the method is a special Cython
functionality. This one disables the Python security division checks. This can boost the division operation up to
36 %, according to Cython documentation.

Now for the wrapper.

.. code-block:: python
    :linenos:

    # random_model.py

    from crandom import NOGILRandomModel

    class RandomModel(NOGILRandomModel):
        """
        Random model
        """
        _scores = {}

        def get_score(self, user, item, **context):
            key = user, item
            try:
                return self._scores[key]
            except KeyError:
                self._scores[key] = random()
                return self._scores[key]

        def get_name(self):
            return "Random"

This class implements get_score and also inherits the **nogil_get_score** form the parent. 
These both functions does the same thing but nogil_get_score is faster and allows multi-threading.
The evaluator class will use nogil_get_score by default and fall back to get_score if no threading support
can be provided.

Just need the setup.py now.

.. code-block:: python
    :linenos:

    # setup.py
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Distutils import build_ext

    setup(
      name = "random",
      cmdclass = {"build_ext": build_ext},
      ext_modules = [
        Extension("crandom", ["crandom.pyx"])
      ],
    )

Now just run python **setup build_ext --inplace** and voilá, you have your first multi-threading model ready to use.

TensorCoFi Implementation
-------------------------

First stage for the TensorCoFi Implementation is declaration of the class inheriting the IFactorModel.

.. literalinclude:: /../../src/testfm/models/tensorcofi.py
    :language: python
    :lines: 25, 28-30, 68, 72-103
    :linenos:

After that we implemented the **train** method. In some point of that it will receive the factors matrices. This
implementation uses the Java TensorCoFi and connect to it by a set of files. This files load 2 numpy.arrays, users and
items.

.. literalinclude:: /../../src/testfm/models/tensorcofi.py
    :language: python
    :lines: 102-103
    :linenos:

This also can be coded as:

.. code-block:: python
    :linenos:

    # Get the matrices location
    user_path, item_path = out.split(" ")

    # Load the both matrices. Notice they are loaded as float32
    user_factor_matrix = np.genfromtxt(open(user_path, "r"), delimiter=",", dtype=np.float32)
    item_factor_matrix = np.genfromtxt(open(item_path, "r"), delimiter=",", dtype=np.float32)

    # They are loaded with factors as rows. So we use the transpose for respect the 3rd clause in the protocol
    self.factors = [user_factor_matrix.transpose(), item_factor_matrix.transpose()]

Also must note that the arrays are loaded as np.float32 arrays. And they get transposed after that to correspond to user
and item rows and factor columns.

After the execution of the train, the fit method go on to fill the C-array with the numpy array data. Actually it just
turn the pointer to the numpy C-data so it don't replicate data and when there is a change in the factors it is also
applied to there.

