Multi-Threading Support
***********************

If you are reading this and you didn't come to this page by chance you are using (or at least trying) the framework
Test.FM. So this exquisite framework is developed on top of `Python <https://www.python.org>`_ technology. The beauty
and grace of the Python language is imprinted in the paradigm of the framework making it easy to use and understand.
However, a big limitation in this technology is the **Global Interpreter Lock** or **GIL**. As many should know, the GIL
keeps the Python VM to execute more than one thread at once making it hard to take advantage of the multi-core
architecture.

A pure Python solution for multiprocessing problem is through c-fork. This is a simple solution and has a thread-like
interface that ships with the modern versions of Python The problem, in our case, is that a process based multiprocess
replicate the processing data when some change is made. This can n-plicate the data size making it very heavy to memory.

Our solution comes is a little more complex. It uses a C-Threads uses openMP and wrapping the C library with `Cython
<http://cython.org>`_. Cython is a compiler that works with Python and add some extra features to use data structures
independent from the architecture.

How It Works?
=============

To be able to execute the in multi-threading the **GIL** must be unlock by the thread. Cython allow this feature but in
order to accomplish it no Python data structure must be touched. So all the **NOGIL** execution code is made with
C-data. To accomplish the multi-threading and still be able to use pure Python structures and code a set of *"built-in"*
interfaces were made. With this interfaces the :ref:`testfm-evaluation-evaluator` is able to decide if it can be
executed multi-threading or it has to be single threaded.

The interfaces used by the evaluator are the :ref:`imodel-nogil` and the :ref:`imeasure-nogil`. This two *"interfaces"* have
a nogil method that needs implementation and compilation with Cython tools. For convenience, and because this one is
very "popular", I also developed the :ref:`model-factor`. This one already have the *nogil_get_score* implemented and
just need Python side implementation.

Getting Started
===============

The first step is :ref:`installing the Test.fm framework <install-testfm>`. After that just run:

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

    eval = Evaluator()

    # Print the model name and the result from the Evaluation
    print model.get_name().ljust(50),
    print eval.evaluate_model(m, testing, all_items=items,)  # <<< Multi-Threading Evaluation

In this example, the :ref:`model-tensorcofi` inherit the NOGILModel so it can be divided among multiple threads.
Actually, TensorCoFi model inherit a more sophisticated *"abstract class"* called :ref:`imodel-factor`. This class
connects a set of C float arrays to the model after the fit method execute. It has a nogil_get_score already implemented
that calculate at C-level the score. So the IFactorModel can be implemented by any new model using **pure** Python and
it still will have nogil functionality.

However there are a certain protocol to follow when implementing the train method.

#. The model attribute for the factors **must be** a list of `numpy.array
   <http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_.

#. The data type must be **numpy.float32**.

#. The rows should be the "objects" and the columns must be the factors, objects = users, items, contexts, etc...


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
    :lines: 102
    :linenos:

Also must note that the arrays are loaded as np.float32 arrays. And they get transposed after that to correspond to user
and item rows and factor columns.

After the execution of the train, the fit method go on to fill the C-array with the numpy array data. Actually it just
turn the pointer to the numpy C-data so it don't replicate data and when there is a change in the factors it is also
applied to there.