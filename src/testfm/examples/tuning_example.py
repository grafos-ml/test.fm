__author__ = 'linas'

import pandas as pd
import testfm
from testfm.models.tensorcofi import PyTensorCoFi as TensorCoFi
from testfm.models.bpr import BPR as TensorCoFi
from testfm.evaluation.evaluator import Evaluator
from pkg_resources import resource_filename

from testfm.evaluation.parameter_tuning import ParameterTuning

if __name__ == "__main__":
    eval = Evaluator()  # Call this before loading the data to save memory (fork of process takes place)

    # Prepare the data
    df = pd.read_csv(resource_filename(testfm.__name__, 'data/movielenshead.dat'),
                     sep="::", header=None, names=['user', 'item', 'rating', 'date', 'title'])
    print df.head()
    training, testing = testfm.split.holdoutByRandom(df, 0.9)

    print "Tuning the parameters."
    tr, validation = testfm.split.holdoutByRandom(training, 0.7)
    pt = ParameterTuning()
    pt.set_max_iterations(100)
    pt.set_z_value(90)
    tf_params = pt.get_best_params(TensorCoFi, tr, validation)
    print tf_params

    tf = TensorCoFi()
    tf.set_params(**tf_params)
    tf.fit(training)
    print tf.get_name().ljust(50),
    print eval.evaluate_model(tf, testing, all_items=training.item.unique())