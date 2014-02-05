__author__ = 'linas'

import pandas as pd
import testfm
from testfm.models.tensorCoFi import TensorCoFi
from pkg_resources import resource_filename

from testfm.evaluation.parameterTuning import ParameterTuning

#prepare the data
df = pd.read_csv(resource_filename(testfm.__name__,'data/movielenshead.dat'),
                 sep="::", header=None, names=['user', 'item', 'rating', 'date', 'title'])
print df.head()
training, testing = testfm.split.holdoutByRandom(df, 0.9)

print "Tuning the parameters."
tr, validation = testfm.split.holdoutByRandom(training, 0.7)
pt = ParameterTuning()
pt.setMaxIterations(10)
pt.setZvalue(80)
tf_params = pt.getBestParams(TensorCoFi, tr, validation)
print tf_params

tf = TensorCoFi()
tf.setParams(**tf_params)
tf.fit(training)
print tf.getName().ljust(50),
print testfm.evaluate_model(tf, testing, all_items=training.item.unique())
