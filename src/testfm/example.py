__author__ = 'linas'

import pandas as pd
import testfm
from testfm.models.baseline_model import Popularity, RandomModel
from testfm.models.tensorCoFi import TensorCoFi
from pkg_resources import resource_filename

#prepare the data
df = pd.read_csv(resource_filename(testfm.__name__,'data/movielenshead.dat'),
    sep="::", header=None, names=['user', 'item', 'rating', 'date', 'title'])
print df.head()
training, testing = testfm.split.holdoutByRandom(df, 0.9)

#tell me what models we want to evaluate
models = [  RandomModel(),
            Popularity(),
            TensorCoFi(),
         ]

#evaluate
items = training.item.unique()
for m in models:
    m.fit(training)
    print m.getName().ljust(50),
    print testfm.evaluate_model(m, testing, all_items=items)
