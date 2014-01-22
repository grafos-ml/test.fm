__author__ = 'linas'

import pandas as pd
import numpy as np
import testfm
from testfm.evaluation.evaluator import Evaluator
from testfm.models.baseline_model import Popularity, RandomModel, IdModel
from testfm.models.ensemble_models import LinearEnsemble

#prepare the data
df = pd.read_csv('../data/movielenshead.dat', sep="::", header=None, names=['user', 'item', 'rating', 'date'])
print df.head()
training, testing = testfm.split.holdoutByRandom(df, 0.8)

#tell me what models we want to evaluate
models = [RandomModel(), Popularity(), IdModel(), LinearEnsemble([RandomModel(), Popularity()], weights=[0.5, 0.5])]

#evaluate
items = training.item.unique()
evaluator = Evaluator()
print "\n\n"
for m in models:
    m.fit(training)
    print m.getName().ljust(50) , evaluator.evaluate_model(m, testing, all_items=items)

print "\n\n"
for alpha in np.linspace(0.0, 1.0, num=10):
    model = LinearEnsemble([RandomModel(), Popularity()], weights=[alpha, 1.0-alpha])
    print model.getName().ljust(50) , evaluator.evaluate_model(model, testing, all_items=items)