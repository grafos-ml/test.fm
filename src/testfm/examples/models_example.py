__author__ = 'linas'

import testfm
import pandas as pd
from testfm.evaluation.evaluator import Evaluator
from testfm.models.baseline_model import Popularity, RandomModel, Item2Item
from testfm.models.tensorCoFi import TensorCoFi
from testfm.models.content_based import TFIDFModel, LSIModel
from testfm.models.ensemble_models import LinearRank
from testfm.models.bpr import BPR
from pkg_resources import resource_filename

#because of Global Interpreter Lock we need to initialize evaluator
eval = Evaluator()

#prepare the data
df = pd.read_csv(resource_filename(testfm.__name__,'data/movielenshead.dat'),
    sep="::", header=None, names=['user', 'item', 'rating', 'date', 'title'])
print df.head()
training, testing = testfm.split.holdoutByRandom(df, 0.9)

#tell me what models we want to evaluate
models = [  RandomModel(),
            BPR(),
            TFIDFModel('title'),
            Popularity(),
            TensorCoFi(dim=20, nIter=5, lamb=0.05, alph=40, user_features=['user'], item_features=['item', 'title']),
            #Item2Item(),
            #LSIModel('title'),
         ]

#models += [LinearRank([models[2], models[3]],  item_features_column=['rating'])]
items = training.item.unique()
for m in models:
    m.fit(training)
    print m.getName().ljust(50),
    print eval.evaluate_model(m, testing, all_items=items,)

eval.close()#need this call to clean up the worker processes