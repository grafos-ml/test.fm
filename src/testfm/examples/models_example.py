__author__ = "linas"

import testfm
import pandas as pd
from testfm.evaluation.evaluator import Evaluator
from testfm.models.baseline_model import Popularity, RandomModel, Item2Item
from testfm.models.tensorcofi import PyTensorCoFi, TensorCoFi, CTensorCoFi
from testfm.models.content_based import TFIDFModel, LSIModel
from testfm.models.bpr import BPR
from pkg_resources import resource_filename
import datetime


if __name__ == "__main__":
    evaluator = Evaluator()

    #prepare the data
    df = pd.read_csv(resource_filename(testfm.__name__, "data/movielenshead.dat"),
                     sep="::", header=None, names=["user", "item", "rating", "date", "title"])
    print df.head()
    training, testing = testfm.split.holdoutByRandom(df, 0.5)

    #tell me what models we want to evaluate
    models = [
        RandomModel(),
        BPR(),
        TFIDFModel("title"),
        Popularity(),
        TensorCoFi(n_factors=20, n_iterations=5, c_lambda=0.05, c_alpha=40),
        PyTensorCoFi(n_factors=20, n_iterations=5, c_lambda=0.05, c_alpha=40),
        CTensorCoFi(n_factors=20, n_iterations=5, c_lambda=0.05, c_alpha=40),
        LSIModel("title")
    ]

    #models += [LinearRank([models[2], models[3]],  item_features_column=["rating"])]
    items = training.item.unique()
    for m in models:
        t = datetime.datetime.now()
        m.fit(training)
        print datetime.datetime.now()-t,
        print m.get_name().ljust(50),
        print evaluator.evaluate_model(m, testing, all_items=items,)
