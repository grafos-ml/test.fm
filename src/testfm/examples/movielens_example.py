__author__ = "linas"

import testfm
import pandas as pd
from testfm.evaluation.evaluator import Evaluator
from testfm.models.baseline_model import Popularity, RandomModel, Item2Item
from testfm.models.tensorcofi import PyTensorCoFi, TensorCoFi, CTensorCoFi
from testfm.models.bpr import BPR
from testfm.models.theano_models import RBM_CF
import datetime

if __name__ == "__main__":
    evaluator = Evaluator()

    #prepare the data
    df = pd.read_csv("netflix.csv",
                     sep=",", header=None, names=["user", "item", "rating"])
    print df.head()
    training, testing = testfm.split.holdoutByRandom(df, 0.8)

    #tell me what models we want to evaluate
    models = [
        RandomModel(),
        Popularity(),
        CTensorCoFi(n_factors=20, n_iterations=5, c_lambda=0.05, c_alpha=40),
        RBM_CF(learning_rate=0.81, training_epochs=7, n_hidden=485),
        #BPR(dim=20),
        #TensorCoFi(n_factors=20, n_iterations=5, c_lambda=0.05, c_alpha=40),
        #PyTensorCoFi(n_factors=20, n_iterations=5, c_lambda=0.05, c_alpha=40),
    ]

    #models += [LinearRank([models[2], models[3]],  item_features_column=["rating"])]
    items = training.item.unique()
    for m in models:
        t = datetime.datetime.now()
        m.fit(training)
        print datetime.datetime.now()-t,
        print m.get_name().ljust(50),
        print evaluator.evaluate_model(m, testing, all_items=items,)
        del m
