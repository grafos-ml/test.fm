__author__ = "linas"

import testfm
import pandas as pd
from testfm.evaluation.evaluator import Evaluator
from testfm.models.baseline_model import Popularity, RandomModel, Item2Item
from testfm.models.tensorcofi import PyTensorCoFi, TensorCoFi, CTensorCoFi
from testfm.models.content_based import TFIDFModel, LSIModel
from testfm.models.bpr import BPR
from testfm.models.theano_models import RBM
from pkg_resources import resource_filename
import datetime
from testfm.evaluation.parameter_tuning import ParameterTuning

if __name__ == "__main__":
    evaluator = Evaluator()

    #prepare the data
    df = pd.read_csv("/Users/mumas/devel/data-mac/1M_movielens/ratings.dat",
                     sep=" ", header=None, names=["user", "item", "rating", "date"])
    print df.head()
    training, testing = testfm.split.holdoutByRandom(df, 0.5)


    print "Tuning the parameters."
    tr, validation = testfm.split.holdoutByRandom(training, 0.7)
    pt = ParameterTuning()
    pt.set_max_iterations(100)
    pt.set_z_value(90)
    params = pt.get_best_params(RBM, tr, validation)
    print params

    #tell me what models we want to evaluate
    models = [
        RandomModel(),
        RBM(n_hidden=20),
        BPR(dim=20),
        Popularity(),
        TensorCoFi(n_factors=20, n_iterations=5, c_lambda=0.05, c_alpha=40),
        PyTensorCoFi(n_factors=20, n_iterations=5, c_lambda=0.05, c_alpha=40),
        CTensorCoFi(n_factors=20, n_iterations=5, c_lambda=0.05, c_alpha=40),
    ]

    #models += [LinearRank([models[2], models[3]],  item_features_column=["rating"])]
    items = training.item.unique()
    for m in models:
        t = datetime.datetime.now()
        m.fit(training)
        print datetime.datetime.now()-t,
        print m.get_name().ljust(50),
        print evaluator.evaluate_model(m, testing, all_items=items,)
