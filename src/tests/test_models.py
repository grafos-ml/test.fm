# -*- coding: utf-8 -*-
__author__ = "linas"

import os
import unittest
import pandas as pd
import numpy as np
from pkg_resources import resource_filename
import testfm
from testfm.models.graphchi_models import SVDpp
from testfm.models.tensorcofi import TensorCoFi, PyTensorCoFi
from testfm.models.baseline_model import IdModel, Item2Item, AverageModel
from testfm.models.ensemble_models import LogisticEnsemble
from testfm.models.content_based import TFIDFModel, LSIModel
from testfm.evaluation.evaluator import Evaluator


def which(program):
    """
    Returns True if program is on the path to be executed in unix
    """

    def is_exe(fpath):
        if os.path.isfile(fpath) and os.access(fpath, os.X_OK):
            return True

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return True
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip("\"")
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return True
    return False


class TestTensorCoFi(unittest.TestCase):

    def tearDown(self):
        import os
        if os.path.exists("user.csv"):
            os.remove("user.csv")
        if os.path.exists("item.csv"):
            os.remove("item.csv")

    def setUp(self):
        self.df = pd.read_csv(resource_filename(testfm.__name__, "data/movielenshead.dat"), sep="::", header=None, 
                              names=["user", "item", "rating", "date", "title"])
        self.df = self.df.head(n=100)

    def test_fit(self):
        tf = TensorCoFi(n_factors=2)
        tf.fit(self.df)
        #item and user are row vectors
        self.assertEqual(len(self.df.user.unique()), tf.factors[0].shape[0])
        self.assertEqual(len(self.df.item.unique()), tf.factors[1].shape[0])

    def test_refit(self):
        """
        [TensorCoFi] Test fit after first fitting
        """
        tf = TensorCoFi(n_factors=2)
        tf.fit(self.df)
        tf.fit(self.df)  # Second fit
        #item and user are row vectors
        self.assertEqual(len(self.df.user.unique()), tf.factors[0].shape[0])
        self.assertEqual(len(self.df.item.unique()), tf.factors[1].shape[0])
        tf.fit(self.df)  # Third fit
        #item and user are row vectors
        self.assertEqual(len(self.df.user.unique()), tf.factors[0].shape[0])
        self.assertEqual(len(self.df.item.unique()), tf.factors[1].shape[0])


    def test_ids_returns(self):
        tf = TensorCoFi(n_factors=2)
        inp = [{"user": 10, "item": 100},
               {"user": 10, "item": 110},
               {"user": 12, "item": 120}]
        inp = pd.DataFrame(inp)
        tf.fit(inp)

        # Test the id in map
        uid = tf.data_map[tf.get_user_column()][10]
        iid = tf.data_map[tf.get_item_column()][100]
        self.assertEquals(uid, 0)
        self.assertEquals(iid, 0)

        # Test number of factors
        self.assertEquals(len(tf.factors[0][uid, :]), tf.number_of_factors)
        self.assertEquals(len(tf.factors[1][iid, :]), tf.number_of_factors)

    def test_score(self):
        tf = TensorCoFi(n_factors=2)
        inp = [{"user": 10, "item": 100},
               {"user": 10, "item": 110},
               {"user": 12, "item": 120}]
        inp = pd.DataFrame(inp)
        tf.fit(inp)
        uid = tf.data_map[tf.get_user_column()][10]
        iid = tf.data_map[tf.get_item_column()][100]
        tf.factors[0][uid, 0] = 0
        tf.factors[0][uid, 1] = 1
        tf.factors[1][iid, 0] = 1
        tf.factors[1][iid, 1] = 5
        self.assertEqual(0*1+1*5, tf.get_score(10, 100))

    def test_fit_for_python_version(self):
        tf = PyTensorCoFi(n_factors=2)
        tf.fit(self.df)
        #item and user are row vectors
        self.assertEqual(len(self.df.user.unique()), tf.factors[0].shape[0])
        self.assertEqual(len(self.df.item.unique()), tf.factors[1].shape[0])

    def test_ids_returns_for_python_version(self):
        tf = PyTensorCoFi(n_factors=2)
        inp = [{"user": 10, "item": 100},
               {"user": 10, "item": 110},
               {"user": 12, "item": 120}]
        inp = pd.DataFrame(inp)
        tf.fit(inp)

        # Test the id in map
        uid = tf.data_map[tf.get_user_column()][10]
        iid = tf.data_map[tf.get_item_column()][100]
        self.assertEquals(uid, 0)
        self.assertEquals(iid, 0)

        # Test number of factors
        self.assertEquals(len(tf.factors[0][uid, :]), tf.number_of_factors)
        self.assertEquals(len(tf.factors[1][iid, :]), tf.number_of_factors)

    def test_score_for_python_version(self):
        tf = PyTensorCoFi(n_factors=2)
        inp = [{"user": 10, "item": 100},
               {"user": 10, "item": 110},
               {"user": 12, "item": 120}]
        inp = pd.DataFrame(inp)
        tf.fit(inp)
        uid = tf.data_map[tf.get_user_column()][10]
        iid = tf.data_map[tf.get_item_column()][100]

        tf.factors[0][uid, 0] = 0
        tf.factors[0][uid, 1] = 1
        tf.factors[1][iid, 0] = 1
        tf.factors[1][iid, 1] = 5
        self.assertEqual(0*1+1*5, tf.get_score(10, 100))


class LogisticTest(unittest.TestCase):

    def setUp(self):
        self.le = LogisticEnsemble(models=[IdModel()])
        inp = [{"user": 10, "item": 100}, {"user": 10,"item": 110},
               {"user": 12, "item": 120}]
        self.df = pd.DataFrame(inp)

    def test_construct_features(self):

        self.le._prepare_feature_extraction(self.df)
        self.assertEqual(self.le._user_count, {10: 2, 12: 1})

    def test_extract_features(self):
        self.le._prepare_feature_extraction(self.df)
        features = self.le._extract_features(10, 5)
        self.assertEqual(features, ([2, 5], 1))

    def test_prepare_data(self):
        X, Y = self.le.prepare_data(self.df)
        self.assertEqual(len(X), 6)
        self.assertEqual(len(Y), 6)

    def test_fit(self):
        self.le.fit(self.df)
        self.assertIsNotNone(self.le.model)

    def test_predict(self):
        self.le.fit(self.df)
        self.assertIsInstance(self.le.get_score(10, 110), float)


class Item2ItemTest(unittest.TestCase):

    def test_fit(self):
        df = pd.DataFrame([{"user": 10, "item": 100}, {"user": 10, "item": 110}, {"user": 12, "item": 100}])
        i2i = Item2Item()
        i2i.fit(df)

        self.assertEqual(i2i._items[100], set([10, 12]))
        self.assertEqual(i2i._items[110], set([10]))

        self.assertEqual(i2i._users[10], set([100, 110]))
        self.assertEqual(i2i._users[12], set([100]))

        self.assertEqual(i2i.similarity(100, 100), 1.0)
        self.assertEqual(i2i.similarity(100, 110), 1.0/2.0)

    def test_score(self):
        df = pd.DataFrame([{"user": 10, "item": 100}, {"user": 10, "item": 110}, {"user": 12, "item": 100}])
        i2i = Item2Item()
        i2i.fit(df)

        self.assertEqual(i2i.get_score(12, 110), 0.5)


class TestLSI(unittest.TestCase):

    def setUp(self):
        self.lsi = LSIModel("title")
        self.df = pd.read_csv(resource_filename(testfm.__name__,"data/movielenshead.dat"), sep="::", header=None,
                              names=["user", "item", "rating", "date", "title"])

    def test_fit(self):
        self.lsi.fit(self.df)
        self.assertEqual(len(self.lsi._user_representation), len(self.df.user.unique()))
        self.assertEqual(len(self.lsi._item_representation), len(self.df.item.unique()))

    def test_score(self):
        self.lsi.fit(self.df)
        #item in the user profile (Booberang) should have higher prediction than movie not in the profile Rob Roy
        self.assertTrue(self.lsi.get_score(1, 122) > self.lsi.get_score(1, 151))

    def test_user_model(self):
        um = self.lsi._get_user_models(self.df)
        self.assertEqual(um[93], ["collateral", "man", "fire"])

    def test_item_model(self):
        im = self.lsi._get_item_models(self.df)
        self.assertEqual(im[122], ["boomerang"])
        self.assertEqual(im[329], ["star", "trek", "generations"])


class TFIDTest(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame([{"user": 10, "item": 100, "desc": "car is very nice"},
                                {"user": 11, "item": 100, "desc": "car is very nice"},
                                {"user": 11, "item": 1, "desc": "oh my god"},
                                {"user": 12, "item": 110, "desc": "the sky sky is blue and nice"}])

    def test_item_user_model(self):
        tfidf = TFIDFModel("desc")
        tfidf.fit(self.df)

        self.assertEqual(tfidf._get_item_models(self.df), {1: ["oh", "god"], 100: ["car", "very", "nice"],
                                                           110: ["sky", "sky", "blue", "nice"]})
        self.assertEqual(tfidf._users, {10: set([100]), 11: set([100, 1]), 12: set([110])})
        self.assertEqual(sorted(tfidf.tfidf.keys()), sorted([100, 1, 110]))

    def test_item_model(self):
        tfidf = TFIDFModel("desc")
        tfidf.fit(self.df)

        self.assertAlmostEqual(tfidf._sim(1, 1), 1, places=2)
        self.assertAlmostEqual(tfidf._sim(100, 100), 1, places=2)
        self.assertGreater(tfidf._sim(1, 1), tfidf._sim(1, 100), "similarities do not make sense")

    def test_get_score(self):
        tfidf = TFIDFModel("desc")
        tfidf.fit(self.df)
        tfidf.k = 1

        #the closes item to 1 (in user 10 profile) is 100, so the score should be equal to the similarity
        self.assertAlmostEqual(tfidf.get_score(10, 1), tfidf._sim(100, 1), places=2)


class SVDppTest(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame([{"user": 10, "item": 100, "rating": 5},
                                {"user": 11, "item": 100, "rating": 4},
                                {"user": 11, "item": 1, "rating": 3},
                                {"user": 12, "item": 110, "rating": 2}])
        self.df_big = pd.read_csv(resource_filename(testfm.__name__, "data/movielenshead.dat"),
                                  sep="::", header=None, names=["user", "item", "rating", "date", "title"])

    @unittest.skipIf(not which("svdpp"), "svdpp is not on the path")
    def test_train(self):
        svdpp = SVDpp()
        self.assertFalse(hasattr(svdpp, "U_bias"))
        svdpp.fit(self.df_big)
        self.assertTrue(hasattr(svdpp, "U_bias"))

        #did graphchi changed the output format? it used to be 40 for users...
        self.assertEqual(svdpp.U.shape, (len(self.df_big.user.unique()), 20))
        self.assertEqual(svdpp.V.shape, (len(self.df_big.item.unique()), 20))
        self.assertEqual(svdpp.U_bias.shape, (len(self.df_big.user.unique()), 1))
        self.assertEqual(svdpp.V_bias.shape, (len(self.df_big.item.unique()), 1))
        self.assertEqual(svdpp.global_mean.shape, (1, 1))

    @unittest.skipIf(not which("svdpp"), "svdpp is not on the path")
    def test_score(self):
        import types

        svdpp = SVDpp()
        svdpp.fit(self.df)
        self.assertTrue(isinstance(svdpp.get_score(10, 100), types.FloatType))
        self.assertTrue(isinstance(svdpp.get_score(12, 110), types.FloatType))

    def test_dump(self):
        svdpp = SVDpp()
        filename = svdpp.dump_data(self.df)
        lines = open(filename).readlines()
        self.assertEqual(lines[0], "%%MatrixMarket matrix coordinate real general\n")
        self.assertEqual(lines[2], "3 3 4\n")
        self.assertEqual(lines[3], "1 2 5\n")
        self.assertEqual(lines[4], "2 2 4\n")

        self.assertEqual(3, len(svdpp.umap))
        self.assertEqual(svdpp.umap[10], 0)
        self.assertEqual(svdpp.umap[11], 1)
        self.assertEqual(svdpp.umap[12], 2)

        self.assertEqual(3, len(svdpp.imap))
        self.assertEqual(svdpp.imap[1], 0)
        self.assertEqual(svdpp.imap[100], 1)
        self.assertEqual(svdpp.imap[110], 2)


class MeanPredTest(unittest.TestCase):

    df = pd.DataFrame([{"user": 10, "item": 100, "rating": 5},
                       {"user": 11, "item": 100, "rating": 4},
                       {"user": 11, "item": 1, "rating": 3},
                       {"user": 12, "item": 110, "rating": 2}])

    def test_fit(self):
        model = AverageModel()
        model.fit(self.df)

        self.assertEqual(model.get_score(10, 100), 4.5)


class PyTensorTest(object):

    def test_user_model_update(self):
        pyTF = PyTensorCoFi()
        Y = np.array([[-1.0920831, -0.01566422], [-0.8727925, 0.22307773], [0.8753245, -0.80181429],
                      [-0.1338534, -0.51448172], [-0.2144651, -0.96081265]])
        user_items = [1, 3, 4]
        res = pyTF.online_user_factors(Y, user_items, p_param=10, lambda_param=0.01)
        assert np.array([-1.18324547, -0.95040477]).all() == res.all(), "Results not equal"

    def test_dynamic_updates(self):
        """
        TensorCoFi dynamic update
        We will take a tensor cofi. Train the model, evaluate it. Then we remove all the user factors
        and recompute them using the online_user_factors to check if the performance is almost the same...
        """
        pyTF = PyTensorCoFi(n_factors=20, n_iterations=5, c_lambda=0.05, c_alpha=40)

        evaluator = Evaluator()
        tf = TensorCoFi(n_factors=2, n_iterations=100, c_lambda=0.05, c_alpha=40)
        df = pd.read_csv(resource_filename(testfm.__name__, "data/movielenshead.dat"), sep="::", header=None,
                         names=["user", "item", "rating", "date", "title"])
        training, testing = testfm.split.holdoutByRandom(df, 0.7)
        users = {user: list(entries) for user, entries in training.groupby("user")["item"]}

        tf.fit(training)
        map1 = evaluator.evaluate_model(tf, testing)  # map of the original model

        #now we try to replace the original factors with on the fly computed factors
        #lets iterate over the training data of items and the users
        for u, items in users.items():
            #user id in the tf
            uid = tf.data_map[tf.get_user_column()][u]  # user id
            iids = [tf.data_map[tf.get_item_column()][i] for i in items]  # item ids that user has seen
            #original_factors = tf.factors["user"][uid]
            new_factors = pyTF.online_user_factors(tf.factors[1], iids, p_param=40, lambda_param=0.05)
            #replace original factors with the new factors
            tf.factors[0][uid, :] = new_factors
            #tf.update_user_factors(uid, new_factors)


        #lets evaluate the new model with real-time updated factors
        map2 = evaluator.evaluate_model(tf, testing)
        #The difference should be smaller than 20%
        assert abs(map1[0]-map2[0]) < 0.2*map1[0]

if __name__ == "__main__":
    unittest.main()