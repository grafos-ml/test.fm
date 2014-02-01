__author__ = 'linas'

import unittest
import pandas as pd
import numpy as np


from testfm.models.tensorCoFi import TensorCoFi, JavaTensorCoFi
from testfm.models.baseline_model import IdModel
from testfm.models.ensemble_models import LogisticEnsemble

class TestTensorCofi(unittest.TestCase):

    def setUp(self):
        self.tf = TensorCoFi()
        self.df = pd.read_csv('../testfm/data/movielenshead.dat',
                              sep="::", header=None, names=['user', 'item', 'rating', 'date', 'title'])
        self.df = self.df.head(n=100)

    def test_fit(self):
        self.tf.fit(self.df)
        #item and user are row vectors
        self.assertEqual(len(self.df.user.unique()), self.tf._users.shape[0])
        self.assertEqual(len(self.df.item.unique()), self.tf._items.shape[0])

    def test_score(self):
        tf = TensorCoFi()
        tf.fit(self.df)
        tf._users = np.arange(12).reshape(3,4)
        tf._items = np.array([1]*12).reshape(3,4)
        self.assertEqual(0+1+2+3, tf.getScore(1,122))

    def test_floatmatrix_to_numpy(self):
        from jnius import autoclass
        FloatMatrix = autoclass('org.jblas.FloatMatrix')
        rand = FloatMatrix.rand(4,2)
        rand_np = self.tf._float_matrix2numpy(rand)

        self.assertEqual(rand.get(0,0), rand_np[0,0])
        self.assertEqual(rand.get(1,0), rand_np[1,0])
        self.assertEqual(rand.get(1,1), rand_np[1,1])
        self.assertEqual((rand.rows, rand.columns), rand_np.shape)

class LogisticTest(unittest.TestCase):

    def setUp(self):
        self.le = LogisticEnsemble(models=[IdModel()])
        inp = [{'user':10, 'item':100}, {'user':10,'item':110}, {'user':12,'item':120}]
        self.df = pd.DataFrame(inp)

    def test_construct_features(self):

        self.le._prepare_feature_extraction(self.df)
        self.assertEqual(self.le._user_count,  {10: 2, 12: 1})

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
        self.assertIsInstance(self.le.getScore(10, 110), float)

if __name__ == '__main__':
    unittest.main()