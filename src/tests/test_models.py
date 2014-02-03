__author__ = 'linas'

import unittest
import pandas as pd
import numpy as np


from testfm.models.tensorCoFi import TensorCoFi, JavaTensorCoFi
from testfm.models.baseline_model import IdModel
from testfm.models.ensemble_models import LogisticEnsemble

class TestTensorCofi(unittest.TestCase):

    def setUp(self):
        self.tf = TensorCoFi(dim=2)
        self.df = pd.read_csv('../testfm/data/movielenshead.dat',
                              sep="::", header=None, names=['user', 'item', 'rating', 'date', 'title'])
        self.df = self.df.head(n=100)

    def test_array(self):
        arr, tmap = self.tf._dataframe_to_float_matrix(self.df)


    def test_fit(self):
        self.tf.fit(self.df)
        #item and user are row vectors
        self.assertEqual(len(self.df.user.unique()), self.tf.factors['user'].shape[0])
        self.assertEqual(len(self.df.item.unique()), self.tf.factors['item'].shape[0])
        self.assertEqual(self.tf.user_features[1], [1])
        self.assertEqual(self.tf.item_features[122], [1])

    def test_ids_returns(self):
        inp = [{'user':10, 'item':100}, {'user':10,'item':110}, {'user':12,'item':120}]
        inp = pd.DataFrame(inp)
        self.tf.fit(inp)
        self.assertEquals(self.tf.user_column_names, ['user'])
        self.assertEquals(self.tf.item_column_names, ['item'])
        self.assertEquals(len(self.tf.user_features), 2)
        self.assertEquals(len(self.tf.item_features), 3)

        self.assertEquals(len(self.tf.factors['user']), 2)
        self.assertEquals(len(self.tf.factors['item']), 3)

        uid = self.tf._dmap['user'][10]
        iid = self.tf._dmap['item'][100]
        self.assertEquals(uid, 1)
        self.assertEquals(iid, 1)

        self.assertEquals(len(self.tf.factors['user'][uid]), 2)
        self.assertEquals(len(self.tf.factors['user'][uid]), self.tf._dim)
        self.assertEquals(len(self.tf.factors['item'][iid]), self.tf._dim)


    def test_score(self):
        tf = TensorCoFi(dim=2)
        inp = [{'user':10, 'item':100}, {'user':10,'item':110}, {'user':12,'item':120}]
        inp = pd.DataFrame(inp)
        tf.fit(inp)
        uid = tf._dmap['user'][10]
        iid = tf._dmap['item'][100]
        self.assertEquals(uid, 1)
        self.assertEquals(iid, 1)

        tf.factors['user'][0][0] = 0
        tf.factors['user'][0][1] = 1
        tf.factors['item'][0][0] = 1
        tf.factors['item'][0][1] = 5


        self.assertEqual(0*1+1*5, tf.getScore(10,100))

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