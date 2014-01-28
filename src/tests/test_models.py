__author__ = 'linas'

import unittest
import pandas as pd
import numpy as np

from testfm.models.tensorCoFi import TensorCoFi, JavaTensorCoFi
from testfm.config import USER, ITEM

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

if __name__ == '__main__':
    unittest.main()