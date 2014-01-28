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
        data, tmap = self.tf._map(self.df)
        tensor = JavaTensorCoFi(2,1,self.tf._lamb,self.tf._alph,
            [len(tmap[USER]),len(tmap[ITEM])])
        tensor.train(data)
        final_model = tensor.getModel()
        t0 = np.fromiter(final_model.get(0).toArray(),dtype=np.float)
        t0.shape = final_model.get(0).rows, final_model.get(0).columns

        users = np.matrix(t0)

        self.assertEqual(users[0,0], final_model.get(0).get(0,0))
        print users
        print final_model.get(0).toString()
        #self.assertEqual(users[1,1], final_model.get(0).get(1,1))

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