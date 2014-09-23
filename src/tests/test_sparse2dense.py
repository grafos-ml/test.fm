__author__ = 'linas'

from timeit import Timer
import numpy as np
import pandas as pd
import unittest

from testfm.models.theano_models import TheanoModel


def numpy_way_reuse(training_set_x, batch_size, n_train_batches,  arr):
    '''the way I do it now'''
    for index in xrange(n_train_batches):
        batch = training_set_x[index * batch_size: (index + 1) * batch_size]
        if index != n_train_batches - 1:
            dense = batch.todense(out=arr)
        else:
            dense = batch.todense()

def numpy_way(training_set_x, batch_size, n_train_batches, arr):
    '''the way I do it now'''
    for index in xrange(n_train_batches):
        batch = training_set_x[index * batch_size: (index + 1) * batch_size].todense()


class TFIDTest(unittest.TestCase):

    def test_reuse_memory(self):
        '''test if reusing of the array is faster than not reusing'''
        batch_size = 100
        df = pd.read_csv("/Users/mumas/devel/data-mac/1M_movielens/ratings.dat",
                            sep=" ", header=None, names=["user", "item", "rating", "date"])
        tm = TheanoModel()
        training_set_x, uid_map, iid_map, user_data = tm._convert(df)
        n_train_batches = len(df.user.unique()) / batch_size

        arr = np.zeros((batch_size, len(df.item.unique())))
        t1 = Timer(lambda: numpy_way(training_set_x, batch_size, n_train_batches,  arr))
        t2 = Timer(lambda: numpy_way_reuse(training_set_x, batch_size, n_train_batches, arr))

        self.assertTrue(t1.timeit(number=100) > t2.timeit(number=100))

