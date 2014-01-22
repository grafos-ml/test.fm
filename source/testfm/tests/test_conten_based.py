__author__ = 'linas'

import unittest
import pandas as pd

from testfm.models.content_based import LSIModel

class TestLSI(unittest.TestCase):

    def setUp(self):
        self.lsi = LSIModel("title")
        self.df = pd.read_csv('../../../data/movielenshead.dat', sep="::", header=None, names=['user', 'item', 'rating', 'date', 'title'])

    def test_fit(self):
        self.lsi.fit(self.df)
        self.assertEqual(len(self.lsi._user_representation), len(self.df.user.unique()))
        self.assertEqual(len(self.lsi._item_representation), len(self.df.item.unique()))

if __name__ == '__main__':
    unittest.main()