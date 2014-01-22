__author__ = 'linas'

import unittest
import pandas as pd

from testfm.models.content_based import LSIModel

class TestLSI(unittest.TestCase):

    def setUp(self):
        self.lsi = LSIModel("title")
        self.df = pd.read_csv('../../../data/movielenshead.dat', sep="::", header=None, names=['user', 'item', 'rating', 'date', 'title'])
        self.lsi.fit(self.df)

    def test_fit(self):
        self.assertEqual(len(self.lsi._user_representation), len(self.df.user.unique()))
        self.assertEqual(len(self.lsi._item_representation), len(self.df.item.unique()))

    def test_self_sim(self):
        #print self.df.head(n=21)
        #item in the user profile (Booberang) should have higher prediction than movie not in the profile Rob Roy
        self.assertTrue(self.lsi.getScore(1, 122) > self.lsi.getScore(1, 151))

if __name__ == '__main__':
    unittest.main()