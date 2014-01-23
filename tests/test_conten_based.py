__author__ = 'linas'

import unittest
import pandas as pd
import testfm

from testfm.models.content_based import LSIModel

class TestLSI(unittest.TestCase):

    def setUp(self):
        self.lsi = LSIModel("title")
        self.df = pd.read_csv('testfm/data/movielenshead.dat', sep="::", header=None, names=['user', 'item', 'rating', 'date', 'title'])

    def test_fit(self):
        self.lsi.fit(self.df)
        self.assertEqual(len(self.lsi._user_representation), len(self.df.user.unique()))
        self.assertEqual(len(self.lsi._item_representation), len(self.df.item.unique()))

    def test_score(self):
        self.lsi.fit(self.df)
        #item in the user profile (Booberang) should have higher prediction than movie not in the profile Rob Roy
        self.assertTrue(self.lsi.getScore(1, 122) > self.lsi.getScore(1, 151))

    def test_user_model(self):
        um = self.lsi._get_user_models(self.df)
        self.assertEqual(um[93], ['collateral', 'man', 'fire'])

    def test_item_model(self):
        im = self.lsi._get_item_models(self.df)
        self.assertEqual(im[122], ['boomerang'])
        self.assertEqual(im[329], ['star', 'trek', 'generations'])

if __name__ == '__main__':
    unittest.main()