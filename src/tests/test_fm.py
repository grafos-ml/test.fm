# -*- coding: utf-8 -*-
"""
Created on 20 January 2014

Test for splitters

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
"""
__author__ = {
    'name': 'joaonrb',
    'e-mail': 'joaonrb@gmail.com'
}
__version__ = 1, 0, 0
__since__ = 20, 1, 2014

import pandas as pd
import testfm
from testfm.settings import USER, ITEM, DATE
from pandas.util.testing import assert_frame_equal


class TestSplitters(object):
    """

    """
    @classmethod
    def setup_class(cls):
        cls.data = pd.DataFrame({
            USER: [1, 1, 1, 3, 1, 3, 4, 3, 5, 5, 4, 6, 5, 6],
            ITEM: [1, 2, 3, 1, 4, 2, 1, 3, 1, 2, 2, 1, 3, 2],
            DATE: [
                838989347,  # 5
                838991904,  # 7
                838992348,  # 9
                838992136,  # 8
                838989279,  # 4
                838993295,  # 11
                838993443,  # 12
                838993638,  # 13
                838988734,  # 3
                838986416,  # 1
                838992657,  # 10
                838988397,  # 2
                838991564,  # 6
                838993709   # 14
            ]
        })
        cls.fractions = [0.25, 0.95]
        cls.holdout_results = [
            (
                {DATE: [838986416, 838988397, 838988734],
                 ITEM: [2, 1, 1],
                 USER: [5, 6, 5]},
                #{DATE: [838989279, 838989347, 838991564, 838991904, 838992136,
                #          838992348, 838992657, 838993295, 838993443, 838993638,
                #          838993709],
                # 'app': [4, 1, 3, 2, 1, 3, 2, 2, 1, 3, 2],
                # 'user': [1, 1, 5, 1, 3, 1, 4, 3, 4, 3, 6]}
                {DATE: [838993709],
                 ITEM: [2],
                 USER: [6]}
            ),
            (
                {DATE: [838986416, 838988397, 838988734, 838989279, 838989347,
                        838991564, 838991904, 838992136, 838992348, 838992657,
                        838993295, 838993443, 838993638],
                 ITEM: [2, 1, 1, 4, 1, 3, 2, 1, 3, 2, 2, 1, 3],
                 USER: [5, 6, 5, 1, 1, 5, 1, 3, 1, 4, 3, 4, 3]},
                {DATE: [838993709],
                 ITEM: [2],
                 USER: [6]}
            )
        ]

    def test_holdout(self):
        """
        Tests the holdout class
        """
        fractions = self.__class__.fractions
        data = self.__class__.data
        results = self.__class__.holdout_results
        for i, fraction in enumerate(fractions):
            train, test = testfm.split.holdout(data, fraction)
            print 'fraction ', fraction, '   ',
            assert((train.to_dict(outtype='list'), test.to_dict(outtype='list'))
                   == results[i])
            print 'passed'

    def test_random(self):
        fractions = self.__class__.fractions
        data = self.__class__.data
        rows = len(data[USER])
        for fraction in fractions:
            train, test = testfm.split.holdoutByRandom(data, fraction,
                                                       clean_not_seen=False)
            i = int(rows * fraction)
            print 'fraction ', fraction, '   ',

            print((train.shape[0], test.shape[0]), (i, rows-i))
            # Test if the number of elements in training correspond to the
            # "fraction" of the data and the test correspond to the rest
            assert((train.shape[0], test.shape[0]) == (i, rows-i)), \
                "Number of rows expected is wrong"

            # Test if every element in the test plus train are in the full data
            # set.
            assert_frame_equal(
                pd.concat((train, test)).sort([DATE]).set_index([DATE]),
                data.sort([DATE]).set_index([DATE]))

            print 'passed'

