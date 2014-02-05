# -*- coding: utf-8 -*-
'''
Created on 20 January 2014

Test for splitters

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
'''
__author__ = {
    'name':'joaonrb',
    'e-mail': 'joaonrb@gmail.com'
}
__version__ = 1,0,0
__since__ = 20,1,2014

import pandas as pd
import testfm
from testfm.config import USER, ITEM, DATE

class TestSplitters(object):
    '''
    
    '''
    @classmethod
    def setup_class(cls):
        cls.data = pd.DataFrame({
            USER: [1,1,1,3,1,3,4,3,5,5,4,6,5,6],
            ITEM: [1,2,3,1,4,2,1,3,1,2,2,1,3,2],
            DATE: [
                838989347, # 5
                838991904, # 7
                838992348, # 9
                838992136, # 8
                838989279, # 4
                838993295, # 11
                838993443, # 12
                838993638, # 13
                838988734, # 3
                838986416, # 1
                838992657, # 10
                838988397, # 2
                838991564, # 6
                838993709  # 14
            ]
        })
        cls.fractions = [0.25,0.95]
        cls.holdout_results = [
            (

                {'date': [838986416, 838988397, 838988734],
                 'app': [2, 1, 1],
                 'user': [5, 6, 5]},
                {'date': [838989279, 838989347, 838991564, 838991904, 838992136,
                          838992348, 838992657, 838993295, 838993443, 838993638,
                          838993709],
                 'app': [4, 1, 3, 2, 1, 3, 2, 2, 1, 3, 2],
                 'user': [1, 1, 5, 1, 3, 1, 4, 3, 4, 3, 6]}
            ),
            (
                {'date': [838986416, 838988397, 838988734, 838989279, 838989347,
                          838991564, 838991904, 838992136, 838992348, 838992657,
                          838993295, 838993443, 838993638],
                 'app': [2, 1, 1, 4, 1, 3, 2, 1, 3, 2, 2, 1, 3],
                 'user': [5, 6, 5, 1, 1, 5, 1, 3, 1, 4, 3, 4, 3]},
                {'date': [838993709],
                 'app': [2],
                 'user': [6]}
            )
        ]

    def test_holdout(self):
        '''
        Tests the holdout class
        '''
        fractions = self.__class__.fractions
        data = self.__class__.data
        results = self.__class__.holdout_results
        for i in xrange(len(fractions)):
            train, test = testfm.split.holdout(data,fractions[i])
            print 'fraction ',fractions[i],'   ',
            assert((train.to_dict(outtype='list'), test.to_dict(outtype='list'))
                   == results[i])
            print 'passed'
