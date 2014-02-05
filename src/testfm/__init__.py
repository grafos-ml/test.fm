# -*- coding: utf-8 -*-
'''
Created on 17 January 2014



.. moduleauthor:: joaonrb <>
'''
__author__ = {
    'name':'joaonrb',
    'e-mail': 'joaonrb@gmail.com'
}
__version__ = 1,0,0
__since__ = 17,1,2014


from testfm.splitter.holdout import HoldoutSplitter, HoldoutSplitterByUser, \
    RandomHoldoutSplitter, RandomSplitter
from evaluation.measures import MAPMeasure
from evaluation.evaluator import Evaluator

class split(object):
    holdout = HoldoutSplitter()
    holdoutByRandomSlow = RandomHoldoutSplitter()
    holdoutByUser = HoldoutSplitterByUser()
    holdoutByRandom = RandomSplitter()

def evaluate_model(factor_model, testing_dataframe, measures=[MAPMeasure()],
    all_items=None, non_relevant_count=100):
    eval = Evaluator()
    return eval.evaluate_model_multiprocessing(factor_model, testing_dataframe,
        measures=measures, all_items=all_items,
        non_relevant_count=non_relevant_count)