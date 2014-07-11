# -*- coding: utf-8 -*-
'''
Created on 17 January 2014



.. moduleauthor:: joaonrb <>
'''
__author__ = {
    'name':'joaonrb',
    'e-mail': 'joaonrb@gmail.com'
}
__version__ = 1, 0, 3
__since__ = 17, 1, 2014


from testfm.splitter.holdout import HoldoutSplitter, HoldoutSplitterByUser, \
    RandomHoldoutSplitter, RandomSplitter


class split(object):
    holdout = HoldoutSplitter()
    holdoutByRandomSlow = RandomHoldoutSplitter()
    holdoutByUser = HoldoutSplitterByUser()
    holdoutByRandom = RandomSplitter()