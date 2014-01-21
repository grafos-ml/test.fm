# -*- coding: utf-8 -*-
'''
Created on 17 of January 2014

Holdout splitter method

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
'''
__author__ = {
    'name':'joaonrb',
    'e-mail': 'joaonrb@gmail.com'
}
__version__ = 1,0,0
__since__ = 17,1,2014

import pandas as pd
import numpy as np
from interface import SplitterInterface

############################################
################# VARIABLES ################
############################################

USER = 'user'
APP = 'app'
DATE = 'date'

class HoldoutSplitter(SplitterInterface):
    '''
    Holdout class for split. The dataframe passed by the call should have a user
    field, a app field and a date field.
    '''
    def split(self,dataList,fraction):
        '''
        Splits every list in dataList by fraction and return 2 dataframes
        (training and test) according the holdout method.
        '''
        training = {k:[] for k in dataList[0].keys()}
        test= training.copy()
        for dataset in dataList:
            for key, value in dataset.items():
                i = int(len(value)*fraction)
                training[key], test[key] = value.values()[:i], \
                    value.values()[i:]

        return pd.DataFrame(training), pd.DataFrame(test)

    def sort(self,dataframe):
        '''
        Doesnt do any sorting
        '''
        return [dataframe.sort([DATE]).to_dict()]


class RandomHoldoutSplitter(HoldoutSplitter):
    '''
    It randomly takes elements for test
    '''

    def sort(self,dataframe):
        '''
        Doesnt do any sorting
        '''
        df = dataframe.copy()
        df.apply(np.random.shuffle)
        return [df.to_dict()]

class HoldoutSplitterByUser(HoldoutSplitter):
    '''
    Takes fraction from each user
    '''

    def sort(self,dataframe):
        '''

        '''
        data = dataframe.sort([DATE,USER]).to_dict()
        users = {}
        for user,app,date in zip(data[USER],data[APP],data[DATE]):
            try:
                users[user].append((user,app,date))
            except KeyError:
                users[user]= [(user,app,date)]
        return [map(zip([USER,APP,DATE],zip(*users[key]))) for key in users]