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
from random import sample
from testfm.splitter.interface import SplitterInterface
from testfm.config import USER, DATE

############################################
################# VARIABLES ################
############################################



class HoldoutSplitter(SplitterInterface):
    '''
    Holdout class for split. The dataframe passed by the call should have a user
    field, a app field and a date field.
    '''
    def __init__(self,sortBy=[DATE]):
        self._sortby = sortBy


    def split(self,dataList,fraction, clean_not_seen=True):
        '''
        Splits every list in dataList by fraction and return 2 dataframes
        (training and tests) according the holdout method.
        '''
        training = {k:[] for k in dataList[0].keys()}
        test= training.copy()
        for dataset in dataList:
            for key, value in dataset.items():
                i = int(len(value)*fraction)
                training[key], test[key] = value[:i],value[i:]

        df_testing = pd.DataFrame(test)
        df_training = pd.DataFrame(training)
        if clean_not_seen:
            df_testing = df_testing[df_testing.user.isin(df_training.user)]
            df_testing = df_testing[df_testing.item.isin(df_training.item)]

        return df_training, df_testing

    def sort(self,dataframe):
        '''
        Doesnt do any sorting
        '''
        return [dataframe.sort(self._sortby).to_dict(outtype='list')]


class RandomSplitter(HoldoutSplitter):

    def split(self,df,fraction, clean_not_seen=True):
        n = int(len(df)*fraction)
        sampler = np.random.permutation(df.shape[0])
        training_idx = sampler[:n-1]
        testing_idx = sampler[n+1:]

        training = df.take(training_idx).values
        testing = df.take(testing_idx).values
        df_training = pd.DataFrame(training)
        df_training.columns = df.columns
        df_training.reindex()
        df_testing = pd.DataFrame(testing)
        df_testing.columns = df.columns

        if clean_not_seen:
            df_testing = df_testing[df_testing.user.isin(df_training.user)]
            df_testing = df_testing[df_testing.item.isin(df_training.item)]

        return df_training, df_testing

    def sort(self,dataframe):
        return dataframe

class RandomHoldoutSplitter(HoldoutSplitter):
    '''
    It randomly takes elements for tests
    '''

    def sort(self,dataframe):
        '''
        Doesnt do any sorting
        '''
        df = dataframe.copy()
        df.apply(np.random.shuffle)
        return [df.to_dict(outtype='list')]

class HoldoutSplitterByUser(HoldoutSplitter):
    '''
    Takes fraction from each user
    '''

    def __init__(self,sortBy=[DATE,USER]):
        super(HoldoutSplitterByUser,self).__init__(sortBy=sortBy)

    def sort(self,dataframe):
        '''

        '''
        data = dataframe.sort(self._sortby).to_dict(outtype='list')
        users = {}
        for d in zip(*[data[USER]]+data.values()):
            try:
                users[d[0]].append(d[1:])
            except KeyError:
                users[d[0]]= [d[1:]]
        return [map(zip(data.values(),zip(*users[key]))) for key in users]