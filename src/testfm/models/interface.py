# -*- coding: utf-8 -*-
'''
Created on 16 January 2014

Interfaces for models

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
'''
__author__ = 'joaonrb'

import numpy as np
from pandas import DataFrame
from sklearn.gaussian_process import GaussianProcess

class ModelInterface(object):
    '''
    
    '''

    def getName(self):
        '''
        Get the informative name for the model.
        :return:
        '''
        return self.__class__.__name__

    def getScore(self,user,item):
        '''
        A score for a user and item that method predicts.
        :param user: id of the user
        :param item: id of the item
        :return:
        '''
        raise NotImplementedError


    def fit(self,training_dataframe):
        '''

        :param training_dataframe: DataFrame a frame with columns 'user', 'item'
        :return:
        '''
        raise NotImplementedError
