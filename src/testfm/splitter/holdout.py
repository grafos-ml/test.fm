# -*- coding: utf-8 -*-
"""
Created on 17 of January 2014

Holdout splitter method

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
"""
__author__ = {
    'name':'joaonrb',
    'e-mail': 'joaonrb@gmail.com'
}
__version__ = 1, 0, 0
__since__ = 17, 1, 2014

import pandas as pd
import numpy as np
from testfm.splitter.interface import SplitterInterface
from testfm.settings import USER, DATE

############################################
################# VARIABLES ################
############################################


class HoldoutSplitter(SplitterInterface):
    """
    Holdout class for split. The data passed by the call should have a user
    field, a app field and a date field.
    """
    def __init__(self, sort_by=[DATE]):
        self._sort_by = sort_by

    def split(self, data_list, fraction, clean_not_seen=True):
        """
        Splits every list in data_list by fraction and return 2 data frames
        (training and tests) according the holdout method.
        """
        rows = data_list.shape[0]
        i = int(rows * fraction)
        df_training = data_list.head(i)
        df_testing = data_list.tail(rows-i)
        if clean_not_seen:
            df_testing = df_testing[df_testing.user.isin(df_training.user)]
            df_testing = df_testing[df_testing.item.isin(df_training.item)]
        return df_training, df_testing

    def sort(self, data):
        """
        Doesnt do any sorting
        """
        return data.sort(self._sort_by)


class RandomSplitter(HoldoutSplitter):
    """
    Split the data in a random way
    """

    def split(self, df, fraction, clean_not_seen=True):
        """

        """
        n = int(len(df)*fraction)
        sampler = np.random.permutation(df.shape[0])
        training_idx = sampler[:n]
        testing_idx = sampler[n:]

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

    def sort(self, data):
        """

        """
        return data


class RandomHoldoutSplitter(HoldoutSplitter):
    """
    It randomly takes elements for tests
    """

    def sort(self, data):
        """
        Doesnt do any sorting
        """
        df = data.copy()
        df.apply(np.random.shuffle)
        return df


class HoldoutSplitterByUser(HoldoutSplitter):
    """
    Takes fraction from each user
    """

    def __init__(self, sort_by=[DATE, USER]):
        super(HoldoutSplitterByUser, self).__init__(sort_by=sort_by)

    def sort(self, data):
        """

        """
        return data.sort(self._sort_by)