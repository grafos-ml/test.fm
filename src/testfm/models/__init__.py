# -*- coding: utf-8 -*-
"""
Created on 16 January 2014
Changed on 16 April 2014

Interfaces for models

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
"""
__author__ = "joaonrb"

import numpy as np
import pandas as pd


class IModel(object):
    """
    Interface class for model
    """

    data_map = None

    @classmethod
    def param_details(cls):
        """
        Return a dictionary with the parameters for the set parameters and
        a tuple with min, max, step and default value.

        {
            'paramA': (min, max, step, default),
            'paramB': ...
            ...
        }
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        Set the parameters in the model.

        kwargs can have an arbitrary set of parameters
        """
        raise NotImplementedError

    @staticmethod
    def get_user_column():
        """
        Get the name of the user column in the pandas.DataFrame
        """
        return "user"

    @staticmethod
    def get_item_column():
        """
        Get the name of the item column in the pandas.DataFrame
        """
        return "item"

    @staticmethod
    def get_rating_column():
        """
        Get the name of the rating column in the pandas.DataFrame
        """
        return "rating"

    @staticmethod
    def get_context_columns(self):
        """
        Get a list of names of all the context column names for this model
        :return:
        """
        raise NotImplemented

    def get_name(self):
        """
        Get the informative name for the model.
        :return:
        """
        return self.__class__.__name__

    def train(self, training_data):
        """
        Train the model with numpy array. The first column is for users, the second for item and the third for rating.

        :param training_data: A numpy array
        """
        raise NotImplemented

    def fit(self, training_data):
        """
        Train the data from a pandas.DataFrame
        :param training_data: DataFrame a frame with columns 'user', 'item', 'contexts'..., 'rating'
        If rating don't exist it will be populated with 1 for all entries
        """
        columns = [self.get_user_column(), self.get_item_column()] + self.get_context_columns()
        data = []
        self.data_map = {}
        for column in columns:
            unique_data = training_data[column].unique()
            self.data_map[column] = pd.Series(xrange(1, len(unique_data)+1), unique_data)
            data.append(map(lambda x: self.data_map[column][x], training_data[column].values))
        data.append(training_data.get(self.get_rating_column(), np.ones((len(training_data,))).tolist()))
        self.train(np.array(data).transpose())

    def users_size(self):
        """
        Return the number of users
        """
        return len(self.data_map[self.get_user_column()])

    def items_size(self):
        """
        Return the number of items
        """
        return len(self.data_map[self.get_item_column()])
