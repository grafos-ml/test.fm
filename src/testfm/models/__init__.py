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

    def get_name(self):
        """
        Get the informative name for the model.
        :return:
        """
        return self.__class__.__name__

    def get_score(self, user, item):
        """
        A score for a user and item that method predicts.
        :param user: id of the user
        :param item: id of the item
        :return:
        """
        raise NotImplementedError

    def train(self, training_data):
        """
        Train the model with numpy array. The first column is for users, the second for item and the third for rating.

        :param training_data: A numpy array
        """
        raise NotImplemented

    def fit(self, training_data):
        """
        Train the data from a pandas.DataFrame
        :param training_data: DataFrame a frame with columns "user", "item"
        """
        users_unique = training_data[self.get_user_column()].unique()
        items_unique = training_data[self.get_item_column()].unique()
        self.data_map = {
            self.get_user_column(): pd.Series(xrange(len(users_unique)), users_unique),
            self.get_item_column(): pd.Series(xrange(len(items_unique)), items_unique),
        }
        data = [
            map(lambda x: self.data_map[self.get_user_column()][x], training_data[self.get_user_column()].values),
            map(lambda x: self.data_map[self.get_item_column()][x], training_data[self.get_item_column()].values),
            training_data.get(self.get_rating_column(), np.ones((1, len(training_data))))
        ]
        self.train(np.array(data).transpose())
