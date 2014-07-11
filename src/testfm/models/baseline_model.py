# -*- coding: utf-8 -*-
"""
Created on 23 January 2014

Base line model

.. moduleauthor:: Linas
"""
__author__ = "linas"

from random import random
from testfm.models.cutil.interface import IModel
from math import log
import numpy as np
from testfm.models.cutil.baseline_model import NOGILRandomModel


class RandomModel(NOGILRandomModel):
    """
    Random model
    """
    _scores = {}

    def get_score(self, user, item, **context):
        key = user, item
        try:
            return self._scores[key]
        except KeyError:
            self._scores[key] = random()
            return self._scores[key]

    def get_name(self):
        return "Random"


class IdModel(IModel):
    """
    Returns the score as the id of the item.
    Used for testing purposes
    """

    def get_score(self, user, item, **context):
        return int(item)

    def fit(self, training_data):
        pass

    def get_name(self):
        return "ItemID"


class ConstantModel(IModel):
    """
    Returns constant for all predictions.
    Don't use this model in any comparison, because the algorithm will tell its
    a perfect model (just because of the evaluation implementation)
    """
    _c = 1.0

    def __init__(self, constant=1.0):
        self._c = constant

    def get_score(self, user, item, **context):
        return self._c

    def get_name(self):
        return "Constant %d" % self._c


class Item2Item(IModel):

    k = 5
    _items = {}
    _users = {}

    @staticmethod
    def compute_jaccard_index(set_1, set_2):
        """
        Computes the Jaccard index for similarity measure between set 1 and set
        2.
        """
        n = len(set_1.intersection(set_2))
        return n / float(len(set_1) + len(set_2) - n)

    def similarity(self, i1, i2):
        """
        Measures the similarity between 2 sets
        """
        return self.compute_jaccard_index(self._items[i1], self._items[i2])

    def fit(self, training_data):
        """
        Stores set of user ids for each item
        """
        self._items = {item: set(entries) for item, entries in training_data.groupby('item')['user']}
        self._users = {user: set(entries) for user, entries in training_data.groupby('user')['item']}

    def get_score(self, user, item, **context):
        """
        Returns the sum of the list whit self.k elements the sorted similarity between items of the user and item(param)
        excluding the item(param) itself.
        """
        scores = (self.similarity(i, item) for i in self._users[user] if i != item)
        return sum(sorted(scores, cmp=lambda x, y: cmp(y, x))[:self.k])

    def set_params(self, k):
        """
        :param k int how many closest items in the user profile to consider.
        """
        self.k = k

    @classmethod
    def param_details(cls):
        """
        Return parameter details for k.
        """
        return {
            'k': (1, 50, 2, 5),
        }


class AverageModel(IModel):
    _avg = {}

    def fit(self, training_data):
        """
        Computes average rating of the item..
        :param training_data: DataFrame training data
        :return:
        """
        movie_stats = training_data.groupby('item').agg({'rating': [lambda x: np.mean(x.apply(lambda y: float(y)))]})
        self._avg = {
            i: m[0]
            for i, m in movie_stats.iterrows()
        }

    def get_score(self, user, item, **context):
        return self._avg[item]


class Popularity(IModel):

    _counts = {}
    mn = float("inf")
    mx = 0.

    def __init__(self, normalize=True):
        self.normalize = normalize

    def get_score(self, user, item, **context):
        #return self._counts.get(item, 0.0)
        return self._counts[item]

    def fit(self, training_data):
        """
        Computes number of times the item was used by a user.
        :param training_data: DataFrame training data
        :return:
        """
        if self.normalize:
            self._counts = {i: log(v+1) for i, v in training_data.item.value_counts().iteritems()}
            s = sorted(self._counts.values())
            mn, mx = s[0], s[-1]
            for k in self._counts.keys():
                self._counts[k] = (self._counts[k]-mn)/(mx-mn)
        else:
            self._counts = {i: v for i, v in training_data.item.value_counts().iteritems()}

    def get_name(self):
        return "Popularity"


class PersonalizedPopularity(IModel):

    _counts = {}

    def get_score(self, user, item, **context):
        try:
            return float(self._counts[user][item])
        except KeyError:
            return 0.0

    def fit(self, training_data):
        #add date dependency
        # normalize ?
        for useritem, count in training_data.groupby(['user']).item.value_counts().iteritems():
            try:
                self._counts[useritem[0]].update({useritem[1]: count})
            except KeyError:
                self._counts.update({useritem[0]: {useritem[1]: count}})

    def get_name(self):
        return "PersonalizedPopularity"