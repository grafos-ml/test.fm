# -*- coding: utf-8 -*-
"""
Created on 23 January 2014

Base line model

.. moduleauthor:: Linas
"""
__author__ = 'linas'

from random import random
from testfm.models.interface import ModelInterface
from math import log
import numpy as np

class RandomModel(ModelInterface):
    """
    Random model
    """
    _scores = {}

    def getScore(self, user, item):
        key = user, item
        try:
            return self._scores[key]
        except KeyError:
            self._scores[key] = random()
            return self._scores[key]

    def fit(self,training_data):
        pass

    def getName(self):
        return "Random"

class IdModel(ModelInterface):
    """
    Returns the score as the id of the item.
    Used for testing purposes
    """

    def getScore(self, user, item):
        return int(item)

    def fit(self, training_data):
        pass

    def getName(self):
        return "ItemID"


class ConstantModel(ModelInterface):
    """
    Returns constant for all predictions.
    Don't use this model in any comparison, because the algorithm will tell its
    a perfect model (just because of the evaluation implementation)
    """
    _c = 1.0

    def __init__(self, constant=1.0):
        self._c = constant

    def getScore(self, user, item):
        return self._c

    def getName(self):
        return "Constant %d" % self._c


class Item2Item(ModelInterface):

    k = 5
    _items = {}
    _users = {}



    def compute_jaccard_index(self, set_1, set_2):
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
        self._items = \
            {item: set(entries)
             for item, entries in training_data.groupby('item')['user']}
        self._users = \
            {user: set(entries)
             for user, entries in training_data.groupby('user')['item']}

    def getScore(self,user,item):
        # scores = [self.similarity(i, item)
        #           for i in self._users[user] if i != item]
        # scores.sort(reverse=True)
        # assert scores == sorted((self.similarity(i, item)
        #           for i in self._users[user] if i != item),
        #                         cmp=lambda x, y: cmp(y,x))

        # Returns the sum of the list whit self.k elements the sorted similarity
        # between items of the user and item(param) excluding the item(param)
        # itself.
        return sum(sorted((self.similarity(i, item)
                           for i in self._users[user] if i != item),
                          cmp=lambda x, y: cmp(y, x))[:self.k])

    def setParams(self, k):
        """
        :param k int how many closest items in the user profile to consider.
        """
        self.k = k

    @classmethod
    def paramDetails(cls):
        """
        Return parameter details for k.
        """
        return {
            'k': (1, 50, 2, 5),
        }

class AverageModel(ModelInterface):
    _avg = {}

    def fit(self, training_data):
        """
        Computes average rating of the item..
        :param training_data: DataFrame training data
        :return:
        """
        movie_stats = training_data.groupby('item').agg({'rating': [np.mean]})
        self._avg = {
            i: m[0]
            for i, m in movie_stats.iterrows()
        }

    def getScore(self, user, item):
        return self._avg[item]


class Popularity(ModelInterface):

    _counts = {}
    mn = float("inf")
    mx = 0.


    def getScore(self, user, item):
        cnt = self._counts.get(item, 0.0)
        # normalize between 0 and 1
        try:
            assert isinstance(self.mx - self.mn, float), 'Not float'
            return (cnt - self.mn)/(self.mx - self.mn)
        except ZeroDivisionError:
            return cnt

    def fit(self, training_data):
        """
        Computes number of times the item was used by a user.
        :param training_data: DataFrame training data
        :return:
        """
        # self.mn = float("inf")  # This is not needed
        # self.mx = 0.  # Neither do this
        # for i, v in training_data.item.value_counts().iteritems():
        #     self._counts[i] = cnt = log(v+1)
        #     self.mn = min(self.mn, cnt)
        #     self.mx = max(self.mx, cnt)

        # Changed to fit behave like other models. Every time it fits it loses
        # the old data.

        # Linas, do you mind this?
        self._counts = {
            i: log(v+1)
            for i, v in training_data.item.value_counts().iteritems()
        }
        s = sorted(self._counts.values())
        self.mn, self.mx = s[0], s[-1]

    def getName(self):
        return "Popularity"


class PersonalizedPopularity(ModelInterface):

    _counts = {}

    def getScore(self, user,  item):

        try:
            return float(self._counts[user][item])
        except KeyError:
            return 0.0

    def fit(self, training_data):
        #add date dependency
        # normalize ?
        for useritem, count in training_data.groupby(['user']).item.value_counts().iteritems():
             try:
                self._counts[useritem[0]].update({useritem[1]:count})
             except KeyError:
                self._counts.update({useritem[0]:{useritem[1]:count}})

    def getName(self):
        return "PersonalizedPopularity"





