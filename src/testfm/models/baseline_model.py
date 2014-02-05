__author__ = 'linas'

from random import random
from testfm.models.interface import ModelInterface
from math import log

class RandomModel(ModelInterface):
    _scores = {}

    def getScore(self,user,item):
        key = (user, item)
        if key in self._scores:
            return self._scores[key]
        else:
            s = random()
            self._scores[key] = s
            return s

    def fit(self,training_dataframe):
        pass

    def getName(self):
        return "Random"

class IdModel(ModelInterface):
    '''
    Returns the score as the id of the item.
    Used for testing purposes
    '''

    def getScore(self,user,item):
        return int(item)

    def fit(self,training_dataframe):
        pass

    def getName(self):
        return "ItemID"


class ConstantModel(ModelInterface):
    '''
    Returns constant for all predictions.
    Don't use this model in any comparison, because the algorithm will tell its a perfect model (just because
    of the evaluation implementation)
    '''
    _c = 1.0

    def __init__(self, constant=1.0):
        self._c = constant

    def getScore(self,user,item):
        return self._c

    def getName(self):
        return "Constant "+str(self._c)


class Item2Item(ModelInterface):

    k = 5

    def compute_jaccard_index(self, set_1, set_2):
        n = len(set_1.intersection(set_2))
        return n / float(len(set_1) + len(set_2) - n)

    def similarity(self, i1, i2):
        return self.compute_jaccard_index(self._items[i1], self._items[i2])

    def fit(self, training_dataframe):
        '''
        Stores set of user ids for each item
        '''
        self._items = {}
        self._users = {}
        grouped = training_dataframe.groupby('item')['user']
        for item, entries in grouped:
            users = set(entries)
            self._items[item] = users

        grouped = training_dataframe.groupby('user')['item']
        for user, entries in grouped:
            items = set(entries)
            self._users[user] = items

    def getScore(self,user,item):
        scores = [self.similarity(i, item) for i in self._users[user] if i != item]
        scores.sort(reverse=True)
        return sum(scores[:self.k])

class Popularity(ModelInterface):

    _counts = {}

    def getScore(self,user,item):
        cnt = self._counts.get(item, 0.0)
        #normalize between 0 and 1
        try:
            norm = (cnt - self.mn)/float(self.mx - self.mn)
            return norm
        except:
            return cnt

    def fit(self, training_dataframe):
        '''
        Computes number of times the item was used by a user.
        :param training_dataframe: DataFrame training data
        :return:
        '''
        self.mn = float("inf")
        self.mx = 0
        for i,v in training_dataframe.item.value_counts().iteritems():
            cnt = log(v+1)
            self._counts[i] = cnt
            self.mn = min(self.mn, cnt)
            self.mx = max(self.mx, cnt)

    def getName(self):
        return "Popularity"


