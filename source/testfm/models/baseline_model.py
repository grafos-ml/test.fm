__author__ = 'linas'

from random import random
from interface import ModelInterface
from pandas import DataFrame

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


class Popularity(ModelInterface):

    _counts = {}

    def getScore(self,user,item):
        return self._counts.get(item, 0.0)

    def fit(self, training_dataframe):
        '''
        Computes number of times the item was used by a user.
        :param training_dataframe: DataFrame training data
        :return:
        '''
        for i,v in training_dataframe.item.value_counts().iteritems():
            self._counts[i] = v

    def getName(self):
        return "Popularity"