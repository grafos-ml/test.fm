__author__ = 'linas'

from random import random
from interface import ModelInterface


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

class IdModel(ModelInterface):
    '''
    Returns the score as the id of the item.
    Used for testing purposes
    '''

    def getScore(self,user,item):
        return int(item)

