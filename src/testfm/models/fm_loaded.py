__author__ = 'linas'

from interface import ModelInterface
from numpy import vdot

class FactorModel(ModelInterface):
    _users = {}
    _items = {}

    def __init__(self, userf, itemf):
        self._users = userf
        self._items = itemf


    def getName(self):
        return 'FactorModel'

    def getScore(self,user,item):
        try:
            return vdot(self._users[user], self._items[item])
        except:
            return 0.0
