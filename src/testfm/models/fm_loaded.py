# -*- coding: utf-8 -*-
"""
Created on 25 January 2014

Factor Model

.. moduleauthor:: Linas
"""

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

    def getScore(self,user, item):
        try:
            return vdot(self._users[user], self._items[item])
        except KeyError:
            return 0.0
