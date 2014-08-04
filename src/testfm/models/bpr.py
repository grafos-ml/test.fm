"""
Created on 12 February 2014



.. moduleauthor:: Alexandros
"""

__author__ = 'alexis'

import numpy as np
import random
from interface import ModelInterface

class BPR(ModelInterface):

    def __init__(self, eta = 0.001, reg = 0.0001, dim = 10, nIter = 15):
                
        self.setParams(eta, reg, dim, nIter)
        self.U = {}
        self.M = {}


    def setParams(self,eta = 0.01, reg = 0.0001, dim = 10, nIter = 15):
        """
        Set the parameters for the BPR model
        """
        self._dim = dim
        self._nIter = nIter
        self._reg = reg
        self._eta = eta

    @classmethod
    def paramDetails(cls):
        """
        Return parameter details for dim, nIter, reg, eta
        """
        return {
                'dim': (10, 20, 4, 15),
                'nIter': (10, 15, 20, 24),
                'reg': (000.1, 0.0001, .001, .0005),
                'eta': (0.01, 0.03, 0.004, 0.0009)
                }

    def fit(self, data):
        """
        Train the model
        """

        datarray =  data[["user","item"]]
        items = data.item.unique()

        for iter in range(self._nIter):
            for idx, row in datarray.iterrows():
                self._additiveupdate(row, items)


    def _additiveupdate(self, row, items):

        #take the factors for user, item and negative item
        u = self.U.get(row['user'], self._initVector())
        m = self.M.get(row['item'], self._initVector())
        rand = random.choice(items)
        m_neg = self.M.get(rand, self._initVector())

        #do updates
        hscore = np.dot(u,m) - np.dot(u, m_neg)
        ploss = self.computePartialLoss(0, hscore)
        # update user
        u += self._eta * ((ploss * (m - m_neg)) + self._reg * u)

        #update positive item
        m += self._eta*((ploss * u) +  self._reg* m)

        #update negative item
        m_neg += self._eta*((ploss * (-u)) +  self._reg* m_neg)

        self.U[row['user']] = u
        self.M[row['item']] = m
        self.M[rand] = m_neg

    def getScore(self, user, item):
        return np.dot(self.U[user], self.M[item])

    def getName(self):
        return "BPR (dim={},iter={},reg={},eta={})".format(
            self._dim, self._nIter, self._reg, self._eta)

    def computePartialLoss(self, y, f):

        return self.gdiff(f)

    def  gdiff(self, score):

        exponential = np.exp(- score)
        return exponential/(1.0 + exponential)

    def _initVector(self):
        return np.random.normal(0, 2.5/self._dim, size=self._dim)
    

