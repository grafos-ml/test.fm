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
		


    def setParams(self,eta = 0.001, reg = 0.0001, dim = 10, nIter = 15):
        """
        Set the parameters for the BPR model
        """
        self._dim = dim
        self._nIter = nIter
        self._reg = reg
        self._eta = eta
        self._model = DictModel(10)
       

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

    def _fit(self,data):
        """
        Train the model
        """
        """ 
        Get user item and convert to numby array
        """
        
        datarray =  data[["user","item"]].values

        #initialize item keys, we need this to sample negative items
        for row in datarray:
                tmp = self._model.getM(row[1])

        #iterate over rows
        for row in datarray:
                self._additiveupdate(row)


    def fit(self,data):
        """
        Get the model
        """
        self._fit(data)


    def _additiveupdate(self, row):

        u = self._model.getU(row[0])
        m = self._model.getM(row[1])
        mneg = self._model.getM(random.choice(self._model.getAllMIDs()))
        hscore = np.dot(u,m) - np.dot(u, mneg)
        ploss = self.computePartialLoss(0, hscore)

        # update user
        u -= self._eta * ((ploss * (m - mneg)) + self._reg * u) 

        #update positive item
        m -= self._eta*((ploss * u) +  self._reg* m)

        #update negative item
        mneg -= self._eta*((ploss * (-u)) +  self._reg* m)        


    def getScore(self, user, item):
        return np.dot(self._model.getU(user), self._model.getM(item))

    def getName(self):
        return "BPR (dim={},iter={},reg={},eta={})".format(
            self._dim, self._nIter, self._reg, self._eta)

    def computePartialLoss(self, y, f):

        return self.gdiff(f)

    def  gdiff(self, score):

        exponential = np.exp(- score)
        return exponential/(1.0 + exponential)
         

class DictModel(object):

    def __init__(self, nFactors): 
        self.__u = {}
        self.__m = {}
        self.__nFactors = nFactors

    def getM(self, j):
        try:
            return self.__m[j]
        except KeyError:
            self.__m[j] = self.__initVector()
            return self.__m[j]

    def getU(self, i):
        try:
            return self.__u[i]
        except KeyError:
           
            self.__u[i] = self.__initVector()
            return self.__u[i]

    def getEmptyVector(self):
        return numpy.zeros(self.__nFactors)

    def getAllMIDs(self):
        return self.__m.keys()

    def getAllUIDs(self):
        return self.__u.keys()
        
    def __initVector(self):
        return np.random.normal(0, 2.5/self.__nFactors, size=self.__nFactors)



    

