"""
Created on 12 February 2014



.. moduleauthor:: Alexandros Karatzoglou
"""

__author__ = "alexis"



import numpy as np
import random
from testfm.models.cutil.interface import IModel


class BPR(IModel):

    def __init__(self, eta=0.03, reg=0.0001, dim=20, n_iter=30):
        self.set_params(eta, reg, dim, n_iter)
        self.U = {}
        self.M = {}

    def set_params(self, eta=0.03, reg=0.0001, dim=20, n_iter=25):
        """
        Set the parameters for the BPR model
        """
        self._dim = int(dim)
        self._n_iter = int(n_iter)
        self._reg = float(reg)
        self._eta = float(eta)
        self.U = {}
        self.M = {}

    @classmethod
    def param_details(cls):
        """
        Return parameter details for dim, nIter, reg, eta
        """
        return {
                "dim": (10, 40, 4, 20),
                "n_iter": (10, 35, 3, 15),
                "reg": (0.0001, 0.01, .001, .0001),
                "eta": (0.01, 0.08, 0.005, 0.03)
        }

    def fit(self, data):
        """
        Train the model
        """
        data_array = data[["user", "item"]]
        items = data.item.unique()

        for iter in range(self._n_iter):
            for idx, row in data_array.iterrows():
                self._additiveupdate(row, items)

    def _additiveupdate(self, row, items):

        #take the factors for user, item and negative item
        u = self.U.get(row["user"], self._init_vector())
        m = self.M.get(row["item"], self._init_vector())
        rand = random.choice(items)
        m_neg = self.M.get(rand, self._init_vector())

        #do updates
        hscore = np.dot(u, m) - np.dot(u, m_neg)
        ploss = self.compute_partial_loss(0, hscore)
        # update user
        u -= self._eta * ((ploss * (m - m_neg)) - self._reg * u)

        #update positive item
        m -= self._eta*((ploss * u) - self._reg * m)

        #update negative item
        m_neg -= self._eta*((ploss * (-u)) -  self._reg* m_neg)

        self.U[row["user"]] = u
        self.M[row["item"]] = m
        self.M[rand] = m_neg

    def get_score(self, user, item):
        return np.dot(self.U[user], self.M[item])

    def get_name(self):
        return "BPR (dim={},iter={},reg={},eta={})".format(self._dim, self._n_iter, self._reg, self._eta)

    def compute_partial_loss(self, y, f):
        """

        :param y:
        :param f:
        :return:
        """
        return self.gdiff(f)

    def gdiff(self, score):
        """

        :param score:
        :return:
        """
        exponential =  np.exp(-score)
        return - exponential/(1.0 + exponential)

    def _init_vector(self):
        return np.random.normal(0, 2.5/self._dim, size=self._dim)
