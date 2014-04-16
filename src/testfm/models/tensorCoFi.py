# -*- coding: utf-8 -*-
"""
Created on 16 January 2014

Connector for the tensor CoFi Java implementation

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
"""
__author__ = {
    "name":"joaonrb",
    "e-mail": "joaonrb@gmail.com"
}
__version__ = 1, 0, 0
__since__ = 16, 1, 2014

from pkg_resources import resource_filename
import testfm
import os
import numpy as np
import scipy as sp
os.environ["CLASSPATH"] = resource_filename(testfm.__name__, "lib/algorithm-1.0-SNAPSHOT-jar-with-dependencies.jar:") + \
    os.environ.get("CLASSPATH", "")


import datetime
import subprocess

from jnius import autoclass
import pandas as pd
from testfm.models import IModel
from testfm.config import USER, ITEM
import math

JavaTensorCoFi = autoclass("es.tid.frappe.recsys.TensorCoFi")
FloatMatrix = autoclass("org.jblas.FloatMatrix")
Arrays = autoclass("java.util.Arrays")


class TensorCoFi(IModel):

    def __init__(self, dim=20, nIter=5, lamb=0.05, alph=40, user_features=["user"], item_features=["item"]):
        """
        Python model creator fro tensor implementation in java

        **Args**

            *pandas.DataFrame* trainData:
                The data to train the tensor

            *int* dim:
                Dimension of some kind. Default = 20.

            *int* nIter:
                Nmber of iteration. Default = 5.

            *float* lamb:
                Lambda value for the algorithm. Default = 0,05.

            *int* alph:
                Alpha number for the algorithm. Default = 40.

        """
        self.set_params(dim, nIter, lamb, alph)
        self.user_features = {}
        self.item_features = {}
        self.factors = {}

        self.user_column_names = user_features
        self.item_column_names = item_features

    @classmethod
    def param_details(cls):
        """
        Return parameter details for dim, nIter, lamb and alph
        """
        return {
            "dim": (10, 20, 2, 20),
            "nIter": (1, 10, 2, 5),
            "lamb": (.1, 1., .1, .05),
            "alph": (30, 50, 5, 40)
        }

    def _dataframe_to_float_matrix(self, df):
        id_map = {}

        self.user_features = {}  # map from user to indexes
        self.item_features = {}  # map from item to indexes

        features = self.user_column_names + self.item_column_names
        mc = FloatMatrix(len(df), len(features))
        for row_id, row_data in enumerate(df.iterrows()):
            _, tuple_var = row_data
            user_idx = []
            item_idx = []
            for i, c in enumerate(features):
                cmap = id_map.get(c, {})
                value = cmap.get(tuple_var[c], len(cmap)+1)
                cmap[tuple_var[c]] = value
                id_map[c] = cmap
                mc.put(row_id, i, value)
                if c in self.user_column_names:
                    user_idx.append(value)
                if c in self.item_column_names:
                    item_idx.append(value)

            self.user_features[tuple_var["user"]] = user_idx
            self.item_features[tuple_var["item"]] = item_idx
        return mc, id_map

    def _fit(self, data):
        """
        Return the model
        """
        #data, tmap = self._map(data)
        data, tmap = self._dataframe_to_float_matrix(data)
        self._dmap = tmap

        dims = [len(self._dmap[c])
                for c in self.user_column_names + self.item_column_names]

        tensor = JavaTensorCoFi(self._dim, self._nIter, self._lamb, self._alph,
                                dims)
        tensor.train(data)

        final_model = tensor.getModel()
        self.factors = {}
        for i, c in enumerate(self.user_column_names+self.item_column_names):
            self.factors[c] = self._float_matrix2numpy(final_model.get(i)).transpose()
        return tensor

    def fit(self, data):
        """
        Prepare the model
        """
        self._fit(data)

    def _float_matrix2numpy(self, java_float_matrix):
        """
        Java Float Matrix is a 1-D array writen column after column.
        Numpy reads row after row, therefore, we need a conversion.
        """
        columns_input = java_float_matrix.toArray()
        split = lambda lst, sz: [np.fromiter(lst[i:i+sz],dtype=np.float)
                                 for i in range(0, len(lst), sz)]
        cols = split(columns_input, java_float_matrix.rows)
        matrix = np.ma.column_stack(cols)
        return matrix

    def get_score(self, user, item):
        names = self.user_column_names + self.item_column_names
        indexes = self.user_features[user] + self.item_features[item]
        for i, name in enumerate(names):
            try:
                ret = np.multiply(ret, self.factors[name][indexes[i]-1])
            except NameError:
                ret = self.factors[name][indexes[i]-1]
        return sum(ret)

    def set_params(self, dim=20, nIter=5, lamb=0.05, alph=40):
        """
        Set the parameters for the TensorCoFi
        """
        self._dim = dim
        self._nIter = nIter
        self._lamb = lamb
        self._alph = alph

    def get_name(self):
        return "TensorCoFi (dim={},iter={},lambda={},alpha={})".format(self._dim, self._nIter, self._lamb, self._alph)


class TensorCoFiByFile(TensorCoFi):

    _dmap = {}

    def _map(self, df):
        id_map = {}

        self.user_features = {}  #map from user to indexes
        self.item_features = {}  #map from item to indexes

        features = self.user_column_names + self.item_column_names
        mc = []
        for row_id, row_data in enumerate(df.iterrows()):
            _, tuple_var = row_data
            user_idx = []
            item_idx = []
            t = []
            for i, c in enumerate(features):
                cmap = id_map.get(c, {})
                value = cmap.get(tuple_var[c], len(cmap)+1)
                cmap[tuple_var[c]] = value
                id_map[c] = cmap
                t.append(value)
                if c in self.user_column_names:
                    user_idx.append(value)
                if c in self.item_column_names:
                    item_idx.append(value)
            mc.append(t)
            self.user_features[tuple_var["user"]] = user_idx
            self.item_features[tuple_var["item"]] = item_idx
        return pd.DataFrame({k: v for k, v in zip(features,zip(*mc))}), id_map

    def fit(self, data):
        data, tmap = self._map(data)
        self._dmap = tmap
        directory = "log/" + datetime.datetime.now().isoformat("_")
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory+"/train.csv", "w") as datafile:
            data.to_csv(datafile, header=False, index=False, cols=["user", "item"])
            name = os.path.dirname(datafile.name)+"/"
        java_jar = resource_filename(testfm.__name__, "lib/algorithm-1.0-SNAPSHOT-jar-with-dependencies.jar")
        sub = subprocess.Popen(["java", "-cp", java_jar, "es.tid.frappe.python.TensorCoPy", name, str(self._dim),
                                str(self._nIter), str(self._lamb), str(self._alph), str(len(tmap[USER])),
                                str(len(tmap[ITEM]))], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = sub.communicate()
        if err:
            #os.remove(name)
            print out
            raise Exception(err)
        users, items = out.split(" ")
        self.factors = {
            "user": np.ma.column_stack(np.genfromtxt(open(users, "r"), delimiter=",")),
            "item": np.ma.column_stack(np.genfromtxt(open(items, "r"), delimiter=","))
        }


class PyTensorCoFi(IModel):
    """
    Python implementation of tensorCoFi algorithm based on the java version from Alexandros Karatzoglou
    """

    def __init__(self, nfactors, niterations, clambda, calpha):
        """
        Constructor

        :param nfactors: Number of factors to the matrices
        :param niterations: Number of iteration in the matrices construction
        :param clambda: I came back when I find it out
        :param calpha: Constant important in weight calculation
        """
        self.number_of_factors = nfactors
        self.constant_lambda = clambda
        self.number_of_iterations = niterations
        self.constant_alpha = calpha
        self.user_to_id = {}
        self.item_to_id = {}
        self.dimensions = None
        self.factors = []
        self.counts = []
        self.base = self.tmp_calc = None
        self.tmp = np.ones((self.number_of_factors, 1))
        self.invertible = np.zeros((self.number_of_factors, self.number_of_factors))
        self.matrix_vector_product = np.zeros((self.number_of_factors, 1))

    def base_for_2_dimensions(self, current_dimension):
        """
        Calculation of base matrix for 2 dimension tensor
        :param current_dimension: dimension to calculate
        :return: A base matrix
        """
        base = self.factors[1 - current_dimension]
        return np.dot(base, base.transpose())

    def standard_base(self, current_dimension):
        """
        Standard base matrix calculation
        :param current_dimension: dimension to calculate
        :return: A base matrix
        """
        base = np.ones((self.number_of_factors, self.number_of_factors))
        for matrixIndex in range(len(self.dimensions)):
            if matrixIndex != current_dimension:
                base = np.multiply(base, np.dot(self.factors[matrixIndex], self.factors[matrixIndex].transpose()))
        return base

    def tmp_or_2_dimensions(self, current_dimension, training_data, row):
        """
        Calculate the tmp matrix for 2 dimension tensor
        :param current_dimension: dimension to calculate
        :param training_data: Matrix with the training data
        :param row: Working row
        :return: A tmp matrix
        """
        column = training_data[row, 1-current_dimension]-1
        return self.tmp * self.factors[1-current_dimension][:, column].reshape(self.number_of_factors, 1)

    def standard_tmp(self, current_dimension, training_data, row):
        """
        Standard tmp calculator
        :param current_dimension: dimension to calculate
        :param training_data: Matrix with the training data
        :param row: Working row
        :return: A tmp matrix
        """
        self.tmp = np.add(np.multiply(self.tmp, 0.), 1.0)
        for column in range(len(self.dimensions)):
            if column != current_dimension:
                self.tmp = \
                    self.tmp * self.factors[column][:, training_data[row, column]-1].reshape(self.number_of_factors, 1)
        return self.tmp

    def train(self, training_data):
        regularizer = np.multiply(np.eye(self.number_of_factors), self.constant_lambda)
        one = np.eye(self.number_of_factors)
        tensor = {}
        for index, dimension in enumerate(self.dimensions):
            tensor[index] = {}
            for row in range(training_data.shape[0]):
                try:
                    tensor[index][training_data[row, index]].append(row)
                except KeyError:
                    tensor[index][training_data[row, index]] = [row]

        for iteration in range(self.number_of_iterations):
            for current_dimension in range(len(self.dimensions)):
                base = self.base(current_dimension)

                for entry in range(1, self.dimensions[current_dimension]+1):
                    row_list = tensor[current_dimension].get(entry, [])
                    for row in row_list:
                        tmp = self.tmp_calc(current_dimension, training_data, row)
                        score = training_data[row, training_data.shape[1] - 1]
                        weight = 1. + self.constant_alpha * math.log(1. + math.fabs(score))
                        try:
                            invertible += (weight - 1.) * (tmp * tmp.transpose())
                        except NameError:
                            invertible = self.invertible + (weight - 1.) * (tmp * tmp.transpose())
                        try:
                            matrix_vector_product = \
                                np.add(matrix_vector_product, np.multiply(tmp, math.copysign(1, score) * weight))
                        except NameError:
                            matrix_vector_product = \
                                np.add(self.matrix_vector_product, np.multiply(tmp, math.copysign(1, score) * weight))

                    invertible = np.add(invertible, base)
                    regularizer = regularizer / self.dimensions[current_dimension]
                    invertible = np.add(invertible, regularizer)
                    invertible = np.linalg.solve(invertible, one)
                    self.factors[current_dimension][:, entry-1] = \
                        np.dot(invertible, matrix_vector_product).reshape(self.number_of_factors)
                    del matrix_vector_product
                    del invertible

    def fit(self, data):
        self.user_to_id = {}
        self.item_to_id = {}
        for uid, user in enumerate(data["user"].unique(), start=1):
            self.user_to_id[user] = uid
        for iid, item in enumerate(data["item"].unique(), start=1):
            self.item_to_id[item] = iid

        np_data = \
            np.matrix([(self.user_to_id[row["user"]], self.item_to_id[row["item"]]) for _, row in data.iterrows()])
        self.dimensions = [len(self.user_to_id), len(self.item_to_id)]
        self.base = self.base_for_2_dimensions if len(self.dimensions) == 2 else self.standard_base
        self.tmp_calc = self.tmp_or_2_dimensions if len(self.dimensions) == 2 else self.standard_tmp

        self.factors = [np.random.rand(self.number_of_factors, i) for i in self.dimensions]
        self.counts = [np.zeros((i, 1)) for i in self.dimensions]
        self.train(np_data)
        self.base = self.tmp_calc = None

    def get_model(self):
        """
        TODO
        """
        return self.factors

    def get_score(self, user, item):
        user_vec = self.factors[0][:, self.user_to_id[user]-1].transpose()
        item_vec = self.factors[1][:, self.item_to_id[item]-1]
        return np.dot(user_vec, item_vec)

    def get_name(self):
        return "Python TensorCoFi"

    def online_user_factors(self, Y, user_item_ids, p_param = 10, lambda_param = 0.01):
        """
        :param Y: application matrix Y.shape = (#apps, #factors)
        :param user_item_ids: the rows that correspond to installed applications in Y matrix
        :param p_param: p parameter
        :param lambda_param: regularizer
        """
        y = Y[user_item_ids]
        base1 = Y.transpose().dot(Y)
        base2 = y.transpose().dot(np.diag([p_param - 1] * y.shape[0])).dot(y)
        base = base1 + base2 + np.diag([lambda_param] * base1.shape[0])
        u_factors = np.linalg.inv(base).dot(y.transpose()).dot(np.diag([p_param] * y.shape[0])).dot(np.ones(y.shape[0]).transpose())
        return u_factors


if __name__ == "__main__":

    import doctest
    doctest.testmod()

    t = TensorCoFiByFile()
    t.fit(pd.DataFrame({
        "user": [1, 1, 3, 4], "item": [1, 2, 3, 4], "rating": [5,3,2,1],
        "date": [11,12,13,14]}))
    t.get_score(1, 4)
