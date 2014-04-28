# -*- coding: utf-8 -*-
"""
Created on 16 January 2014

Connector for the tensor CoFi Java implementation

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
"""
__author__ = {
    "name": "joaonrb",
    "e-mail": "joaonrb@gmail.com"
}
__version__ = 1, 2
__since__ = 16, 1, 2014
__change__ = 16, 4, 2014

from pkg_resources import resource_filename
import testfm
import os
import numpy as np
import shutil
import datetime
import subprocess
from testfm.models import IModel
import math
from testfm.models.cutil.interface import IFactorModel


class TensorCoFi(IFactorModel):

    number_of_factors = 20
    number_of_iterations = 5
    constant_lambda = .05
    constant_alpha = 40

    def __init__(self, n_factors=None, n_iterations=None, c_lambda=None, c_alpha=None, other_context=None):
        """
        Constructor

        :param n_factors: Number of factors to the matrices
        :param n_iterations: Number of iteration in the matrices construction
        :param c_lambda: I came back when I find it out
        :param c_alpha: Constant important in weight calculation
        """
        self.set_params(n_factors, n_iterations, c_lambda, c_alpha)
        self.factors = []
        self.context_columns = other_context

    @classmethod
    def param_details(cls):
        """
        Return parameter details for n_factors, n_iterations, c_lambda and c_alpha
        """
        return {
            "n_factors": (10, 20, 2, 20),
            "n_iterations": (1, 10, 2, 5),
            "c_lambda": (.1, 1., .1, .05),
            "c_alpha": (30, 50, 5, 40)
        }

    def get_context_columns(self):
        """
        Get a list of names of all the context column names for this model
        :return:
        """
        return self.context_columns or []

    def train(self, data):
        """
        Train the model
        """
        directory = "log/" + datetime.datetime.now().isoformat("_")
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory+"/train.csv", "w") as datafile:
            np.savetxt(datafile, data, delimiter=", ")
            name = os.path.dirname(datafile.name)+"/"
        java_jar = resource_filename(testfm.__name__, "lib/algorithm-1.0-SNAPSHOT-jar-with-dependencies.jar")
        sub = subprocess.Popen(["java", "-cp", java_jar, "es.tid.frappe.python.TensorCoPy", name,
                                str(self.number_of_factors), str(self.number_of_iterations), str(self.constant_lambda),
                                str(self.constant_alpha), str(self.users_size()),
                                str(self.items_size())], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = sub.communicate()
        if err:
            #os.remove(name)
            print out
            raise Exception(err)
        users, items = out.split(" ")
        self.factors = [
            np.genfromtxt(open(users, "r"), delimiter=",").transpose(),
            np.genfromtxt(open(items, "r"), delimiter=",").transpose()
        ]
        shutil.rmtree("log")

    def get_model(self):
        return self.factors

    #def get_score(self, user, item):
    #    user_vec = self.factors[0][:, self.data_map[self.get_user_column()][user]-1].transpose()
    #    item_vec = self.factors[1][:, self.data_map[self.get_item_column()][item]-1]
    #    return np.dot(user_vec, item_vec)

    @staticmethod
    def online_user_factors(matrix_y, user_item_ids, p_param=10, lambda_param=0.01):
        """
        :param matrix_y: application matrix Y.shape = (#apps, #factors)
        :param user_item_ids: the rows that correspond to installed applications in Y matrix
        :param p_param: p parameter
        :param lambda_param: regularizer
        """
        y = matrix_y[user_item_ids]
        base1 = matrix_y.transpose().dot(matrix_y)
        base2 = y.transpose().dot(np.diag([p_param - 1] * y.shape[0])).dot(y)
        base = base1 + base2 + np.diag([lambda_param] * base1.shape[0])
        u_factors = np.linalg.inv(base).dot(y.transpose()).dot(np.diag([p_param] * y.shape[0]))
        u_factors = u_factors.dot(np.ones(y.shape[0]).transpose())
        return u_factors

    def set_params(self, n_factors, n_iterations, c_lambda, c_alpha):
        """
        Set the parameters for the TensorCoFi
        """
        self.number_of_factors = n_factors or self.number_of_factors
        self.number_of_iterations = n_iterations or self.number_of_iterations
        self.constant_lambda = c_lambda or self.constant_lambda
        self.constant_alpha = c_alpha or self.constant_alpha

    def get_name(self):
        return "TensorCoFi(n_factors=%s, n_iterations=%s, c_lambda=%s, c_alpha=%s)" % \
               (self.number_of_factors, self.number_of_iterations, self.constant_lambda, self.constant_alpha)


class PyTensorCoFi(TensorCoFi):
    """
    Python implementation of tensorCoFi algorithm based on the java version from Alexandros Karatzoglou
    """

    def __init__(self, n_factors=None, n_iterations=None, c_lambda=None, c_alpha=None):
        """
        Constructor

        :param n_factors: Number of factors to the matrices
        :param n_iterations: Number of iteration in the matrices construction
        :param c_lambda: I came back when I find it out
        :param c_alpha: Constant important in weight calculation
        """
        super(PyTensorCoFi, self).__init__(n_factors, n_iterations, c_lambda, c_alpha)
        self.user_to_id = {}
        self.item_to_id = {}
        self.dimensions = None
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
        self.dimensions = [self.users_size(), self.items_size()]
        self.base = self.base_for_2_dimensions if len(self.dimensions) == 2 else self.standard_base
        self.tmp_calc = self.tmp_or_2_dimensions if len(self.dimensions) == 2 else self.standard_tmp
        self.factors = [np.random.rand(self.number_of_factors, i) for i in self.dimensions]
        self.counts = [np.zeros((i, 1)) for i in self.dimensions]

        regularizer = np.multiply(np.eye(self.number_of_factors), self.constant_lambda)
        one = np.eye(self.number_of_factors)
        tensor = {}
        for index, dimension in enumerate(self.dimensions):
            tensor[index] = {}
            for row in xrange(training_data.shape[0]):
                try:
                    tensor[index][training_data[row, index]].append(row)
                except KeyError:
                    tensor[index][training_data[row, index]] = [row]

        for iteration in range(self.number_of_iterations):
            for current_dimension, dimension in enumerate(self.dimensions):
                base = self.base(current_dimension)

                for entry in range(1, dimension+1):
                    matrix_vector_product = self.matrix_vector_product
                    invertible = self.invertible
                    for row in tensor[current_dimension].get(entry, ()):
                        tmp = self.tmp_calc(current_dimension, training_data, row)
                        score = training_data[row, training_data.shape[1]-1]
                        weight = 1. + self.constant_alpha * math.log(1. + math.fabs(score))

                        invertible = np.add(invertible, (weight - 1.) * (tmp * tmp.transpose()))
                        matrix_vector_product = \
                            np.add(matrix_vector_product, np.multiply(tmp, math.copysign(1, score) * weight))

                    invertible += base
                    regularizer /= dimension
                    invertible += regularizer
                    invertible = np.linalg.solve(invertible, one)
                    self.factors[current_dimension][:, entry-1] = \
                        np.dot(invertible, matrix_vector_product).reshape(self.number_of_factors)

        self.base = self.tmp_calc = None
        self.factors[0] = self.factors[0].transpose()
        self.factors[1] = self.factors[1].transpose()

    def get_name(self):
        return "Python TensorCoFi(n_factors=%s, n_iterations=%s, c_lambda=%s, c_alpha=%s)" % \
               (self.number_of_factors, self.number_of_iterations, self.constant_lambda, self.constant_alpha)
