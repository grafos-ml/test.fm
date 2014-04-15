__author__ = 'joaonrb'

import numpy as np
import scipy as sp
import math


class PyTensorCoFi(object):
    """
    private int[] dimensions; //count entries in each dimension

    private final int d;
    private final int iter;
    private final float p;
    private final float lambda;
    private ArrayList<FloatMatrix> Factors; //factors for each dimension
    private ArrayList<FloatMatrix> Counts;
    """
    def __init__(self, d, ite, Lambda, p):
        self.d = d
        self.dimensions = None
        self.Lambda = Lambda
        self.ite = ite
        self.p = p

    def train(self, dataArray):

        temp = np.ones((self.d, 1))
        regularizer = np.multiply(np.eye(self.d), self.Lambda)
        matrixVectorProd = np.zeros((self.d, 1))
        one = np.eye(self.d)
        invertible = np.zeros((self.d, self.d))

        #List<Map<Integer, List<Integer>>> tensor = new ArrayList<Map<Integer, List<Integer>>>();
        tensor = []

        dataRowList = []

        t = {}
        for i, dimension in enumerate(self.dimensions):
            tensor.append({})

            for j in xrange(dimension):
                tensor[i][j+1] = []

            for dataRow in range(dataArray.shape[0]):
                index = dataArray[dataRow, i]
                t = tensor[i]
                t[index].append(dataRow)

        for ite in range(self.ite):
            for currentDimension in range(len(self.dimensions)):
                if len(self.dimensions) == 2:
                    base = self.factors[1 - currentDimension]
                    base = np.dot(base, base.transpose())
                else:
                    base = np.ones((self.d, self.d))
                    for matrixIndex in range(len(self.dimensions)):
                        if matrixIndex != currentDimension:
                            base = np.multiply(base, np.dot(self.factors[matrixIndex],
                                                            self.factors[matrixIndex].transpose()))

                #if ite == 0:
                    # Heavy
                #    for dataEntry in range(self.dimensions[currentDimension]):
                #        counts = 0

                #        for dataRow in range(dataArray.shape[0]):
                #            if dataArray[dataRow, currentDimension] == dataEntry:
                #             counts += 1
                #            if counts == 0:
                #                counts = 1
                #            self.counts[currentDimension][dataEntry, 0] = counts

                for dataEntry in range(1, self.dimensions[currentDimension]+1):
                    dataRowList = tensor[currentDimension][dataEntry]
                    for dataRow in dataRowList:
                        temp = np.add(np.multiply(temp, 0.), 1.0)
                        for dataCol in range(len(self.dimensions)):
                            if dataCol != currentDimension:
                                temp = temp * self.factors[dataCol][:, dataArray[dataRow, dataCol]-1].reshape(self.d, 1)
                        score = dataArray[dataRow, dataArray.shape[1] - 1]
                        weight = 1. + self.p * math.log(1. + math.fabs(score))
                        #invertible = invertible.rankOneUpdate((weight - 1.0), temp)
                        invertible += (weight - 1.) * (temp * temp.transpose())

                        #matrixVectorProd = matrixVectorProd.addColumnVector(temp.mul((float) Math.signum(score)*weight)
                        matrixVectorProd = np.add(matrixVectorProd, np.multiply(temp, math.copysign(1, score) * weight))
                    invertible = np.add(invertible, base)
                    regularizer = regularizer / self.dimensions[currentDimension]

                    invertible = np.add(invertible, regularizer)
                    invertible = np.linalg.solve(invertible, one)

                    self.factors[currentDimension][:, dataEntry-1] = np.dot(invertible, matrixVectorProd).reshape(self.d)
                    invertible = np.multiply(invertible,  0.)
                    matrixVectorProd = np.multiply(matrixVectorProd, 0.)

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
        self.factors = [np.random.rand(self.d, i) for i in self.dimensions]
        self.counts = [np.zeros((i, 1)) for i in self.dimensions]
        #for dim in self.dimensions:
        #    self.factors.append(np.random.rand(self.d, dim))
        #    self.counts.append(np.zeros((dim, 1)))
        self.train(np_data)

    def get_model(self):
        """
        TODO
        """
        return self.factors

    def getScore(self, user, item):
        user_vec = self.factors[0][:, self.user_to_id[user]-1].transpose()
        item_vec = self.factors[1][:, self.item_to_id[item]-1]
        return np.dot(user_vec, item_vec)

    def getName(self):
        return "Python Implementations of TensorCoFi"