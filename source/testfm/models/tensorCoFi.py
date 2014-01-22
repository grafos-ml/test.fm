# -*- coding: utf-8 -*-
'''
Created on 16 January 2014

Connector for the tensor CoFi Java implementation

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
'''
__author__ = {
    'name':'joaonrb',
    'e-mail': 'joaonrb@gmail.com'
}
__version__ = 1,0,0
__since__ = 16,1,2014

import os
os.environ['CLASSPATH'] = '../lib/*'

from jnius import autoclass
import numpy
from testfm.models.interface import ModelInterface
from testfm.config import USER, ITEM

TensorCoFi = autoclass('es.tid.frappe.recsys.TensorCoFi')
MySQLDataReader = autoclass('es.tid.frappe.mysql.MySQLDataReader')
FloatMatrix = autoclass('org.jblas.FloatMatrix')


class TensorCoFi(ModelInterface):

    def __init__(self,dim=20, nIter=5, lamb=0.05, alph=40):
        '''
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

        **Return**

            *numpy.Array*:

        '''
        self._dim = dim
        self._nIter = nIter
        self._lamb = lamb
        self._alph = alph
        #self.model = TensorCoFi(dim,nIter,lamb,alph,self.reader.getDims())
        self._users, self._apps = [], []

    def _map(self,dataframe):
        d, md, rmd = dataframe.to_dict(outtype='list'), {USER: {}, ITEM: {}},\
            {USER: {}, ITEM: {}}
        ndf = []
        uid_counter, iid_counter = 0, 0
        for i in xrange(len(dataframe.index)):
            try:
                nu = md[USER][d[USER][i]]
            except KeyError:
                nu = md[USER][d[USER][i]] = uid_counter
                rmd[USER][uid_counter] = d[USER][i]
                uid_counter += 1
            try:
                ni = md[ITEM][d[ITEM][i]]
            except KeyError:
                ni = md[ITEM][d[ITEM][i]] = iid_counter
                rmd[ITEM][iid_counter] = d[ITEM][i]
                iid_counter += 1
            ndf.append((nu,ni))
            # There's no need to have mor fields than this
            #for key in d not in [USER,ITEM]:
            #    rmd[key].append(d[key][i])
        return FloatMatrix(ndf), rmd

    def fit(self,dataframe):
        '''
        Return the model
        '''
        data, map = self._mapData(dataframe)

        tensor = TensorCoFi(self.dim,self.nIter,self.lamb,self.alph,
            [len(map[USER]),len(map[ITEM])])
        tensor.train(data)

        final_model = tensor.getModel()
        t0 = numpy.fromiter(final_model.get(0).toArray(),dtype=numpy.float)
        t0.shape = final_model.get(0).rows, final_model.get(0).columns
        t1 = numpy.fromiter(final_model.get(1).toArray(),dtype=numpy.float)
        t1.shape = final_model.get(1).rows, final_model.get(1).columns
        self._users, self._apps = t0, t1

    def getScore(self,user,item):
        '''
        Get a app score for a given user
        '''
        return (self._users.transpose()[user.pk-1] * self._apps)[item.pk-1]



