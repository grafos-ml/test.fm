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
        d = dataframe.to_dict(outtype='list')
        md = {key: {} for key in d}
        for i in xrange(len())

    def fit(self,dataframe):
        '''
        Return the model
        '''
        data, map = self._mapData(dataframe)



        self.model.train()
        final_model = self.model.getModel()
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



