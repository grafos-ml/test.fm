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

from pkg_resources import resource_filename
import testfm
import os
os.environ['CLASSPATH'] = resource_filename(testfm.__name__,'lib/'
    'algorithm-1.0-SNAPSHOT-jar-with-dependencies.jar')

from jnius import autoclass
import numpy
from testfm.models.interface import ModelInterface
from testfm.config import USER, ITEM

JavaTensorCoFi = autoclass('es.tid.frappe.recsys.TensorCoFi')
MySQLDataReader = autoclass('es.tid.frappe.mysql.MySQLDataReader')
FloatMatrix = autoclass('org.jblas.FloatMatrix')
Arrays = autoclass('java.util.Arrays')


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
        rows = len(d[USER])
        ndf = FloatMatrix.zeros(rows,2)
        uid_counter, iid_counter = 1, 1
        for i in xrange(rows):
            try:
                nu = md[USER][d[USER][i]]
            except KeyError:
                md[USER][d[USER][i]] = uid_counter
                rmd[USER][uid_counter] = d[USER][i]
                nu = uid_counter
                uid_counter += 1
            try:
                ni = md[ITEM][d[ITEM][i]]
            except KeyError:
                md[ITEM][d[ITEM][i]] = iid_counter
                rmd[ITEM][iid_counter] = d[ITEM][i]
                ni = iid_counter
                iid_counter += 1
            ndf.put(i,float(nu))
            ndf.put(i+rows,float(ni))
        self._map = md
        return ndf, rmd

    def fit(self,dataframe):
        '''
        Return the model
        '''
        data, tmap = self._map(dataframe)
        tensor = JavaTensorCoFi(self._dim,self._nIter,self._lamb,self._alph,
            [len(tmap[USER]),len(tmap[ITEM])])
        tensor.train(data)

        final_model = tensor.getModel()
        t0 = numpy.fromiter(final_model.get(0).toArray(),dtype=numpy.float)
        t0.shape = final_model.get(0).rows, final_model.get(0).columns
        t1 = numpy.fromiter(final_model.get(1).toArray(),dtype=numpy.float)
        t1.shape = final_model.get(1).rows, final_model.get(1).columns
        self._users, self._apps = numpy.matrix(t0), numpy.matrix(t1)

    def getScore(self,user,item):
        '''
        Get a app score for a given user
        '''
        # nem todas as apps the teste estao no tensor
        a = (self._users.transpose()[self._map[USER][user]-1] * self._apps)
        try:
            return a[0,self._map[ITEM][item]-1]
        except KeyError:
            return 0.0



