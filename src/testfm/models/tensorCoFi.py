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

import datetime
import subprocess

from jnius import autoclass
import numpy
import pandas as pd
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
        self._dmap = md
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

        try:
            a = (self._users.transpose()[self._dmap[USER][user]-1] * self._apps)
            return a[0,self._dmap[ITEM][item]-1]
        except KeyError:
            return 0.0

class TensorCoFiByFile(TensorCoFi):

    def _map(self,dataframe):
        d, md, rmd, result = dataframe.to_dict(outtype='list'), {USER: {},
            ITEM: {}},{USER: {}, ITEM: {}},{USER: [], ITEM: []}
        rows = len(d[USER])
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
            result[USER].append(nu), result[ITEM].append(ni)
        self._dmap = md
        return result, rmd

    def fit(self,dataframe):
        data, tmap = self._map(dataframe)
        direc = datetime.datetime.now().isoformat('_')
        if not os.path.exists(direc):
            os.makedirs(direc)
        with open(direc+'/train.csv','w') as datafile:
            pd.DataFrame(data).to_csv(datafile,header=False, index=False,
                cols=['user','item'])
            name = os.path.dirname(datafile.name)+'/'
        sub = subprocess.Popen(['java','-cp',
            resource_filename(testfm.__name__,'lib/'
            'algorithm-1.0-SNAPSHOT-jar-with-dependencies.jar'),
            'es.tid.frappe.python.TensorCoPy',name,str(self._dim),
            str(self._nIter),str(self._lamb),str(self._alph),
            str(len(tmap[USER])),str(len(tmap[ITEM]))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = sub.communicate()
        if err:
            #os.remove(name)
            raise Exception(err)
        users, items = out.split(' ')
        self._users, self._apps = numpy.matrix(numpy.genfromtxt(
            open(users,'r'),delimiter=',')),\
            numpy.matrix(numpy.genfromtxt(open(items,'r'),delimiter=','))


if __name__ == '__main__':
    t = TensorCoFiByFile()
    t.fit(pd.DataFrame({'user' : [1, 1, 3, 4], 'item' : [1, 2, 3, 4], \
            'rating' : [5,3,2,1], 'date': [11,12,13,14]}))
    t.getScore(1,4)



