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
from interface import ModelInterface

TensorCoFi = autoclass('es.tid.frappe.recsys.TensorCoFi')
MySQLDataReader = autoclass('es.tid.frappe.mysql.MySQLDataReader')
FloatMatrix = autoclass('org.jblas.FloatMatrix')


class TensorCoFiModel(ModelInterface):

    def __init__(self,host,user,psw,dim=20, nIter=5, lamb=0.05, alph=40,
        port=3306,dbname='raqksixq_ffosv1'):
        '''
        Python model creator fro tensor implementation in java

        **Args**

        **Return**

            *numpy.Array*:

        '''
        self.reader = MySQLDataReader(host,port,dbname,user,psw)
        self.model = TensorCoFi(dim,nIter,lamb,alph,self.reader.getDims())
        self.users, self.apps = self.getModel()

    def getData(self):
        return self.reader.getData()

    def getModel(self):
        self.model.train(self.getData())
        final_model = self.model.getModel()
        t0 = numpy.fromiter(final_model.get(0).toArray(),dtype=numpy.float)
        t0.shape = final_model.get(0).rows, final_model.get(0).columns
        t1 = numpy.fromiter(final_model.get(1).toArray(),dtype=numpy.float)
        t1.shape = final_model.get(1).rows, final_model.get(1).columns
        return t0, t1

    def getScore(self,user,item):
        return (self.users.transpose()[user.pk-1] * self.apps)[item.pk-1]

if __name__ == '__main__':
    HOST = '192.168.188.128'
    USER = 'raqksixq_frappe'
    PSW = 'sp21o61h4'
    t = TensorCoFiModel(HOST,USER,PSW)
    print t


