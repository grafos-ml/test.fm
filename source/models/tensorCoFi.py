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

TensorCoFi = autoclass('es.tid.frappe.recsys.TensorCoFi')
MySQLDataReader = autoclass('es.tid.frappe.mysql.MySQLDataReader')
FloatMatrix = autoclass('org.jblas.FloatMatrix')


class PythonTensorCoFi(object):

    def __init__(self,dim=20, nIter=5, lamb=0.05, alph=40,
        port=3306,dbname='raqksixq_ffosv1'):
        '''
        Python model creator fro tensor implementation in java

        **Args**

        **Return**

            *numpy.Array*:

        '''
        self.reader = MySQLDataReader(HOST,port,dbname,USER,PDW)
        self.model = TensorCoFi(dim,nIter,lamb,alph,self.reader.getDims())

    def getData(self):
        return self.reader.getData()

    def getModel(self):
        self.model.train(self.getData())
        final_model = self.model.getModel()
        print final_model

if __name__ == '__main__':
    HOST = '192.168.188.128'
    USER = 'raqksixq_frappe'
    PDW = 'sp21o61h4'
    t = PythonTensorCoFi().getModel()
    print t


