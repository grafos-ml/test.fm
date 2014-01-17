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


class TensorCoFiModel(ModelInterface):
    '''
    TensorCoFi Model generator from the system database.
    '''

    def __init__(self,host,dbname,user,psw,dim=20, nIter=5, lamb=0.05, alph=40,
        port=3306,):
        '''
        Python model creator fro tensor implementation in java

        **Args**

            *String* host:
                Database url or ip.

            *String* dbname:
                The name of the database using to fetch data

            *String* user:
                Username for database authentication.

            *String* psw:
                User password for authentication purposes.

            *int* dim:
                Dimension of some kind. Default = 20.

            *int* nIter:
                Nmber of iteration. Default = 5.

            *float* lamb:
                Lambda value for the algorithm. Default = 0,05.

            *int* alph:
                Alpha number for the algorithm. Default = 40.

            *int* port:
                Port number for the database. Default = 3306.

        **Return**

            *numpy.Array*:

        '''
        self.reader = MySQLDataReader(host,port,dbname,user,psw)
        self.model = TensorCoFi(dim,nIter,lamb,alph,self.reader.getDims())
        self.users, self.apps = self.getModel()

    def getData(self):
        '''
        Get the data from the database
        '''
        return self.reader.getData()

    def getModel(self):
        '''
        Return the model
        '''
        self.model.train(self.getData())
        final_model = self.model.getModel()
        t0 = numpy.fromiter(final_model.get(0).toArray(),dtype=numpy.float)
        t0.shape = final_model.get(0).rows, final_model.get(0).columns
        t1 = numpy.fromiter(final_model.get(1).toArray(),dtype=numpy.float)
        t1.shape = final_model.get(1).rows, final_model.get(1).columns
        return t0, t1

    def getScore(self,user,item):
        '''
        Get a app score for a given user
        '''
        return (self.users.transpose()[user.pk-1] * self.apps)[item.pk-1]

if __name__ == '__main__':
    HOST = '192.168.188.128'
    USER = 'raqksixq_frappe'
    PSW = 'sp21o61h4'
    DB = 'raqksixq_ffosv1'
    t = TensorCoFiModel(HOST,DB,USER,PSW)
    print t


