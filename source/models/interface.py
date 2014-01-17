# -*- coding: utf-8 -*-
'''
Created on 16 January 2014

Interfaces for models

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
'''
__author__ = 'joaonrb'


class ModelInterface(object):
    '''
    
    '''

    def getScore(self,user,item):
        raise NotImplementedError

