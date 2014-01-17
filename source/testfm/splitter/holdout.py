# -*- coding: utf-8 -*-
'''
Created on 17 of January 2014

Holdout splitter method

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
'''
__author__ = {
    'name':'joaonrb',
    'e-mail': 'joaonrb@gmail.com'
}
__version__ = 1,0,0
__since__ = 17,1,2014

############################################
################# VARIABLES ################
############################################

USER = 'user'
APP = 'app'
DATE = 'date'

class Holdout(object):
    '''
    Holdout class for split
    '''
    __sort_methods__ = {
        'by_date': 'sort_by_date',
        'default': 'sort_default'
    }


    def __call__(self, dataset,fraction=0.9):
        '''
        This provides the class to be called and execute a split
        '''
        return self.sort(dataset,fraction)

    def sort_by_date(self,dataset,fraction):
        '''
        See if date can be sorted by data.
        '''
        cache = {}
        for e in d.values:
            cache[e[0]] = cache[e[0]] + [tuple(e[1:])] if e[0] in cache else \
                [tuple(e[1:])]
        train, testing = [], []
        for user, data in cache.values():
            i = len(data)
            data = [tuple(user,app,d) for app,d in sorted(data,cmp=lambda x,y:
                cmp(x[1],y[1]))]
            train += data[:int(i*fraction)]
            testing += data[int(i*fraction):]
        return train, testing

    def sort_default(self,dataset,fraction):
        '''
        See if date can be sorted by data.
        '''
        cache = {}
        for e in dataset.values:
            cache[e[0]] = cache[e[0]] + [tuple(e[1:])] if e[0] in cache else \
                [tuple(e[1:])]
        train, testing = [], []
        for user, data in cache.values():
            i = len(data)
            data = [tuple(user,app,d) for app,d in sorted(data)]
            train += data[:int(i*fraction)]
            testing += data[int(i*fraction):]
        return train, testing


    def sort(self,dataset,fraction,smethod=None):
        '''
        Try to sort dataset by time
        '''
        if smethod in Holdout.__sort_methods__:
            return getattr(self,Holdout.__sort_methods__[smethod])\
                (dataset,fraction)
        try:
            result = self.sort_by_date(dataset,fraction)
        except KeyError:
            result = self.sort_default(dataset,fraction)
        return result