# -*- coding: utf-8 -*-
"""
Created on 17 January 2014

Slitter interface

.. moduleauthor:: joaonrb <joanrb@gmail.com>
"""
__author__ = {
    'name':'joaonrb',
    'e-mail': 'joaonrb@gmail.com'
}
__version__ = 1,0,0
__since__ = 17,1,2014

class SplitterInterface(object):
    '''
    Splitter interface. To be implemented for different split methods
    '''

    def __call__(self, data, fraction, **kwargs):
        """
        Divides the data into train data and tests data
        """
        return self.split(self.sort(data), fraction, **kwargs)


