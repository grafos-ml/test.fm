# -*- coding: utf-8 -*-
"""
Created at March 12, 2014

Connect to the okapi to create some model.

.. moduleauthor:: joaonrb <joaonrb@gmail.com>

"""
__author__ = "joaonrb"


import hashlib


class ModelConnector(object):
    """
    Connect a model to okapi
    """

    @staticmethod
    def hash_data(data):
        """
        Retrieves a hash value out of the data

        >>> import pandas as pd
        >>> data = pd.DataFrame({i: [j**i for j in range(20, 30)] for i in range(2, 4)})
        >>> ModelConnector.hash_data(data)
        '37693cfc748049e45d87b8c7d8b9aacd'

        :param data: data to hash
        """
        md5 = hashlib.md5()
        for row in data:
            md5.update(str(row))
        return md5.hexdigest()

    @staticmethod
    def hash_file(jar):
        """
        Return the hash of a jar file

        >>> from pkg_resources import resource_filename
        >>> import testfm
        >>> jar = open(resource_filename(testfm.__name__, "lib/algorithm-1.0-SNAPSHOT-jar-with-dependencies.jar"))
        >>> ModelConnector.hash_file(jar)
        'c728db9090bd2d3201850c7625573ea6'

        :param jar: Jar file to get the hash
        :return:
        """
        md5 = hashlib.md5()
        for line in jar:
            md5.update(str(line))
        return md5.hexdigest()

    """
    TODO: Implementation of Fabric to SSH connection
    """

