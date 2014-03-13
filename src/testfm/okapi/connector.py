# -*- coding: utf-8 -*-
"""
Created at March 12, 2014

Connect to the okapi to create some model.

.. moduleauthor:: joaonrb <joaonrb@gmail.com>

"""
__author__ = "joaonrb"


import hashlib
from fabric.api import run


class ModelConnector(object):
    """
    Connect a model to okapi
    """

    OKAPI_REPOSITORY = "okapi"
    OKAPI_JAR_REPOSITORY = OKAPI_REPOSITORY + "/jar"
    OKAPI_DATA_REPOSITORY = OKAPI_REPOSITORY + "/data"
    OKAPI_RESULTS_REPOSITORY = OKAPI_REPOSITORY + "/results"

    @staticmethod
    def get_jar_location(jar):
        """
        Returns the location in the remote system of this jar file

        :param jar: The jar file
        :return: The location of the jar
        """
        return "%(dir)s/%(jar_place)s/%(jar_name)s" % {
            "dir": ModelConnector.OKAPI_JAR_REPOSITORY,
            "jar_place": ModelConnector.hash_file(jar),
            "jar_name": jar.name
        }

    @staticmethod
    def get_data_location(data):
        """
        Returns the location in the remote system of this pandas data object

        :param data: The pandas data object
        :return: The location of the data
        """
        return "%(dir)s/%(data_set)s" % {
            "dir": ModelConnector.OKAPI_DATA_REPOSITORY,
            "data_set": ModelConnector.hash_data(data)
        }

    def get_result_location(self, data):
        """
        Returns the location in the remote system of the result for this pandas data object

        :param data: The pandas data object
        :return: The location of the data
        """
        return "%(dir)s/%(model)s/%(data_set)s" % {
            "dir": ModelConnector.OKAPI_RESULTS_REPOSITORY,
            "model": self.name,
            "data_set": ModelConnector.hash_data(data)
        }

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

    @property
    def name(self):
        raise NotImplemented()

    def call_okapi(self, data):
        """
        Call the okapi framework to make a model

        :param data: Data to produce the model
        """
        if self.result_exist_for(data):
            return self.result
        else:
            self.prepare_environment()
            return self.execute_okapi()

    def result_exist_for(self, data):
        """
        Check if the result of this algorithm with this data is already computed.

        :param data: Data to compute
        :return: True if the data is already computed with this algorithm
        """
        result_file = self.get_result_location(data)
        command = "[ -f %s ] && echo 1 || echo 0" % result_file
        do_result_exist = run(command, quiet=True)
        if do_result_exist:
            self.read_result(result_file)
            return True
        return False

