# -*- coding: utf-8 -*-
"""
Created at March 12, 2014

Connect to the okapi to create some model.

.. moduleauthor:: joaonrb <joaonrb@gmail.com>

"""

__author__ = "joaonrb"

import logging
import hashlib
import numpy as np
import pandas as pd
from fabric.api import env, run
from pkg_resources import resource_filename
from testfm import okapi
from testfm.models.cutil.interface import IModel
import getpass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.WARNING)

logger.addHandler(logging.StreamHandler())

#OKAPI_REMOTE = "joaonrb@igraph-01"  # <-- Change to the proper remote account
#env.host_string = OKAPI_REMOTE

REMOTE_HOST = "igraph-01"

OKAPI_COMMAND = "hadoop jar %(okapi_jar)s org.apache.giraph.GiraphRunner -Dmapred.job.name=OkapiTrainModelTask " \
                "-Dmapred.reduce.tasks=0 -libjars %(okapi_jar)s -Dmapred.child.java.opts=-Xmx1g " \
                "-Dgiraph.zkManagerDirectory=%(manager_dir)s -Dgiraph.useSuperstepCounters=false %(model_class)s " \
                "-eif ml.grafos.okapi.cf.CfLongIdFloatTextInputFormat -eip %(input)s " \
                "-vof org.apache.giraph.io.formats.IdWithValueTextOutputFormat -op %(output)s -w 1 " \
                "-ca giraph.numComputeThreads=1 -ca minItemId=1 -ca maxItemId=%(max_item_id)s"


class OkapiConnectorError(Exception):
    """
    Generic exception for okapi connection models for test.fm
    """

    @classmethod
    def raise_this(cls, msg=""):
        """
        Raise an exception of this
        :param msg: The error message
        """
        raise cls(msg)


class OkapiNoResultError(OkapiConnectorError):
    """
    Exception thrown when trying to fetch a result that isn't calculated and fetched
    """


class OkapiJarNotInRepository(OkapiConnectorError):
    """
    Exception when trying to load a jar that is not in the repository
    """


class BaseOkapiModel(IModel):
    """
    New Okapi Connector abstract model
    """

    _data_hash = "data"
    _users = None
    _items = None
    _max_item_id = None
    _std_input = "okapi/input/%(hash)s"
    _std_output = "okapi/output/%(model)s/%(hash)s"
    _manager_dir = "okapi/_bsp"
    _model_java_class = None
    _okapi_local_repository = resource_filename(okapi.__name__, "lib/")
    _okapi_jar = "okapi-0.3.2-SNAPSHOT-jar-with-dependencies.jar"
    _source = ""

    def __init__(self, host=None, user=None, okapi_jar_dir=None, okapi_jar_base_name=None, public_key_path=None,
                 hadoop_source=None, okapi_output=None, okapi_input=None, model_java_class=None):
        """
        Constructor

        :param host: Host of the machine to call hadoop. Default is "localhost"
        :param user: The user that owns the session
        :param okapi_jar_dir: The path to the jar files in the machine(remote or local in case of localhost)
        :param okapi_jar_base_name: The name of the okapi jar file
        :param public_key_path: Public key to connect to remote session
        :param hadoop_source: The path to the hadoop source if a different source needed source
        :param okapi_output: Okapi output path
        :param okapi_input: Okapi input path
        """
        env.host_string = host or "localhost"
        env.user = user or getpass.getuser()
        if public_key_path:
            env.key_filename = public_key_path

        self._okapi_local_repository = okapi_jar_dir or self._okapi_local_repository
        self._okapi_jar = okapi_jar_base_name or self._okapi_jar
        self._model_java_class = model_java_class

        # Change source
        self._source = hadoop_source or self._source

        self._std_input = okapi_input or self._std_input
        self._std_output = okapi_output or self._std_output

    def fit(self, data=None, **kwargs):
        """
        Fits the model according to this data

        :param data: The training data to
        """
        if not self.result_exist(data):
            if not self.data_in_machine(data):
                self.put_data_in_machine(data)
            self.process_result(data, **kwargs)
        self._users, self._items = self.get_result()

    def result_exist(self, data):
        """
        Return True if result in the hadoop machine

        @:param data: The data to produce the result
        """
        return True

    def data_in_machine(self, data):
        """
        Check if data in machine

        :return: True if data in machine
        """
        return True

    def process_result(self, data, **kwargs):
        """
        Hask okapi the model

        :param kwargs: Extra parameters to pass on okapi. FORMAT TODO
        """
        run(self.source % "hadoop dfs -rmr %s" % self._manager_dir, quiet=True)
        hadoop_command = OKAPI_COMMAND % {
            "model_class": self._model_java_class,
            "okapi_jar": "%s%s" % (self._okapi_local_repository, self._okapi_jar),
            "max_item_id": self.get_item_len(data, **kwargs),
            "input": self.input,
            "output": self.output,
            "manager_dir": self._manager_dir
        }
        logger.debug("... Execute okapi: %s" % self.source % hadoop_command)
        run(self.source % hadoop_command, quiet=True)
        logger.debug("Done!")

    def get_item_len(self, data, **kwargs):
        """
        Get the max_item_id parameter
        :param kwargs:
        :return:
        """
        try:
            return self._max_item_id or kwargs["max_item_id"]
        except KeyError:
            return len(set([row["item"] for _, row in data.iterrows()]))

    def get_result(self):
        """
        Return the result from okapi
        """
        logger.debug("... Get result from hadoop")
        okapi_file = run(self.source % "for f in `hadoop dfs -ls %s/part-* | "
                         "awk '{print $8}' | "
                         "sort -V`; "
                         "do hadoop dfs -cat $f; done;" % self.output, quiet=True)
        result = self.output_okapi_to_pandas(okapi_file)
        logger.debug("Done!")
        return result

    @staticmethod
    def output_okapi_to_pandas(result_data):
        """
        Return 2 pandas DataFrame. The first for the user and the second for the items.

        :param result_data: String with output from okapi
        :return: A tuple with 2 DataFrame. (user, item)
        """
        data = {"0": {}, "1": {}}
        okapi_data = result_data.split("\n")
        for line in okapi_data:
            obj_id, obj_type, factors = line.replace("; ", ",").replace("\t", " ").split(" ")
            data[obj_type][obj_id] = eval(factors)
        result = pd.DataFrame(data["0"]), pd.DataFrame(data["1"])
        return result

    def get_score(self, user, item):
        """
        A score for a user and item that method predicts.
        :param user: id of the user
        :param item: id of the item
        :return:
        """
        return np.dot(self._users[str(user)].transpose(), self._items[str(item)])

    @property
    def input(self):
        """
        The input data path
        """
        return self._std_input % {"hash": self._data_hash}

    @property
    def output(self):
        """
        The input data path
        """
        return self._std_output % {"hash": self._data_hash, "model": self.name}

    @property
    def source(self):
        if self._source:
            return "source %s && %s" % (self._source, "%s")
        return "%s"


class DynamicOkapiModel(BaseOkapiModel):
    """
    A more dynamic approach to the connector.
    It moves the data into remote if is not there
    """

    def input_pandas_to_okapi(self, data):
        """
        Return a string with the data in the pandas DataFrame in okapi format

        :param data: Pandas data frame with the data. The data should have column for user and for item. Ot can have \
        also a column for rating. If it doesn't the rating is 1
        :type data: pandas.DataFrame
        :return: The data string in okapi format
        """
        data = data[:]
        # If data don't have ratings than the rating column is created with 1.
        if "rating" not in data:
            data["rating"] = [1. for _ in xrange(len(data))]


        # Make a generator with lines for okapi file format
        okapi_rows = []
        items = set([])
        for _, row in data.iterrows():
            okapi_rows.append("%(user)s %(item)s %(rating)s" % {  # okapi line
                "user": row["user"],
                "item": row["item"],
                "rating": row["rating"]
            })
            items.add(row["item"])
        self._max_item_id = len(items)
        return "\n".join(okapi_rows)

    @staticmethod
    def output_okapi_to_pandas(result_data):
        """
        Return 2 pandas DataFrame. The first for the user and the second for the items.

        :param result_data: String with output from okapi
        :return: A tuple with 2 DataFrame. (user, item)
        """
        data = {"0": {}, "1": {}}
        okapi_data = result_data.split("\n")
        for line in okapi_data:
            obj_id, obj_type, factors = line.replace("; ", ",").replace("\t", " ").split(" ")
            data[obj_type][obj_id] = eval(factors)
        result = pd.DataFrame(data["0"]), pd.DataFrame(data["1"])
        return result

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
        for _, row in data.iterrows():
            md5.update(str(row))
        result = md5.hexdigest()
        return result

    def result_exist(self, data):
        """
        Return True if result in the hadoop machine

        :param data: The data to produce the result
        """
        self._data_hash = self.hash_data(data)
        logger.debug("... Check if result exists")
        check_if_exists = self.source % "hadoop dfs -ls %s" % self.output
        if 0 == run(check_if_exists, warn_only=True, quiet=True).return_code:
            logger.debug("Result already exist")
            return True
        else:
            logger.debug("Result don't exist")
            return False

    def put_data_in_machine(self, data):
        """
        Put the data into hadoop
        :param data: Pandas DataFrame
        """
        logger.debug("... Push the data to hadoop")
        okapi_data = self.input_pandas_to_okapi(data)
        run(self.source % ("echo '%(data)s' > %(data_path_tmp)s && "
            "hadoop dfs -copyFromLocal %(data_path_tmp)s %(data_path)s && "
            "rm %(data_path_tmp)s" % {
            "data": okapi_data,
            "data_path": self.input,
            "data_path_tmp": self._data_hash
            }), quiet=True)
        logger.debug("Done!")

    def data_in_machine(self, data):
        """
        Check if data in machine

        :return: True if data in machine
        """
        logger.debug("... Check if data in the hadoop")
        check_if_exists = "hadoop dfs -ls %s" % self.input
        if 0 == run(self.source % check_if_exists, warn_only=True, quiet=True).return_code:
            logger.debug("Data already in hadoop")
            return True
        else:
            logger.debug("Data not in hadoop")
            return False


class RandomOkapi(DynamicOkapiModel):
    """
    Random okapi model creator
    """

    def __init__(self, **kwargs):
        """
        Constructor for Random
        :param kwargs: Parameters for BaseOkapiModel
        """
        super(RandomOkapi, self).__init__(model_java_class="ml.grafos.okapi.cf.ranking.RandomRankingComputation",
                                          **kwargs)

    @property
    def name(self):
        """
        The name of this model
        :return:
        """
        return "random"

import logging

logging.disable(logging.WARNING)

class PopularityOkapi(DynamicOkapiModel):
    """
    Popularity models calculated using hadoops okapi
    """

    def __init__(self, **kwargs):
        """
        Constructor for Popularity
        :param kwargs: Parameters for BaseOkapiModel
        """
        super(PopularityOkapi,
              self).__init__(model_java_class="ml.grafos.okapi.cf.ranking.PopularityRankingComputation", **kwargs)

    @property
    def name(self):
        """
        The name of this model
        :return:
        """
        return "popularity"


class BPROkapi(DynamicOkapiModel):
    """
    BPR models calculated using hadoops okapi
    """

    def __init__(self, **kwargs):
        """
        Constructor for BPR
        :param kwargs: Parameters for BaseOkapiModel
        """
        super(BPROkapi,
              self).__init__(model_java_class="ml.grafos.okapi.cf.ranking.BPRRankingComputation", **kwargs)

    @property
    def name(self):
        """
        The name of this model
        :return:
        """
        return "BPR"




class TFMAPOkapi(DynamicOkapiModel):
    """
    TFMAP models calculated using hadoops okapi
    """

    def __init__(self, **kwargs):
        """
        Constructor for TFMAP
        :param kwargs: Parameters for BaseOkapiModel
        """
        super(TFMAPOkapi,
              self).__init__(model_java_class="ml.grafos.okapi.cf.ranking.TFMAPRankingComputation", **kwargs)

    @property
    def name(self):
        """
        The name of this model
        :return:
        """
        return "TFMAP"


class SGDOkapi(DynamicOkapiModel):
    """
    SGD models calculated using hadoops okapi
    """

    def __init__(self, **kwargs):
        """
        Constructor for SGD
        :param kwargs: Parameters for BaseOkapiModel
        """
        super(SGDOkapi,
              self).__init__(model_java_class="ml.grafos.okapi.cf.sgd.Sgd$InitUsersComputation", **kwargs)

    @property
    def name(self):
        """
        The name of this model
        :return:
        """
        return "SGD"


class ALSOkapi(DynamicOkapiModel):
    """
    ALS models calculated using hadoops okapi
    """

    def __init__(self, **kwargs):
        """
        Constructor for ALS
        :param kwargs: Parameters for BaseOkapiModel
        """
        super(ALSOkapi,
              self).__init__(model_java_class="ml.grafos.okapi.cf.als.Als$InitUsersComputation", **kwargs)

    @property
    def name(self):
        """
        The name of this model
        :return:
        """
        return "ALS"


class SVDOkapi(DynamicOkapiModel):
    """
    SVD models calculated using hadoops okapi
    """

    def __init__(self, **kwargs):
        """
        Constructor for SVD
        :param kwargs: Parameters for BaseOkapiModel
        """
        super(SVDOkapi,
              self).__init__(model_java_class="ml.grafos.okapi.cf.svd.Svdpp$InitUsersComputation", **kwargs)

    @property
    def name(self):
        """
        The name of this model
        :return:
        """
        return "SVD"


class ClimfOkapi(DynamicOkapiModel):
    """
    Climf models calculated using hadoops okapi
    """

    def __init__(self, **kwargs):
        """
        Constructor for Climf
        :param kwargs: Parameters for BaseOkapiModel
        """
        super(ClimfOkapi,
              self).__init__(model_java_class="ml.grafos.okapi.cf.ranking.ClimfRankingComputation", **kwargs)

    @property
    def name(self):
        """
        The name of this model
        :return:
        """
        return "Climf"


if __name__ == "__main__":
    import testfm
    df = pd.read_csv(
        resource_filename(
            testfm.__name__, 'data/movielenshead.dat'), sep="::", header=None,
        names=['user', 'item', 'rating', 'date', 'title'])
    for r_class in [RandomOkapi,
                    PopularityOkapi,
                    #BPROkapi,
                    #TFMAPOkapi,
                    #SGDOkapi,
                    #ALSOkapi,
                    #SVDOkapi,
                    #ClimfOkapi
                    ]:
        r = r_class(host="igraph-01", user="joaonrb",
                    okapi_jar_dir="okapi/jar/efe97a00d2a1b3f30dbbaddb3f3dd4c7/",
                    okapi_jar_base_name="okapi-0.3.2-SNAPSHOT-jar-with-dependencies.jar",
                    hadoop_source="/data/b.ajf/hadoop1_env.sh")
        r.fit(df)
        print r.getScore(1, 1)