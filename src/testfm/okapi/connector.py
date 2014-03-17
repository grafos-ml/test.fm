# -*- coding: utf-8 -*-
"""
Created at March 12, 2014

Connect to the okapi to create some model.

.. moduleauthor:: joaonrb <joaonrb@gmail.com>

"""
__author__ = "joaonrb"


import os
import logging
import hashlib
from fabric.api import env, run, put
from pkg_resources import resource_filename
from testfm import okapi

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.addHandler(logging.StreamHandler())

OKAPI_REMOTE = "joaonrb@igraph-01"  # <-- Change to the proper remote account

env.hosts = [
    OKAPI_REMOTE
]

HADOOP_COMMAND = "hadoop jar %(hadoop_parameters)s"
GIRAPH_COMMAND = HADOOP_COMMAND % {
    "hadoop_parameters": "%(giraph_jar)s org.apache.giraph.GiraphRunner -Dmapred.job.name=OkapiTrainModelTask "
                         "-Dmapred.reduce.tasks=0 -libjars %(giraph_jar)s,%(okapi_jar)s %(giraph_parameters)s"
}
OKAPI_COMMAND = GIRAPH_COMMAND % {
    "giraph_parameters": "-Dmapred.child.java.opts=-Xmx1g -Dgiraph.zkManagerDirectory=_bsp "
                         "-Dgiraph.useSuperstepCounters=false %(model_class)s "
                         "-eif ml.grafos.okapi.cf.CfLongIdFloatTextInputFormat -eip %(input)s "
                         "-vof org.apache.giraph.io.formats.IdWithValueTextOutputFormat -op %(output)s -w 1 "
                         "-ca giraph.numComputeThreads=1 -ca minItemId=1 -ca maxItemId=%(max_item_id)s",
    "giraph_jar": "%(giraph_jar)s",
    "okapi_jar": "%(okapi_jar)s"
}

HADOOP_SOURCE = "source /data/b.ajf/hadoop1_env.sh && %s"

class OkapiConnectorError(Exception):
    """
    Generic exception for okapi conection models for test.fm
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


class ModelConnector(object):
    """
    Connect a model to okapi
    """

    OKAPI_REPOSITORY = "okapi"
    OKAPI_JAR_REPOSITORY = OKAPI_REPOSITORY + "/jar"
    OKAPI_DATA_REPOSITORY = OKAPI_REPOSITORY + "/data"
    OKAPI_RESULTS_REPOSITORY = OKAPI_REPOSITORY + "/results"

    OKAPI_LOCAL_REPOSITORY = resource_filename(okapi.__name__, "lib/")
    GIRAPH_JAR = "giraph-1.1.0-SNAPSHOT-for-hadoop-0.20.203.0-jar-with-dependencies.jar"
    OKAPI_JAR = "okapi-0.3.2-SNAPSHOT-jar-with-dependencies.jar"

    EXTRA_JAR = []

    _result = None
    data_map = {}

    @staticmethod
    def get_jar_location(jar):
        """
        Returns the location in the remote system of this jar file

        :param jar: The jar file
        :return: The location of the jar
        """
        return "%(dir)s/%(jar_place)s" % {
            "dir": ModelConnector.OKAPI_JAR_REPOSITORY,
            "jar_place": ModelConnector.hash_file(jar)
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
        okapi_rows = ("%(user)s %(item)s %(rating)s" % {  # okapi line
            "user": self.data_map["user_to_id"][row["user"]],
            "item": self.data_map["item_to_id"][row["item"]],
            "rating": row["rating"]
        } for _, row in data.iterrows())
        return "\n".join(okapi_rows)

    def map_data(self, data):
        """
        Maps the data to indexes starting in one
        :param data: Pandas DataFrame with the data
        """
        self.data_map = {
            "user_to_id": {},
            "id_to_user": {},
            "item_to_id": {},
            "id_to_item": {}
        }
        users = enumerate(set(data["user"]))
        items = enumerate(set(data["item"]))
        for user_id, user in users:
            self.data_map["user_to_id"][user] = user_id
            self.data_map["id_to_user"][user_id] = user

        for item_id, item in items:
            self.data_map["item_to_id"][item] = item_id
            self.data_map["id_to_item"][item_id] = item

    @staticmethod
    def exist_for(path):
        """
        Check if the path exists in the remote

        :param path: Data to compute
        :return: True if the data is already computed with this algorithm
        """
        command = "[ -f %s ] && echo 1 || echo 0" % path
        do_result_exist = run(command, quiet=True)
        return bool(int(do_result_exist))

    @property
    def jar_dependencies(self):
        """
        Return a generator with the needed jars

        :return: A generator with strings
        """
        yield self.GIRAPH_JAR
        yield self.OKAPI_JAR
        for jar in self.extra_jar:
            yield jar

    @property
    def extra_jar(self):
        """
        Load extra jars in to remote
        :return:
        """
        extras = self.EXTRA_JAR
        for jar in set(extras):
            yield jar

    @property
    def result(self):
        """
        Get the result of this model

        :return: The result
        :raise OkapiNoResultError: Raise when the result was not fetched
        """
        return self._result or OkapiNoResultError.raise_this("This model has not ben calculated or is result as not "
                                                             "been fetched.")

    @property
    def name(self):
        """
        Return the name of this model. The default version returns the name of the python class
        :return: The name of the python class
        """
        return self.__class__.__name__

    def call_okapi(self, data):
        """
        Call the okapi framework to make a model

        :param data: Data to produce the model
        """
        # Map the data
        self.map_data(data)
        logger.info("Checking if result is computed ..")
        result_file = self.get_result_location(data)
        if not self.result_exist_for(result_file):
            logger.info("- Result is not computed yet ..")
            logger.info("Preparing environment ..")
            command = self.get_remote_command(data)
            self.execute_okapi(command)
            self.push_result_from_hadoop(result_file)
        self.read_result(result_file)
        logger.info("- Done ..")
        return self.result

    def result_exist_for(self, result_file):
        """
        Check if the result of this algorithm with this data is already computed.

        :param result_file: Location in the remote of the data to compute
        :return: True if the data is already computed with this algorithm
        """
        return self.exist_for(result_file)

    def read_result(self, file_location):
        """
        Get the result in file_location to self.result

        :param file_location: The location of the result in the remote location
        """
        logger.info("Reading result from remote to local ..")
        result_data = run("cat %s" % file_location, quiet=True)
        print(result_data)
        self._result = result_data
        logger.info("- Result in local ..")

    def get_jar(self, jar_name):
        """
        Return a open file of the jar. This jar is expected to be in the OKAPI_LOCAL_REPOSITORY of the class.

        :param jar_name: A string for the jar file name
        :return: The jar as a open file.
        :raise OkapiJarNotInRepository: When the jar is not in the repository
        """
        try:
            jar_file = open(self.OKAPI_LOCAL_REPOSITORY+"/%s" % jar_name)
        except IOError:
            raise OkapiJarNotInRepository("The jar %s is not in %s" % (jar_name, self.OKAPI_LOCAL_REPOSITORY))
        return jar_file

    @staticmethod
    def upload(some_file, to=None):
        """
        Upload "some_file" to the remote

        :param some_file: The file path in the local machine to put in the remote
        :param to: The path in the remote to put the file
        :raise TypeError: When to is not a string
        """
        if not (isinstance(to, str) and isinstance(some_file, str)):
            raise TypeError("First parameter and to keyword parameter must be a string")
        run("[ -d %s ] || mkdir %s" % (to, to))
        put(some_file, to, quiet=True)

    def upload_jar(self, jar_file, to=None):
        """
        Upload jar_file to remote directory to the file "to"

        :param jar_file: The file to go to the remote
        :param to: The remote file where jar_file is going to be be copied
        :raise TypeError: When to is not a string
        """
        logger.info("Uploading %s to remote:%s .." % (jar_file.name, to))
        jar_path = os.path.abspath(jar_file.name)
        self.upload(jar_path, to=to)

    def upload_data(self, data, to=None):
        """
        Copy the data inside the pandas DataFrame to remote location
        :param data: Pandas DataFrame with the data
        :param to: The remote file where the data is going to be be copied
        """
        logger.info("Uploading data to remote:%s .." % to)
        if not isinstance(to, str):
            raise TypeError("Keyword parameter \"to\" must be a string")
        data_in_okapi = self.input_pandas_to_okapi(data)
        run("echo \"%(okapi_data)s\">%(to)s" % {
            "okapi_data": data_in_okapi,
            "to": to
            }, quiet=True)

    def get_remote_command(self, data):
        """
        Prepare the environment to this model.

        :param data: data that also should be prepared
        :return: The command
        """
        jars = {}

        logger.info("Preparing jars ..")
        # Check if jars exist and load them if they don't
        for jar_name in self.jar_dependencies:  # jar_dependencies should give a list with all jars needed
            jar_file = self.get_jar(jar_name)  # Get a open file of the jar
            jar_location = self.get_jar_location(jar_file)  # Get the location where the jar should be in the remote

            # Keep the locations in a mapped structure
            jars[jar_name] = jar_location

            logger.info("Check if %s is in remote .." % jar_name)
            # If the jar don't exist in the remote than a copy should be uploaded
            if not self.exist_for(jar_location+"/%s" % jar_name):
                logger.info("- Jar %s is not in remote .." % jar_name)
                self.upload_jar(jar_file, to=jar_location)
            logger.info("- %s ready in remote .." % jar_name)

        logger.info("Preparing data ..")
        # Check if data is in the remote. If don't it upload it.
        data_location = self.get_data_location(data)
        logger.info("Check if data %s is in remote .." % data_location)
        if not self.exist_for(data_location):
            logger.info("- Data %s is not in remote .." % data_location)
            self.upload_data(data, to=data_location)

        logger.info("- Data is ready in remote ..")

        self.put_data_to_hadoop(data_location)
        return self.build_command(jars=jars)

    @property
    def std_output_name(self):
        """
        The standard output name in the hadoop file system
        :return: A string
        """
        return "%s_output" % self.name

    @property
    def std_input_name(self):
        """
        The standard input name in the hadoop file system
        :return: A string
        """
        return "%s_input" % self.name

    def build_command(self, jars=None):
        """
        Build the okapi model creation command

        :param data: The location of the data in the remote
        :param jars: The location of the jars by name
        :type jars: dict
        :return: A str with the full okapi command
        """
        giraph = jars[self.GIRAPH_JAR]+"/%s" % self.GIRAPH_JAR
        okapi = jars[self.OKAPI_JAR]+"/%s" % self.OKAPI_JAR
        command = "HADOOP_PATH=%s %s" % ("%s:%s" % (giraph, okapi), OKAPI_COMMAND % {
            "model_class": self.model_class,
            "giraph_jar": giraph,
            "okapi_jar": okapi,
            "max_item_id": len(self.data_map["item_to_id"]),
            "input": self.std_input_name,
            "output": self.std_output_name
        })
        return command

    def put_data_to_hadoop(self, data_location):
        """
        Puts the data in the hadoop file system. If there is data with the same name there it erases it
        :param data_location: The location of the data in the remote file system
        """
        logger.info("Pushing data to hadoop ..")
        logger.info("Remove %s if exists .." % self.std_input_name)
        remove_old_command = "hadoop dfs -rmr %s" % self.std_input_name
        run(remove_old_command, quiet=True)
        logger.info("- %s removed .." % self.std_input_name)

        logger.info("Create %s with new data .." % self.std_input_name)
        put_new_data = HADOOP_SOURCE % "hadoop dfs -copyFromLocal %s %s" % (data_location, self.std_input_name)
        run(put_new_data, quiet=True)

        logger.info("- Data in hadoop ..")

    @staticmethod
    def execute_okapi(command):
        """
        Execute this command fetch the result and returns it.

        :param command: Hadoop command to be executed
        :param data: The data location
        :return: The data in a pandas DataFrame
        """
        logger.info("Hadoop is building the model ..")
        logger.info("Running: %s" % command)
        run(HADOOP_SOURCE % command, quiet=False)
        logger.info("- Hadoop finished ..")

    def push_result_from_hadoop(self, data_location):
        """
        Push the result from the hadoop to the remote
        :param data_location: The location of the data in the remote
        """
        logger.info("Pushing the result from hadoop to remote ..")
        run(HADOOP_SOURCE % "hadoop dfs -copyToLocal %s/* okapi/tmp" % self.std_output_name, quiet=True)
        run("for f in `ls okapi/tmp | sort -V`; do cat $f >> output; done;", quiet=True)
        logger.info("- Result in remote ..")


class RandomOkapi(ModelConnector):
    """
    Test
    """

    @property
    def name(self):
        return "random"

    @property
    def model_class(self):
        return "Pop"

if __name__ == "__main__":
    import pandas as pd
    import testfm
    env.hosts = [OKAPI_REMOTE]
    df = pd.read_csv(
        resource_filename(
            testfm.__name__, 'data/movielenshead.dat'), sep="::", header=None,
        names=['user', 'item', 'rating', 'date', 'title'])
    r = RandomOkapi()
    r.call_okapi(df)