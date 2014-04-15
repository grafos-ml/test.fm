"""
Nosetest package for the okapi connector class. This test implies that you have access to the okapi server or something
"""

__author__ = 'joaonrb'

import os
from testfm.okapi.connector import RandomOkapi, OkapiNoResultError, PopularityOkapi, BPROkapi, REMOTE_HOST
import testfm
from testfm.splitter.holdout import RandomSplitter
from testfm.models.baseline_model import Popularity
from testfm.models.bpr import BPR
import pandas as pd
from pkg_resources import resource_filename
import unittest

ON_REMOTE_NETWORK = os.system("ping -c 1 " + REMOTE_HOST) == 0


class TestOkapi(object):
    """
    Test for okapi connector
    """

    df = None
    random_okapi = None
    popularity = None
    bpr = None

    def setUp(self):
        """
        Setup the test package
        """
        self.df = pd.read_csv(resource_filename(testfm.__name__, 'data/movielenshead.dat'), sep="::", header=None,
                              names=['user', 'item', 'rating', 'date', 'title'])
        self.random_okapi = RandomOkapi(host="igraph-01", user="joaonrb",
                                        okapi_jar_dir="okapi/jar/efe97a00d2a1b3f30dbbaddb3f3dd4c7/",
                                        okapi_jar_base_name="okapi-0.3.2-SNAPSHOT-jar-with-dependencies.jar",
                                        hadoop_source="/data/b.ajf/hadoop1_env.sh")
        self.popularity = PopularityOkapi(host="igraph-01", user="joaonrb",
                                          okapi_jar_dir="okapi/jar/efe97a00d2a1b3f30dbbaddb3f3dd4c7/",
                                          okapi_jar_base_name="okapi-0.3.2-SNAPSHOT-jar-with-dependencies.jar",
                                          hadoop_source="/data/b.ajf/hadoop1_env.sh")
        self.bpr = BPROkapi(host="igraph-01", user="joaonrb",
                            okapi_jar_dir="okapi/jar/efe97a00d2a1b3f30dbbaddb3f3dd4c7/",
                            okapi_jar_base_name="okapi-0.3.2-SNAPSHOT-jar-with-dependencies.jar",
                            hadoop_source="/data/b.ajf/hadoop1_env.sh")

    '''
    @unittest.skipIf(not ON_REMOTE_NETWORK, "Not in igraph-01 network")
    def test_get_result(self):
        """
        Test if result exist behavior and get result.
        """
        result_file = self.random_okapi.get_result_location(self.df)
        if self.random_okapi.result_exist_for(result_file):
            user, item = self.random_okapi.result
            assert isinstance(user, pd.DataFrame), "First element in tuple is not a pandas DataFrame"
            assert isinstance(item, pd.DataFrame), "Second element in tuple is not a pandas DataFrame"
        else:
            try:
                result = self.random_okapi.result
            except OkapiNoResultError:
                pass
            else:
                assert False, "RandomOkapi.result is returning %s when RandomOkapi.result_exist_for is false" % result

    @unittest.skipIf(not ON_REMOTE_NETWORK, "Not in igraph-01 network")
    def test_map(self):
        """
        Test the mapping
        """
        self.random_okapi.map_data(self.df)
        users = enumerate(set(self.df["user"]), start=1)
        items = enumerate(set(self.df["item"]), start=1)
        for user_id, user in users:
            assert self.random_okapi.data_map["user_to_id"][user] == user_id, "Mapping in user to id is not correct"
            assert self.random_okapi.data_map["id_to_user"][user_id] == user, "Mapping in id user is not correct"

        for item_id, item in items:
            assert self.random_okapi.data_map["item_to_id"][item] == item_id, "Mapping in item to id is not correct"
            assert self.random_okapi.data_map["id_to_item"][item_id] == item, "Mapping in id item is not correct"

    @unittest.skipIf(not ON_REMOTE_NETWORK, "Not in igraph-01 network")
    def test_repeating_spliced_data(self):
        """
        Test repeating data on the same model by splitting a dataFrame
        """
        splitter = RandomSplitter()
        for i in range(10):
            training, testing = splitter.split(self.df, 0.20)
            self.popularity.fit(training)
            for _, row in testing.iterrows():
                score = self.popularity.getScore(row["user"], row["item"])
                assert isinstance(score, float), "The result of the score is not a float"
    '''
    @unittest.skipIf(not ON_REMOTE_NETWORK, "Not in igraph-01 network")
    def test_popularity(self):
        """
        Test the popularity okapi algorithm
        """
        splitter = RandomSplitter()
        training, testing = splitter.split(self.df, 0.20)
        pop = Popularity(normalize=False)
        pop.fit(training)
        self.popularity.fit(training)
        for _, row in testing.iterrows():
            assert row["user"] in training["user"]
            python_score = pop.getScore(row["user"], row["item"])
            okapi_score = self.popularity.getScore(row["user"], row["item"])
            assert okapi_score == python_score, \
                "Okapi popularity(%f) don't give the same score as his python implementation(%f)" % (okapi_score,
                                                                                                     python_score)
    '''
    @unittest.skipIf(not ON_REMOTE_NETWORK, "Not in igraph-01 network")
    def test_bpr(self):
        """
        Test the bpr okapi algorithm
        """
        splitter = RandomSplitter()
        training, testing = splitter.split(self.df, 0.20)
        bpr = BPR()
        bpr.fit(training)
        self.bpr.fit(training)
        for _, row in testing.iterrows():
            okapi_score = self.bpr.getScore(row["user"], row["item"])
            python_score = bpr.getScore(row["user"], row["item"])
            assert okapi_score == python_score, \
                "Okapi bpr(%f) don't give the same score as his python implementation(%f)" % (okapi_score, python_score)
    '''