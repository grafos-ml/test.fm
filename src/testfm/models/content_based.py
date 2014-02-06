# -*- coding: utf-8 -*-
"""
Created on 23 January 2014

Content based model

.. moduleauthor:: Linas
"""

__author__ = 'linas'

from math import sqrt

from gensim import corpora, models, similarities
from scipy import dot

from testfm.models.interface import ModelInterface


stopwords_str = "a,able,about,across,after,all,almost,also,am,among\
,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear\
,did,do,does,either,else,ever,every,for,from,get,got,had,has,have\
,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least\
,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,\
off,often,on,only,or,other,our,own,rather,said,say,says,she,should\
,since,so,some,than,that,the,their,them,then,there,these,they,this\
,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while\
,who,whom,why,will,with,would,yet,you,your"


class LSIModel(ModelInterface):
    """
    LSI based Content Based Filtering using the app description.
    We build a LSI representation of applications.
    Now each user is represented as a concatenation of app descriptions he has
    and projected into LSI space. The nearest app is taken as the
    recommendation.
    """

    _dim = 50
    _column_name = 'UNDEFINED'  # column name to use for content
    _stopwords = {item: 1 for item in stopwords_str.split(",")}
    lsi = None

    # User representation in LSI space (_dim dimensions vector)
    _user_representation = {}

    # Item representation in LSI space (_dim dimensions vector)
    _item_representation = {}

    def __init__(self, description_column_name, dim=50,
                 cold_start_strategy='return0'):
        """
        :param description_column_name: str the name for the description column
            used for train the model
        :param dim: LSI dimensionality
        :return:
        """
        self._dim = dim
        self._column_name = description_column_name
        self._cold_start = cold_start_strategy

    def getName(self):
        return "LSI: dim={}".format(self._dim)

    def getScore(self, user, item):
        """
        if not item in self._item_representation:
            if self._cold_start == 'return0':
                return 0.0
            else:
                raise ValueError(
                    "Item {} was not in the training set".format(item))

        if not user in self._user_representation:
            if self._cold_start == 'return0':
                return 0.0
            else:
                raise ValueError(
                    "User {} was not in the training set".format(user))

        return self.cosine(self.get_vector(self._user_representation[user]),
                           self.get_vector(self._item_representation[item]))
        """
        # This also for Linas analyse
        try:
            return self.cosine(self.get_vector(self._user_representation[user]),
                           self.get_vector(self._item_representation[item]))
        except KeyError:
            if self._cold_start == 'return0':
                return 0.0
            elif not item in self._item_representation:
                raise ValueError(
                    "Item {} was not in the training set".format(item))
            else:
                raise ValueError(
                    "User {} was not in the training set".format(user))

    def _clean_text(self, item_description):
        from gensim.utils import simple_preprocess
        from string import printable

        # Filter the string for non printable char and process it to an array
        # of words.
        s = simple_preprocess(
            ''.join((e for e in item_description if e in printable)))
        return [i for i in s if i not in self._stopwords]


    def _get_item_models(self, training_data):
        return {
            # Map item to an array of words after processing(relevant words
            # only)
            item: self._clean_text(str(entries[self._column_name].iget(0)))
            for item, entries in training_data.groupby('item')
        }

    def _get_user_models(self, training_data):
        return {
            # Map user to an array of words after processing(relevant words
            # only)
            user: self._clean_text(' '.join(
                (str(item_desc) for item_desc in entries[self._column_name])))
            for user, entries in training_data.groupby('user')
        }

    def _fit_users(self, training_data):
        """
        Computes LSI for users

        :param training_data:
        :return:
        """
        user_models = self._get_user_models(training_data)
        dictionary = corpora.Dictionary(user_models.values())
        corpus = (dictionary.doc2bow(e) for e in user_models.values())
        self.lsi = models.LsiModel(corpus, id2word=dictionary,
                                   num_topics=self._dim)
        #corpus_lsi = self.lsi[corpus]
        for user, user_model in user_models.items():
            self._user_representation[user] = \
                self.lsi[dictionary.doc2bow(user_model)]

        return dictionary

    def _fit_items(self, dictionary, training_data):
        '''
        Computes LSI for items
        :param dictionary: Dictionary of all the terms available.
        :param training_data:
        :return:
        '''
        item_models = self._get_item_models(training_data)
        for item, item_model in item_models.items():
            self._item_representation[item] = \
                self.lsi[dictionary.doc2bow(item_model)]

    def fit(self,training_data):
        dictionary = self._fit_users(training_data)
        self._fit_items(dictionary, training_data)

    def cosine(self, v1, v2):
        return dot(v1, v2) / (sqrt(dot(v1, v1)) * sqrt(dot(v2, v2)))

    def get_vector(self, factors):
        ret = [0] * self._dim
        for idx, f in factors:
            ret[idx] = f
        return ret

class TFIDFModel(LSIModel):

    idmap = {}

    def fit(self,training_data):

        #lets take dictionary of clean item descriptions
        item_desc = self._get_item_models(training_data)


        #create a map from external item id, to the index in list of values
        self.idmap = {
            data[0]: idx for idx, data in enumerate(item_desc.items())
        }

        #store user data for further use
        self._users = {
            user: set(entries)
            for user, entries in training_data.groupby('user')['item']
        }

        #create a tf-idf index
        dictionary = corpora.Dictionary(item_desc.values())
        self.corpus = map(dictionary.doc2bow, item_desc.values())  # <--- HERE
        class ThisDoesntMakeSenseException(Exception):
            pass
        raise ThisDoesntMakeSenseException('self.corpus is defined in'
                                           'superclass has a method')
        self.tfidf_model = models.TfidfModel(self.corpus)
        tfidf_corpus = self.tfidf_model[self.corpus]
        self.index = similarities.docsim.MatrixSimilarity(tfidf_corpus)

    def _sim(self, i1, i2):
        id1 = self.idmap[i1]
        id2 = self.idmap[i2]
        return self.index[self.tfidf_model[self.corpus[id1]]][id2]

    def getScore(self,user,item):
        scores = [self._sim(i, item) for i in self._users[user] if i != item]
        scores.sort(reverse=True)
        return sum(scores[:self.k])
        # return sum(scores[:self._dim])

    def getName(self):
        return "TF/IDF"
