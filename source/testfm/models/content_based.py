__author__ = 'linas'

from gensim import corpora, models
from interface import ModelInterface
from scipy import dot
from math import sqrt

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
    '''
    LSI based Content Based Filtering using the app description.
    We build a LSI representation of applications.
    Now each user is represented as a concatanation of app descriptions he has
    and projected into LSI space. The nearest app is taken as the recommendation.

    '''

    _dim = 50
    _column_name = 'UNDEFINED' #column name to use for content
    _stopwords = dict([(item, 1) for item in stopwords_str.split(",")])
    _user_representation = {} #user representation in LSI space (_dim dimensions vector)
    _item_representation = {} #item representation in LSI space (_dim dimensions vector)

    def __init__(self, description_column_name, dim=50, cold_start_strategy='return0'):
        '''
        :param description_column_name: str the name for the description column used for train the model
        :param dim: LSI dimensionality
        :return:
        '''
        self._dim = dim
        self._column_name = description_column_name
        self._cold_start = cold_start_strategy

    def getName(self):
        return "LSI: dim={}".format(self._dim)

    def getScore(self,user,item):
        if not item in self._item_representation:
            if self._cold_start == 'return0':
                return 0.0
            else:
                raise ValueError("Item {} was not in the training set".format(item))

        if not user in self._user_representation:
            if self._cold_start == 'return0':
                return 0.0
            else:
                raise ValueError("User {} was not in the training set".format(user))

        return self.cosine(self.get_vector(self._user_representation[user]),
                           self.get_vector(self._item_representation[item]))

    def _clean_text(self, item_description):
        from gensim.utils import simple_preprocess
        from string import printable

        s = filter(lambda x: x in printable, item_description)
        s = simple_preprocess(s)
        s = [i for i in s if i not in self._stopwords]
        return s

    def _get_item_models(self, training_dataframe):
        ret = {}
        grouped = training_dataframe.groupby('item')
        for item, entries in grouped:
            #print entries[self._column_name]
            desc = entries[self._column_name].iget(0)
            desc = self._clean_text(desc)
            ret[item] = desc
        return ret

    def _get_user_models(self, training_dataframe):
        ret = {}
        grouped = training_dataframe.groupby('user')
        for user, entries in grouped:
            user_model = " ".join([item_desc for item_desc in entries[self._column_name]])
            user_model = self._clean_text(user_model)
            ret[user] = user_model
        return ret

    def _fit_users(self, training_dataframe):
        '''
        Computes LSI for users

        :param training_dataframe:
        :return:
        '''
        user_models = self._get_user_models(training_dataframe)
        dictionary = corpora.Dictionary(user_models.values())
        corpus = map(dictionary.doc2bow, user_models.values())
        self.lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=self._dim)
        #corpus_lsi = self.lsi[corpus]
        for user, user_model in user_models.items():
            um = dictionary.doc2bow(user_model)
            self._user_representation[user] = self.lsi[um]

        return dictionary

    def _fit_items(self, dictionary, training_dataframe):
        '''
        Computes LSI for items
        :param dictionary: Dictionary of all the terms available.
        :param training_dataframe:
        :return:
        '''
        item_models = self._get_item_models(training_dataframe)
        for item, item_model in item_models.items():
            im = dictionary.doc2bow(item_model)
            self._item_representation[item] = self.lsi[im]

    def fit(self,training_dataframe):
        dictionary = self._fit_users(training_dataframe)
        self._fit_items(dictionary, training_dataframe)

    def cosine(self, v1, v2):
        return dot(v1, v2) / (sqrt(dot(v1, v1)) * sqrt(dot(v2, v2)))

    def get_vector(self, factors):
        ret = [0] * self._dim
        for idx, f in factors:
            ret[idx] = f
        return ret