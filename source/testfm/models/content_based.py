__author__ = 'linas'

from random import random
from interface import ModelInterface
from pandas import DataFrame


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

    _dim = 10
    _column_name = 'UNDEFINED'
    _stopwords = dict([(item, 1) for item in stopwords_str.split(",")])
    _user_representation = {}

    def __init__(self, description_column_name, dim=10):
        '''
        :param description_column_name: str the name for the description column used for train the model
        :param dim: LSI dimensionality
        :return:
        '''
        self._dim = dim
        self._column_name = description_column_name

    def getScore(self,user,item):
        key = (user, item)
        if key in self._scores:
            return self._scores[key]
        else:
            s = random()
            self._scores[key] = s
            return s

    def _clean_text(self, item_description):
        from gensim.utils import simple_preprocess
        from string import printable

        s = filter(lambda x: x in printable, item_description)
        s = simple_preprocess(s)
        s = [i for i in s if i not in self._stopwords]
        return s

    def fit(self,training_dataframe):
        from gensim import corpora, models, similarities
        texts = []#descriptions of items
        id_map = {}

        training_dataframe = DataFrame(training_dataframe)
        grouped = training_dataframe.groupby('user')
        for user, entries in grouped:
            user_model = " ".join([item_desc for item_desc in entries[self._column_name]])
            user_model = self._clean_text(user_model)
            id_map[len(texts)] = user
            texts.append(user_model)

        print texts
        dictionary = corpora.Dictionary(texts)
        corpus = map(dictionary.doc2bow, texts)
        self.lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=self._dim)
        corpus_lsi = self.lsi[corpus]
        for user_idx, factors in enumerate(corpus_lsi):
            self._user_representation[id_map[user_idx]] = factors

