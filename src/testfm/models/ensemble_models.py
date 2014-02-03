__author__ = 'linas'

'''
Ensemble models are the ones that take several already built models and combine
them into a single model.
'''



from testfm.models.interface import ModelInterface

class LinearEnsemble(ModelInterface):

    _models = []
    _weights = []

    def __init__(self, models, weights=None):
        '''
        :param models: list of ModelInterface subclasses
        :param weights: list of floats with weights telling how to combine the 
            models
        :return:
        '''

        if weights is not None:
            if len(models) != len(weights):
                raise ValueError("The weights vector length should be the same "
                    "as number of models")

        self._weights = weights
        self._models = models

    def fit(self,training_dataframe):
        pass

    def getScore(self,user,item):
        '''
        :param user:
        :param item:
        :return:
        >>> from testfm.models.baseline_model import IdModel, ConstantModel
        >>> model1 = IdModel()
        >>> model2 = ConstantModel(1.0)
        >>> ensamble = LinearEnsemble([model1, model2], weights=[0.5, 0.5])
        >>> ensamble.getScore(0, 5)
        3.0

        3 because we combine two models in a way: 5 (id of item)*0.5+1(constant
        factor)*0.5

        '''
        predictions = [m.getScore(user, item) for m in self._models]
        return sum([w*p for w,p in zip(self._weights, predictions)])

    def getName(self):
        models = ",".join([m.getName() for m in self._models])
        weights = ",".join(["{:1.4f}".format(w) for w in self._weights])
        return "Linear Ensamble ("+models+"|"+weights+")"


class LogisticEnsemble(ModelInterface):
    '''
    A linear ensemble model which is learned using logistic regression.
    '''

    _user_count = {}
    _item_features = None

    def getScore(self,user,item):
        x, y = self._extract_features(user, item)
        return float(self.model.predict(x))

    def getName(self):
        models = ",".join([m.getName() for m in self._models])
        return "Logistic Ensamble ("+models+")"

    def __init__(self, models, item_features_column=[]):
        self._models = models
        self.item_features_column = item_features_column

    def _prepare_feature_extraction(self, df):
        '''
        Extracts size of user profile info and item price
        '''
        grouped = df.groupby('user')
        for user, entries in grouped:
            self._user_count[user] = len(entries)

        if self.item_features_column:
            grouped = df.groupby(['item']+self.item_features_column)
            self._item_features = {}
            for item, entries in grouped:
                self._item_features[item[0]] = item[1:]
        #print self._item_features

    def _extract_features(self, user, item, relevant=True):
        '''
        Gives proper feature for the logistic function to train on.
        '''
        features = [self._user_count.get(user, 0)]
        if self._item_features:
            features += [f for f in self._item_features[item]]
        features += [m.getScore(user, item) for m in self._models]

        if relevant:
            return features, 1
        else:
            return features, 0

    def prepare_data(self, df):
        from random import choice
        X = []
        Y = []
        items = df.item.unique()
        self._prepare_feature_extraction(df)

        for _, tuple in df.iterrows():
            x, y = self._extract_features(tuple['user'], tuple['item'], relevant=True)
            X.append(x)
            Y.append(y)
            bad_item = choice(items)
            x, y = self._extract_features(tuple['user'], bad_item, relevant=False)
            X.append(x)
            Y.append(y)
        return X, Y

    def fit(self, df):
        from sklearn.linear_model import LogisticRegression

        X, Y = self.prepare_data(df)
        self.model = LogisticRegression(C=10, penalty='l1', tol=0.1)
        self.model.fit(X, Y)

class LinearFit(LogisticEnsemble):

    def fit(self, df):
        from sklearn.linear_model import LinearRegression

        X, Y = self.prepare_data(df)
        self.model = LinearRegression(copy_X=False)
        self.model.fit(X, Y)
        #print self.model.coef_

    def _extract_features(self, user, item, relevant=True):
        '''
        Gives proper feature for the logistic function to train on.
        '''

        features = [1, self._user_count.get(user, 0)]
        features += [m.getScore(user, item) for m in self._models]

        if relevant:
            return features, 1
        else:
            return features, 0

    def getName(self):
        models = ",".join([m.getName() for m in self._models])
        return "Linear Ensamble ("+models+")"


class LinearRank(LogisticEnsemble):

    def __init__(self, models, item_features_column=[]):
        super(LinearRank, self).__init__(models, item_features_column)

    def fit(self, df):
        from sklearn.linear_model import LinearRegression

        X, Y = self.prepare_data(df)
        self.model = LinearRegression(copy_X=False)
        self.model.fit(X, Y)
        #print self.model.coef_

    def getName(self):
        models = ",".join([m.getName() for m in self._models])
        return "LinearRank Ensamble ("+models+")"

    def prepare_data(self, df):
        from random import choice
        X = []
        Y = []
        items = df.item.unique()
        self._prepare_feature_extraction(df)

        for _, tuple in df.iterrows():
            x, y = self._extract_features(tuple['user'], tuple['item'], relevant=True)
            bad_item = choice(items)
            x2, y2 = self._extract_features(tuple['user'], bad_item, relevant=False)
            X.append([a-b for a,b in zip(x,x2)])
            Y.append(1)
            X.append([b-a for a,b in zip(x,x2)])
            Y.append(-1)

        return X, Y

if __name__ == '__main__':
    import doctest
    doctest.testmod()
