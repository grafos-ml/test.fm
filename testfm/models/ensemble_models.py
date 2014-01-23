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

if __name__ == '__main__':
    import doctest
    doctest.testmod()
