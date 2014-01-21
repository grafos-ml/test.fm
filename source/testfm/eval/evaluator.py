__author__ = 'linas'

from random import sample

from meassures import Measure, MAP_measure
from testfm.models.interface import ModelInterface
from testfm.models.fake_model import IdModel
from pandas import DataFrame
'''
Takes the model, testing data and evaluation meassure and spits out the score.
'''

class Evaluator(object):

    def evaluate_model(self, factor_model, testing_dataframe, measures=[MAP_measure()], non_relevant=100):
        """
        Evaluates the model by the following algorithm:
            1. for each user:
                2. take all relevant items from the testing_dataframe
                3. inject #non_relevant random items
                4. predict the score for each
                5. sort according to the score
                6. evaluate according to each measure
            7.average the scores for each user
        >>> mapm = MAP_measure()
        >>> model = IdModel()
        >>> eval = Evaluator()
        >>> df = DataFrame({'user' : [1, 1, 3, 4], 'item' : [1, 2, 3, 4], 'rating' : [5,3,2,1], 'date': [11,12,13,14]})
        >>> len(eval.evaluate_model(model, df, non_relevant=2))
        1

        #not the best test, I need to put seed in order to get an expected behaviour

        :param factor_model: ModelInterface  an instance of ModelInterface
        :param testing_dataframe: DataFrame pandas dataframe of testing data
        :param measures: list a list of measure we want to compute (instnces of)
        :return: list of score corresponding to measures
        """

        if not isinstance(factor_model, ModelInterface):
            raise ValueError("Factor model should be an instance of ModelInterface")
        if not isinstance(testing_dataframe, DataFrame):
            raise ValueError("Testing data should be a pandas dataframe")
        for m in measures:
            if not isinstance(m, Measure):
                raise ValueError("Measures should contain only Measure instances")

        partial_measures = {}#a temp dictionary to store partial measures

        items = testing_dataframe.item.unique()

        #1. for each user:
        grouped = testing_dataframe.groupby('user')
        for user, entries in grouped:
            #2. take all relevant items from the testing_dataframe
            ranked_list = []
            for i,r in zip(entries['item'], entries['rating']):
                prediction = factor_model.getScore(user,i)
                ranked_list.append((True, prediction))

            #3. inject #non_relevant random items
            for nr in sample(items, non_relevant):
                prediction = factor_model.getScore(user,nr)
                ranked_list.append((False, prediction))

            #5. sort according to the score
            ranked_list.sort(key=lambda x: x[1], reverse=True)

            #6. evaluate according to each measure
            for measure in measures:
                pm = partial_measures.get(measure, 0.0)
                pm += measure.measure(ranked_list)
                partial_measures[measure] = pm

        #7.average the scores for each user
        return [partial_measures[measure]/len(grouped) for measure in measures]


if __name__ == "__main__":
    import doctest
    doctest.testmod()