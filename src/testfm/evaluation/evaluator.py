__author__ = 'linas'

from random import sample

from pandas import DataFrame

from testfm.evaluation.meassures import Measure, MAP_measure
from testfm.models.interface import ModelInterface
from testfm.models.baseline_model import IdModel
'''
Takes the model,testing data and evaluation measure and spits out the score.
'''

class Evaluator(object):

    def evaluate_model(self, factor_model, testing_dataframe, measures=
        [MAP_measure()],all_items=None, non_relevant_count=100):
        """
        Evaluates the model by the following algorithm:
            1. for each user:
                2. take all relevant items from the testing_dataframe
                3. inject #non_relevant random items
                4. predict the score for each item
                5. sort according to the predicted score
                6. evaluate according to each measure
            7.average the scores for each user
        >>> mapm = MAP_measure()
        >>> model = IdModel()
        >>> evaluation = Evaluator()
        >>> df = DataFrame({'user' : [1, 1, 3, 4], 'item' : [1, 2, 3, 4], \
            'rating' : [5,3,2,1], 'date': [11,12,13,14]})
        >>> len(evaluation.evaluate_model(model, df, non_relevant_count=2))
        1

        #not the best tests, I need to put seed in order to get an expected \
            behaviour

        :param factor_model: ModelInterface  an instance of ModelInterface
        :param testing_dataframe: DataFrame pandas dataframe of testing data
        :param measures: list of measure we want to compute (instnces of)
        :param all_items: list of items available in the data set (used for
            negative sampling).
         If set to None, then testing items are used for this
        :param non_relevant_count: int number of non relevant items to add to
            the list for performance evaluation
        :return: list of score corresponding to measures
        """


        # Don't need all this in production. We should assume the method work
        # work well if is used well. (Zen Python)
        if not isinstance(factor_model, ModelInterface):
            raise ValueError("Factor model should be an instance of "
                "ModelInterface")

        if not isinstance(testing_dataframe, DataFrame):
            raise ValueError("Testing data should be a pandas dataframe")

        if not 'item' in testing_dataframe.columns:
            raise ValueError("Testing data should be a pandas dataframe with "
                "'item' column")
        if not 'user' in testing_dataframe.columns:
            raise ValueError("Testing data should be a pandas dataframe with "
                "'user' column")

        for m in measures:
            if not isinstance(m, Measure):
                raise ValueError("Measures should contain only Measure "
                    "instances")
        #######################

        partial_measures = {}  # a temp dictionary to store sums of measures we
                               # compute

        #all_items = all_items or testing_dataframe.item.unique()
        if all_items is None:
            all_items = testing_dataframe.item.unique()

        #1. for each user:
        grouped = testing_dataframe.groupby('user')
        for user, entries in grouped:

            #2. take all relevant items from the testing_dataframe
            ranked_list = [(True, factor_model.getScore(user,i))
                for i in entries['item']]

            #3. inject #non_relevant random items
            ranked_list += [(False, factor_model.getScore(user,nr))
                for nr in sample(all_items, non_relevant_count)]

            #5. sort according to the score
            ranked_list.sort(key=lambda x: x[1], reverse=True)

            #6. evaluate according to each measure
            for measure in measures:
                try:
                    partial_measures[measure] += measure.measure(ranked_list)
                except KeyError:
                    partial_measures[measure] = measure.measure(ranked_list)

        #7.average the scores for each user
        return [partial_measures[measure]/len(grouped) for measure in measures]


if __name__ == "__main__":
    import doctest
    doctest.testmod()