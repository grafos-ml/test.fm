# -*- coding: utf-8 -*-
"""
Created on 17 April 2014

Evaluator for the test.fm framework version 2

.. moduleauthor:: joaonrb
"""
__author__ = "joaonrb"


from testfm.evaluation.measures import MAPMeasure
from random import sample


def partial_measure(user, entries, factor_model, all_items, non_relevant_count, measure, k=None):
    if non_relevant_count is None:
        # Add all items except relevant
        ranked_list = [(False, factor_model.get_score(user, nr)) for nr in all_items if nr not in entries['item']]
        # Add relevant items
        ranked_list += [(True, factor_model.get_score(user, r)) for r in entries['item']]
    else:
        #2. inject #non_relevant random items
        ranked_list = [(False, factor_model.get_score(user, nr)) for nr in sample(all_items, non_relevant_count)]
        #2. add all relevant items from the testing_data
        ranked_list += [(True, factor_model.get_score(user, i)) for i in entries['item']]

        #shuffle(ranked_list)  # Just to make sure we don't introduce any bias (AK: do we need this?)

    #number of relevant items
    n = entries['item'].size
    #5. sort according to the score
    ranked_list.sort(key=lambda x: x[1], reverse=True)

    #6. evaluate according to each measure
    return measure.measure(ranked_list[:k], n=n)


class Evaluator(object):
    """
    Takes the model, testing data and evaluation measure and spits out the score.
    """

    measures = [MAPMeasure()]

    def __init__(self, measures=None):
        """
        Constructor
        :param measures: list of measure we want to compute
        """
        self.measures = measures or self.measures

    def evaluate_model(self, factor_model, testing_data, all_items=None, non_relevant_count=100):
        """
        Evaluate the model using some testing data in pandas.DataFrame

        Evaluates the model by the following algorithm:
            1. for each user:
                2. take all relevant items from the testing_data
                3. inject #non_relevant random items
                4. predict the score for each item
                5. sort according to the predicted score
                6. evaluate according to each measure
            7.average the scores for each user
        >>> import pandas as pd
        >>> from testfm.models.baseline_model import IdModel
        >>> model = IdModel()
        >>> evaluation = Evaluator()
        >>> df = pd.DataFrame({"user" : [1, 1, 3, 4], "item" : [1, 2, 3, 4], "rating" : [5,3,2,1], \
        "date": [11,12,13,14]})
        >>> len(evaluation.evaluate_model(model, df, non_relevant_count=2))
        1

        #not the best tests, I need to put seed in order to get an expected behaviour

        :param factor_model: ModelInterface  an instance of ModelInterface
        :param all_items: list of items available in the data set (used for
            negative sampling).
         If set to None, then testing items are used for this
        :param non_relevant_count: int number of non relevant items to add to
            the list for performance evaluation
        :return: list of score corresponding to measures
        """
        ret = []
        if all_items is None:
            all_items = testing_data.item.unique()

        #1. for each user:
        for m in self.measures:
            scores = []
            for user, entries in testing_data.groupby('user'):
                pm = partial_measure(user, entries, factor_model, all_items, non_relevant_count, m)
                scores.append(pm)
            #7.average the scores for each user
            ret.append(sum(scores)/len(scores))
        return ret


class MultiThreadingEvaluator(Evaluator):
    """
    A try to multi-thread version for the Evaluator
    """

    def evaluate_model(self, factor_model, testing_data, all_items=None, non_relevant_count=100):
        """
        An implementation of simple Evaluator with Python threading

        >>> import pandas as pd
        >>> from testfm.models.baseline_model import IdModel
        >>> model = IdModel()
        >>> evaluation = MultiThreadingEvaluator()
        >>> df = pd.DataFrame({"user" : [1, 1, 3, 4], "item" : [1, 2, 3, 4], "rating" : [5,3,2,1], \
        "date": [11,12,13,14]})
        >>> len(evaluation.evaluate_model(model, df, non_relevant_count=2))
        1

        :param factor_model: ModelInterface  an instance of ModelInterface
        :param all_items: list of items available in the data set (used for
            negative sampling).
         If set to None, then testing items are used for this
        :param non_relevant_count: int number of non relevant items to add to
            the list for performance evaluation
        :return: list of score corresponding to measures
        """
