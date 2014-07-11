# -*- coding: utf-8 -*-
"""
Created on 23 January 2014

Evaluator for the test.fm framework

.. moduleauthor:: Linas
"""
__author__ = 'linas'

from random import sample
from math import sqrt
from testfm.evaluation.cutil.measures import MAPMeasure
from testfm.models.cutil.interface import IFactorModel
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from multiprocessing import cpu_count
from testfm.models.cutil.interface import NOGILModel
from testfm.evaluation.cutil.evaluator import evaluate_model


def partial_measure(user, entries, factor_model, all_items, non_relevant_count, measure, k=None):
    #if isinstance(factor_model, IFactorModel):
    #    return factor_model.partial_measure(user, entries, all_items, non_relevant_count, measure)
    if non_relevant_count is None:
        # Add all items except relevant
        ranked_list = [(False, factor_model.get_score(user, nr)) for nr in all_items if nr not in entries['item']]
    else:
        #2. inject #non_relevant random items
        nr_items = [i for i in all_items if i not in entries['item']]
        ranked_list = [(False, factor_model.get_score(user, nr))
                       for nr in sample(nr_items, non_relevant_count \
                if len(nr_items) > non_relevant_count else len(nr_items))]
    #2. add all relevant items from the testing_data
    ranked_list += [(True, factor_model.get_score(user, i)) for i in entries['item']]

        #shuffle(ranked_list)  # Just to make sure we don't introduce any bias (AK: do we need this?)

    #number of relevant items
    n = entries['item'].size
    #5. sort according to the score
    ranked_list.sort(key=lambda x: x[1], reverse=True)

    #6. evaluate according to each measure
    if k is None:
        return {measure.name: measure.measure(ranked_list, n=n)}
    else:
        return {measure.name: measure.measure(ranked_list[:k], n=n)}


class Evaluator(object):
    """
    Takes the model,testing data and evaluation measure and spits out the score.
    """

    def __init__(self, use_multi_threading=True):
        self.use_muilti = use_multi_threading

    def evaluate_model(self, factor_model, testing_data, measures=None, all_items=None,
                       non_relevant_count=100, k=None):
        """
        Evaluate the model using some testing data in pandas.DataFrame. The Evaluator check if the model in evaluation
        is able to be executed with multi-threading. If so it executes a low level routine using C-Threads otherwise
        execute a single thread routine.

        :param factor_model: An instance that Should implement IModel
        :param measures: List of measure we want to compute. They should implement IMeasure. Default: MAPMeasure
        :param all_items: List of items available in the data set (used for negative sampling). If set to None, only
            testing items will be used.

        :param non_relevant_count: int number of non relevant items to add to the list for performance evaluation
        :return: List of score corresponding to measures
        """
        measures = measures or [MAPMeasure()]

        #all_items = all_items or testing_dataframe.item.unique()
        if all_items is None:
            all_items = testing_data.item.unique()
        #1. for each user:
        grouped = testing_data.groupby('user')

        if self.use_muilti and isinstance(factor_model, NOGILModel):
            return [e/len(grouped) for e in evaluate_model(factor_model, testing_data, measures, all_items, non_relevant_count, k)]
        #return self.evaluate_model_multiprocessing(factor_model, testing_data, measures=measures, all_items=all_items,
        #                                           non_relevant_count=non_relevant_count, k=k)
        # compute
        results = [partial_measure(user, entries, factor_model, all_items, non_relevant_count, m, k) \
                   for user, entries in grouped for m in measures]
        #print [v["MAPMeasure"] for v in results]
        partial_measures = sum((Counter(r) for r in results), Counter())
        #7.average the scores for each user
        return [partial_measures[measure.name]/len(grouped) for measure in measures]

    def evaluate_model_rmse(self, model, testing_data):
        """
        This is just a hack to evaluate RMSE. Nobody should bother with RMSE anymore, so no good support for it.
        """
        sum = 0.0
        for idx, row in testing_data.iterrows():
            p = model.get_score(row['user'], row['item'])
            sum += (p - float(row['rating'])) ** 2
        return sqrt(sum/len(testing_data))

if __name__ == "__main__":
    import doctest
    doctest.testmod()