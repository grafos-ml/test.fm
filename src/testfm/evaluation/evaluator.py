# -*- coding: utf-8 -*-
"""
Created on 23 January 2014

Evaluator for the test.fm framework

.. moduleauthor:: Linas
"""
__author__ = 'linas'

from random import sample, shuffle

from pandas import DataFrame
from multiprocessing import Pool
from itertools import izip, repeat
from math import sqrt

from testfm.evaluation.measures import Measure, MAPMeasure
from testfm.models.interface import ModelInterface
from testfm.models.baseline_model import IdModel

def pm(args):
    """
    Helper for the threading/multiprocess system
    """
    return args[0].partial_measure(*args[1:])


class Evaluator(object):
    """
    Takes the model,testing data and evaluation measure and spits out the score.
    """
    def evaluate_model(self, factor_model, testing_data,
                       measures=[MAPMeasure()], all_items=None,
                       non_relevant_count=100):
        """
        Evaluate the model using some testing data in pandas.DataFrame

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
        return self.evaluate_model_multiprocessing(factor_model, testing_data,
                                                   measures=measures,
                                                   all_items=all_items,
                                                   non_relevant_count=
                                                   non_relevant_count)


    def evaluate_model_rmse(self, model, testing_data):
        '''
        This is just a hack to evaluate RMSE. Nobody should bother with RMSE anymore, so no good support for it.
        '''
        sum = 0.0
        for idx, row in testing_data.iterrows():
            p = model.getScore(row['user'], row['item'])
            sum += (p - float(row['rating'])) ** 2
        return sqrt(sum/len(testing_data))


    def evaluate_model_threads(self, factor_model, testing_data, measures=
        [MAPMeasure()],all_items=None, non_relevant_count=100):
        """
        Evaluates the model by the following algorithm:
            1. for each user:
                2. take all relevant items from the testing_data
                3. inject #non_relevant random items
                4. predict the score for each item
                5. sort according to the predicted score
                6. evaluate according to each measure
            7.average the scores for each user
        >>> mapm = MAPMeasure()
        >>> model = IdModel()
        >>> evaluation = Evaluator()
        >>> df = DataFrame({'user' : [1, 1, 3, 4], 'item' : [1, 2, 3, 4], \
            'rating' : [5,3,2,1], 'date': [11,12,13,14]})
        >>> len(evaluation.evaluate_model_threads(model, df, \
        non_relevant_count=2))
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

        from concurrent.futures import ThreadPoolExecutor

        # Change to assertions. In production run with the -O option on python
        # to skipp this (Zen Python)
        assert isinstance(factor_model, ModelInterface), \
            "Factor model should be an instance of ModelInterface"

        assert isinstance(testing_data, DataFrame), \
            "Testing data should be a pandas.DataFrame"

        for column in ['item','user']:
            assert column in testing_data.columns, \
                "Testing data should be a pandas.DataFrame with " \
                "'{}' column".format(column)
        for m in measures:
            assert isinstance(m, Measure), \
                "Measures should contain only Measure instances"
        #######################

        if all_items is None:
            all_items = testing_data.item.unique()

        #1. for each user:
        grouped = testing_data.groupby('user')
        with ThreadPoolExecutor(max_workers=4) as e:
            jobs = (e.submit(pm, (Evaluator, user, entries, factor_model,
                                  all_items, non_relevant_count, measures))
                    for user, entries in grouped)
            #7.average the scores for each user
            results = [job.result() for job in jobs]
        return [sum(result)/len(result) for result in zip(*results)]

    def evaluate_model_multiprocessing(self, factor_model, testing_data,
                                       measures=[MAPMeasure()], all_items=None,
                                       non_relevant_count=100):
        """
        Evaluates the model by the following algorithm:
            1. for each user:
                2. take all relevant items from the testing_data
                3. inject #non_relevant random items
                4. predict the score for each item
                5. sort according to the predicted score
                6. evaluate according to each measure
            7.average the scores for each user
        >>> mapm = MAPMeasure()
        >>> model = IdModel()
        >>> evaluation = Evaluator()
        >>> df = DataFrame({'user' : [1, 1, 3, 4], 'item' : [1, 2, 3, 4], \
        'rating': [5,3,2,1], 'date': [11,12,13,14]})
        >>> a = evaluation.evaluate_model_multiprocessing(model, \
        df, non_relevant_count=2)
        >>> print len(a)
        1


        #not the best tests, I need to put seed in order to get an expected \
            behaviour

        :param factor_model: ModelInterface  an instance of ModelInterface
        :param testing_data: DataFrame pandas.DataFrame of testing data
        :param measures: list of measure we want to compute (instances of)
        :param all_items: list of items available in the data set (used for
            negative sampling).
         If set to None, then testing items are used for this
        :param non_relevant_count: int number of non relevant items to add to
            the list for performance evaluation
        :return: list of score corresponding to measures
        """

        # Change to assertions. In production run with the -O option on python
        # to skipp this (Zen Python)
        assert isinstance(factor_model, ModelInterface), \
            "Factor model should be an instance of ModelInterface"

        assert isinstance(testing_data, DataFrame), \
            "Testing data should be a pandas.DataFrame"

        for column in ['item','user']:
            assert column in testing_data.columns, \
                "Testing data should be a pandas.DataFrame with " \
                "'{}' column".format(column)
        for m in measures:
            assert isinstance(m, Measure), \
                "Measures should contain only Measure instances"
        #######################

        if all_items is None:
            all_items = testing_data.item.unique()

        #1. for each user:
        grouped = testing_data.groupby('user')

        pool = Pool()
        u, e = zip(*[(user, entries) for user, entries in grouped])
        res = pool.map(pm, izip(repeat(Evaluator), u, e,
                                repeat(factor_model), repeat(all_items),
                                repeat(non_relevant_count), repeat(measures)))

        #7.average the scores for each user
        ret = [sum(measure_list)/len(measure_list)
               for measure_list in zip(*res)]
        pool.close()
        pool.join()
        return ret

    @classmethod
    def partial_measure(cls, user, entries, factor_model, all_items,
                        non_relevant_count, measures):
        #2. take all relevant items from the testing_data
        ranked_list = [(True, factor_model.getScore(user, i))
                       for i in entries['item']]

        #3. inject #non_relevant random items
        ranked_list += [(False, factor_model.getScore(user,nr))
                        for nr in sample(all_items, non_relevant_count)]

        shuffle(ranked_list)  # Just to make sure we don't introduce any bias

        #5. sort according to the score
        ranked_list.sort(key=lambda x: x[1], reverse=True)

        #6. evaluate according to each measure
        return [measure.measure(ranked_list) for measure in measures]

if __name__ == "__main__":
    import doctest
    doctest.testmod()