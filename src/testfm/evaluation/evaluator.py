# -*- coding: utf-8 -*-
"""
Created on 23 January 2014

Evaluator for the test.fm framework

.. moduleauthor:: Linas
"""
__author__ = 'linas'

from random import sample
from pandas import DataFrame
from itertools import izip, repeat
from math import sqrt
from testfm.evaluation.measures import MAPMeasure
from testfm.models.baseline_model import IdModel

from multiprocessing import Process, Queue, current_process, cpu_count


#
# Function run by worker processes
#
def worker(input, output):
    for func, args in iter(input.get, "STOP"):
        result = func(*args)
        output.put(result)

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
    if k is None:
        return measure.measure(ranked_list, n = n)
    else:
        return measure.measure(ranked_list[:k], n = n)


class EvaluatorPool(object):
    """
    The thread / process poll for the evaluator
    """

    def __init__(self, workers=None):
        self.workers = workers or cpu_count()

        # Create queues
        self._jobs = 0
        self._task_queue = Queue()
        self._done_queue = Queue()
        # Start worker processes
        for i in range(self.workers):
            Process(target=worker, args=(self._task_queue, self._done_queue)).start()

    def put(self, function, *args):
        """
        Create a Job with function(*args, **kwargs)

        :param function: Function to be executed
        :type function: callable
        :param args: Tuple of parameters
        """
        self._task_queue.put((function, args))
        self._jobs += 1

    def __next__(self):
        if self._jobs == 0:
            raise StopIteration
        result = self._done_queue.get()
        self._jobs -= 1
        return result

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __del__(self):
        for i in range(self.workers):
            self._task_queue.put("STOP")


POOL = EvaluatorPool()


def pm(args):
    """
    Helper for the threading/multiprocess system
    """
    return partial_measure(*args[1:])


class Evaluator(object):
    """
    Takes the model,testing data and evaluation measure and spits out the score.
    """
    def evaluate_model(self, factor_model, testing_data, measures=[MAPMeasure()], all_items=None,
                       non_relevant_count=100, k=None):
        """
        Evaluate the model using some testing data in pandas.DataFrame

        :param factor_model: ModelInterface  an instance of ModelInterface
        :param measures: list of measure we want to compute (instnces of)
        :param all_items: list of items available in the data set (used for
            negative sampling).
         If set to None, then testing items are used for this
        :param non_relevant_count: int number of non relevant items to add to
            the list for performance evaluation
        :return: list of score corresponding to measures
        """
        return self.evaluate_model_multiprocessing(factor_model, testing_data, measures=measures, all_items=all_items,
                                                   non_relevant_count=non_relevant_count, k=k)

    def evaluate_model_rmse(self, model, testing_data):
        """
        This is just a hack to evaluate RMSE. Nobody should bother with RMSE anymore, so no good support for it.
        """
        sum = 0.0
        for idx, row in testing_data.iterrows():
            p = model.get_score(row['user'], row['item'])
            sum += (p - float(row['rating'])) ** 2
        return sqrt(sum/len(testing_data))

    def evaluate_simple(self, factor_model, testing_data, measures=[MAPMeasure()], all_items=None,
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
            'rating' : [5,3,2,1], 'date': [11,12,13,14]})
        >>> len(evaluation.evaluate_simple(model, df, \
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
        ret = []
        from concurrent.futures import ThreadPoolExecutor
        if all_items is None:
            all_items = testing_data.item.unique()

        #1. for each user:
        for m in measures:
            scores = []
            for user, entries in testing_data.groupby('user'):
                pm = partial_measure(user, entries, factor_model, all_items, non_relevant_count, m)
                scores.append(pm)
            #7.average the scores for each user
            ret.append(sum(scores)/len(scores))
        return ret

    def evaluate_model_multiprocessing(self, factor_model, testing_data, measures=[MAPMeasure()], all_items=None,
                                       non_relevant_count=100, k=None):
        """
        Evaluates the model by the following algorithm:
            1. for each user:
                2. take all relevant items from the testing_data
                3. inject #non_relevant random items
                4. predict the score for each item
                5. sort according to the predicted score
                6. evaluate according to each measure
            7.average the scores for each user
        # >>> mapm = MAPMeasure()
        # >>> model = IdModel()
        # >>> evaluation = Evaluator()
        # >>> df = DataFrame({'user' : [1, 1, 3, 4], 'item' : [1, 2, 3, 4], \
        # 'rating': [5,3,2,1], 'date': [11,12,13,14]})
        # >>> a = evaluation.evaluate_model_multiprocessing(model, \
        # df, non_relevant_count=2)
        # >>> evaluation.close()
        # >>> print len(a)
        # 1

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
        ret =[]
        if all_items is None:
            all_items = testing_data.item.unique()

        #1. for each user:
        grouped = testing_data.groupby('user')

        u, e = zip(*[(user, entries) for user, entries in grouped])
        for m in measures:
            for job in izip(u, e, repeat(factor_model), repeat(all_items), repeat(non_relevant_count), repeat(m),
                            repeat(k)):
                POOL.put(partial_measure, *job)
            scores = [job_result for job_result in POOL]
            #7.average the scores for each user
            ret.append(sum(scores)/len(scores))
        return ret

    def close(self):
        del POOL

if __name__ == "__main__":
    import doctest
    doctest.testmod()