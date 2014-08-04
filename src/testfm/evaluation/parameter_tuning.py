#! -*- encoding=utf-8 -*-
"""
Parameter tuning based on the Gaussian Processes.
Idea based on http://arxiv.org/pdf/1206.2944.pdf
Because it is more natural for me, I will use UCB method.

0. Estimate method performance at 2 points
1. Run GP and estimate mean and variance of each parameter at many points
2. Try the new point where mean+variance is highest. We use 99 percentile to merge these two.
3. Iterate until the new point is the same as you already tried (this means you found maximum with high confidence)
"""

__author__ = "linas"

import numpy as np
import scipy.stats as st
from sklearn.gaussian_process import GaussianProcess
from testfm.evaluation.evaluator import Evaluator


class ParameterTuning(object):
    """
    ###
        model = TensorCoFi()
        params = parameterTuening.getbestparams(model, training, testing)
        for p in params
            model.set(p)
        model.fit()
    """

    __z_score = st.norm.ppf(.9)
    __max_iterations = 100

    @classmethod
    def set_z_value(cls, percentage):
        """
        Set a new z value based in percentage.

        :param percentage: Float between 0 and 100
        """
        cls.__z_score = st.norm.ppf(percentage / 100.)

    @classmethod
    def set_max_iterations(cls, new_max):
        """
        Set the number of max iterations to newMax
        """
        cls.__max_iterations = new_max

    @staticmethod
    def tune(model, training, testing, non_relevant_count=100, **kwargs):
        """
        Return a mean for the predictive power
        """
        model.set_params(**kwargs)
        model.fit(training)
        evaluator = Evaluator()
        # Return the MAPMeasure in position 0
        measure = evaluator.evaluate_model(model, testing, non_relevant_count=non_relevant_count)[0]
        print "tried {} = {}".format(kwargs, measure)
        return measure

    @staticmethod
    def get_best_params(model, training, testing, non_relevant_count=100, **kwargs):
        """
        Search for the best set of parameters in the model

        use ParameterTuning().getBestParameters(model,parA=(0,10,0.1,3)...)
        (min,max,step,default)
        """
        # Create a grid of parameters
        kwargs = kwargs or model.param_details()
        grid = zip(*(x.flat for x in np.mgrid[[slice(*row[:3]) for row in kwargs.values()]]))
        m_instance = model()
        values = {k: ParameterTuning.tune(m_instance, training, testing, non_relevant_count,
                                          **dict(zip(kwargs.keys()[:2], k)))
                  for k in zip(*(v[:2] for v in kwargs.values()))}

        gp = GaussianProcess(theta0=.1, thetaL=.001, thetaU=5.)

        # To make it reasonable we limit the number of iterations
        for i in xrange(0, ParameterTuning.__max_iterations):

            # Get a list of parameters and the correspondent result
            param, response = zip(*values.items())

            # Fit the GaussianProcess model with the parameters and results
            gp.fit(np.array(param), np.array(response).T)

            # Get prediction
            y_predicted, mse = gp.predict(grid, eval_MSE=True)

            # get upper confidence interval. 2.576 z-score corresponds to 99th
            # percentile
            ucb_u = y_predicted + np.sqrt(mse) * ParameterTuning.__z_score

            next_list = zip(ucb_u, grid)
            next_list.sort(reverse=True)
            new_x = next_list[0][1]

            if new_x not in values:
                values[new_x] = ParameterTuning.tune(m_instance, training, testing, 
                                                     **{k: v for k, v in zip(kwargs, new_x)})
            else:
                break
        sv = sorted(values.items(), cmp=lambda x, y: cmp(y[1], x[1]))
        assert sv[0][1] > sv[-1][1], "Sorted from lowest to highest"
        return {k: v for k, v in zip(kwargs, sv[0][0])}