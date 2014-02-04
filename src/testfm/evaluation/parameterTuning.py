__author__ = 'linas'

'''
Parameter tuning based on the Gaussian Processes.
Idea based on http://arxiv.org/pdf/1206.2944.pdf
Because it is more natural for me, I will use UCB method.

0. Estimate method performance at 2 points
1. Run GP and estimate mean and variance of each parameter at many points
2. Try the new point where mean+variance is highest. We use 99 percentile to merge these two.
3. Iterate until the new point is the same as you already tried (this means you found maximum with high confidence)

'''

import numpy as np
import math
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
import testfm

def foo(X):
    """The function to evaluate"""
    ret = []
    for x in X:
        r = 2.*math.sqrt(sum([n**2 for n in x]))
        ret.append(0. if r == 0 else (math.sin(r) / r))
    return ret

def get_init_points(dims):
    '''gives initial 2 points for the system to train on'''
    #p1  = tuple([mi for mi,ma in dims])
    #p2  = tuple([ma for mi,ma in dims])
    #assert [p1, p2] == zip(*dims)
    #return [p1,p2]
    return zip(*dims)

def get_grid(dims, n_bins=10):
    '''This is the full mesh grid and serves as discretization of the parameter search space'''
    slices = np.mgrid[[slice(row[0], row[1], (row[1]-row[0])/n_bins)
        for row in dims.values()]]
    return zip(*(x.flat for x in slices))

def _subspace(i, max_x, X):
    from copy import copy
    ret = []
    for x in X:
        c = list(copy(max_x))
        c[i] = x
        ret.append(c)
    return ret

def plot(dims, max_x, max_y, gp, n_bins=100):
    '''
    Plots the current estimation of parameters in 2d space. It is beautiful and
    not very useful :). You can imagine, that it is impossible to visualize
    parameter behaviour in multi-dimensional space. Therefore, it takes the
    current maximum (max_x), and varies parameter by parameter while others
    stay fixed.
    '''
    #pl.xlabel('Hyper Parameter Value')
    #    pl.ylabel('f(x)')

    fig = pl.figure()
    #fig.suptitle('GP based parameter estimates')
    for i, k in enumerate(dims.keys()):
        X = np.arange(dims[k][0], dims[k][1], (dims[k][1]-dims[k][0])/n_bins)
        ax = fig.add_subplot(len(dims), 1, i+1)
        grid = _subspace(i, max_x, X)

        ax.plot(max_x[i], max_y, 'o', color="red")
        #ax.plot(grid, f(grid), 'r:', label=u'$f(x)')
        y_pred, MSE = gp.predict(grid, eval_MSE=True)
        ax.plot(grid, y_pred, 'b-', label=u'Prediction')
        ax.set_xlabel('Parameter '+k)
        UCB_u = y_pred + np.sqrt(MSE) * 2.576
        UCB_l = y_pred - np.sqrt(MSE) * 2.576
        ax.fill(np.concatenate([grid, grid[::-1]]), np.concatenate(
            [UCB_u,(UCB_l)[::-1]]), alpha=.5, fc='b', ec='None',
            label='99% confidence interval')
    fig.tight_layout()
    pl.show()

def get_next_UCB_point(X, f, depth=0):
    print "trial #{}".format(depth)

    gp = GaussianProcess(theta0=1e-1, thetaL=1e-3, thetaU=5.)
    gp.fit(np.array(X), np.array(f(X)).T)
    y_pred, MSE = gp.predict(grid, eval_MSE=True)
    #get upper confidence interval. 2.576 z-score corresponds to 99th percentile
    UCB_u = y_pred + np.sqrt(MSE) * 2.576
    next_list = zip(UCB_u, grid)
    next_list.sort(reverse=True)
    new_x = next_list[0][1]
    if new_x not in X:
        X.append(new_x)
        return get_next_UCB_point(X, f, depth+1)
    else:
        return (next_list[0][0], next_list[0][1], gp)

if __name__ == "__main__":
    #init the dimensions to the minimum and maximum
    dims = {"u":(-10.0, 10.0), "i":(-10.0, 8.0)}
    grid = get_grid(dims)
    X = get_init_points(dims.values())
    print X
    y,x, gp = get_next_UCB_point(X, foo)
    print "y={} at x={}".format(y, x)
    plot(dims, x, y, gp)

class ParameterTuning(object):
    '''
    ###
        model = TensorCoFi()
        params = parameterTuening.getbestparams(model, training, testing)
        for p in params
            model.set(p)
        model.fit()
    '''
    @staticmethod
    def tune(model,training,testing,**kwargs):
        '''
        Return a mean for the predictive power
        '''
        m = model(**kwargs)
        m.fit(training)

        # Return the MAP_mesure in position 0
        return testfm.evaluate_model(m,testing)[0]


    @staticmethod
    def getBestParams(model,training,testing,**kwargs):
        '''
        Search for the best set of parameters in the model

        use ParameterTuning().getBestParameters(model,parA=(0,10,0.1,3)...)
        (min,max,step,default)
        '''
        # Create a grid of parameters
        grid = zip(*(x.flat for x in np.mgrid[[slice(*row[:3])
            for row in kwargs.values()]]))
        values = {k: ParameterTuning.tune(model,training,testing,**dict(zip(
            kwargs.keys()[:2],k))) for k in zip(*(v[:2]
            for v in kwargs.values()))}

        gp = GaussianProcess(theta0=1e-1, thetaL=1e-3, thetaU=5.)
        while True:
            param, response = zip(*values.items())
            gp.fit(np.array(param), np.array(response).T)
            y_pred, MSE = gp.predict(grid, eval_MSE=True)
            #get upper confidence interval. 2.576 z-score corresponds to 99th
            # percentile
            UCB_u = y_pred + np.sqrt(MSE) * 2.576
            next_list = zip(UCB_u, grid)
            next_list.sort(reverse=True)
            new_x = next_list[0][1]
            if new_x not in values:
                values[new_x] = ParameterTuning.tune(model,training,testing,
                    **dict(zip(kwargs,new_x)))
            else:
                return new_x

            #Precisa de Receber os parametros







