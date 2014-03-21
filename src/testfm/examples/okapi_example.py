__author__ = 'linas'
'''
Shows how to execute remote model building.
'''
import testfm
import pandas as pd
from testfm.evaluation.evaluator import Evaluator
from testfm.okapi.connector import PopularityOkapi
from pkg_resources import resource_filename

#prepare the data
df = pd.read_csv(resource_filename(testfm.__name__,'data/movielenshead.dat'),
    sep="::", header=None, names=['user', 'item', 'rating', 'date', 'title'])
print df.head()

print len(df.item.unique())
print df.item.unique()

#tell me what models we want to evaluate
models = [  PopularityOkapi(),
        ]

#setup the environment
from fabric.api import env
env.host_string = 'linas@igraph-01'


eval = Evaluator()

for m in models:
    m.fit(df)
    print m.getName().ljust(50),
    print eval.evaluate_model(m, df)

eval.close()#need this call to clean up the worker processes
