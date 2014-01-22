__author__ = 'linas'

import pandas as pd
import testfm
from testfm.evaluation.evaluator import Evaluator
from testfm.models.baseline_model import Popularity, RandomModel

df = pd.read_csv('../data/movielenshead.dat', sep="::", header=None, names=['user', 'item', 'rating', 'date'])
print df.head()

training, testing = testfm.split.holdoutByRandom(df, 0.8)

rand = RandomModel()
pop = Popularity()
pop.fit(training)

evaluator = Evaluator()

items = training.item.unique()

print "Pop       \t\t", evaluator.evaluate_model(pop, testing)
print "Random    \t\t", evaluator.evaluate_model(rand, testing)
print "Pop tri   \t\t", evaluator.evaluate_model(pop, testing, all_items=items)
print "Random tri\t\t", evaluator.evaluate_model(rand, testing, all_items=items)
