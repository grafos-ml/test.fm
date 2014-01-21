__author__ = 'linas'

import pandas as pd
import testfm


df = pd.read_csv('../data/movielenshead.dat', sep="::", header=None, names=['user', 'item', 'rating', 'date'])
print df.head()

training, testing = testfm.split.holdoutByRandom(df, 0.9)
print len(training), len(testing)
