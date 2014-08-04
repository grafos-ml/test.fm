__author__ = 'linas'
'''
Shows how to execute a remote model building.
'''
import testfm
import pandas as pd
from testfm.evaluation.evaluator import Evaluator
from testfm.models.baseline_model import Popularity, RandomModel
from testfm.okapi.connector import PopularityOkapi, BPROkapi
from pkg_resources import resource_filename

# Prepare the data
df = pd.read_csv(resource_filename(testfm.__name__, 'data/movielenshead.dat'), sep="::", header=None,
                 names=['user', 'item', 'rating', 'date', 'title'])
print df.head()

#########################################
################ WARNING ################
#########################################
# This example don't work. Only for     #
# specific environments.                #
# Thank you.                            #
#########################################

# Tell me what models we want to evaluate
models = [
    RandomModel(),
    PopularityOkapi(hadoop_source="/data/b.ajf/hadoop1_env.sh",
                    host="igraph-01",
                    okapi_jar_dir="okapi/jar/",
                    #host='54.72.18.118', user='hadoop',
                    #okapi_jar_dir='/Users/linas/devel/okapi/target/',
                    #okapi_jar_base_name='okapi-0.3.2-SNAPSHOT-jar-with-dependencies.jar',
                    #public_key_path='/Users/linas/.ssh/hack-okapi.pem'
                    ),
    Popularity(normalize=False),
    BPROkapi(hadoop_source="/data/b.ajf/hadoop1_env.sh",
             host="igraph-01",
             okapi_jar_dir="okapi/jar/",
             #host='54.72.18.118', user='hadoop',
             #okapi_jar_dir='/Users/linas/devel/okapi/target/',
             # #okapi_jar_base_name='okapi-0.3.2-SNAPSHOT-jar-with-dependencies.jar',
             #public_key_path='/Users/linas/.ssh/hack-okapi.pem'
             )
]

# Setup the environment
evaluator = Evaluator()

for m in models:
    m.fit(df)
    print m.get_name().ljust(50),
    print evaluator.evaluate_model(m, df)
