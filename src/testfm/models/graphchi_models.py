__author__ = 'linas'

import os
import logging
import subprocess
import tempfile
import datetime
import numpy as np
from interface import ModelInterface
logger = logging.getLogger(__name__)
from scipy.io.mmio import mminfo, mmread, mmwrite

class GraphchiBase(ModelInterface):

    def __init__(self, tmp_dir="/tmp"):
        self.tmp_dir = tmp_dir

    def getScore(self, user, item):
        uid = self.umap[user]
        iid = self.imap[item]
        pred = self.global_mean + self.U_bias[uid] + self.V_bias[iid] + np.dot(self.U[uid], self.V[iid])
        return float(pred)

    def fit(self, training_data):
        '''
        executes something on the lines
        svdpp --training=smallnetflix_mm
        '''

        training_filename = self.dump_data(training_data)
        logger.debug("Started training model {}".format(__name__))
        cmd = " ".join(["svdpp",
                        "--training={}".format(training_filename),
                        "--biassgd_lambda=1e-4",
                        "--biassgd_gamma=1e-4",
                        "--minval=1",
                        "--maxval=5",
                        "--max_iter=6",
                        "--quiet=1"])
        self.execute_command(cmd)

        self.global_mean = self.read_matrix(training_filename+"_global_mean.mm")
        self.U = self.read_matrix(training_filename+"_U.mm")
        self.V = self.read_matrix(training_filename+"_V.mm")
        self.U_bias = self.read_matrix(training_filename+"_U_bias.mm")
        self.V_bias = self.read_matrix(training_filename+"_V_bias.mm")

    def dump_data(self, df):
        #first we need to reindex user and item and store their new ids
        _,df['u'] = np.unique(df.user, return_inverse=True)
        _,df['i'] = np.unique(df.item, return_inverse=True)

        self.umap = {
            key[0]: key[1]+1 for key, _ in df.groupby(['user', 'u'])
        }

        self.imap = {
            key[0]: key[1]+1 for key, _ in df.groupby(['item', 'i'])
        }

        filename = tempfile.mkstemp(prefix='graphchi', dir=self.tmp_dir, suffix=".mtx")
        f = os.fdopen(filename[0], "w")
        print filename[1]
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("% Generated {}\n".format(datetime.datetime.now()))
        f.write("{} {} {}\n".format(len(df.user.unique()), len(df.item.unique()), len(df)))
        for idx, row in df.iterrows():
            f.write("{} {} {}\n".format(row['u']+1,row['i']+1,row['rating']))
        f.close()
        return filename[1]

    def execute_command(self, cmd):
        if not os.environ['GRAPHCHI_ROOT']:
            raise EnvironmentError("Please set GRAPHCHI_ROOT")
        print cmd
        sub = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = sub.communicate()

    def read_matrix(self, filename):
        logger.debug("Loading matrix market matrix ")
        f = open(filename, "rb")
        m = mmread(f)
        f.close()
        return m
