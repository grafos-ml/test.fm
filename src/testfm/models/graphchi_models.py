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

class SVDpp(ModelInterface):

    def __init__(self, tmp_dir="/tmp"):
        self.tmp_dir = tmp_dir
        params = {k: v[2] for k,v in self.paramDetails().items()}
        self.setParams(**params)

    def getScore(self, user, item):
        uid = self.umap[user]
        iid = self.imap[item]
        u = np.add(self.U[uid, :10], self.U[uid, 10:])
        pred = self.global_mean + self.U_bias[uid] + self.V_bias[iid] + np.dot(u, self.V[iid])
        return float(pred)


    def setParams(self, nIter=5, lamb=0.05, gamma=0.01):
        """
        Set the parameters for the TensorCoFi
        """
        self._nIter = nIter
        self._lamb = lamb
        self._gamm = gamma


    @classmethod
    def paramDetails(cls):
        """
        Return parameter details for parameters
        """
        return {
            #sorry but authors of svdpp do not provide a way to set dimensionality 'dim': (2, 100, 2, 5),
            'nIter': (1, 20, 2, 5),
            'lamb': (.1, 1., .1, .05),
            'gamma': (0.001, 1.0, 0.1, 0.01)
        }

    def fit(self, training_data):
        '''
        executes something on the lines
        svdpp --training=smallnetflix_mm
        '''

        training_filename = self.dump_data(training_data)
        logger.debug("Started training model {}".format(__name__))
        cmd = " ".join(["svdpp",
                        "--training={} ".format(training_filename),
                        "--biassgd_lambda={}".format(self._lamb),
                        "--biassgd_gamma={}".format(self._gamm),
                        "--minval=1 ",
                        "--maxval=5 ",
                        "--max_iter={}".format(self._nIter),
                        "--quiet=1 "])
        logger.debug(cmd)
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
            key[0]: key[1] for key, _ in df.groupby(['user', 'u'])
        }

        self.imap = {
            key[0]: key[1] for key, _ in df.groupby(['item', 'i'])
        }

        filename = tempfile.mkstemp(prefix='graphchi', dir=self.tmp_dir, suffix=".mtx")
        f = os.fdopen(filename[0], "w")
        #print filename[1]
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("% Generated {}\n".format(datetime.datetime.now()))
        f.write("{} {} {}\n".format(len(df.user.unique()), len(df.item.unique()), len(df)))
        for idx, row in df.iterrows():
            r = int(row['rating'])
            if r == 0:
                r = 1
            f.write("{} {} {}\n".format(row['u']+1,row['i']+1,r))
        f.close()
        return filename[1]

    def execute_command(self, cmd):
        if not os.environ['GRAPHCHI_ROOT']:
            raise EnvironmentError("Please set GRAPHCHI_ROOT")
        #print cmd
        sub = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = sub.communicate()

    def read_matrix(self, filename):
        '''reads grapchi format in the numpy array. The order is wrong in grapchi, therefore, we play tricks.

        About the second (order of rows vs. columns), we have a problem that in distributed graphlab we
        output the matrix by rows, since each node is a row, and nodes are on different machines.
        So to be compatible I left GraphChi code to confirm to GraphLab. In case it helps, here is a matlab mmread.m
        function which switches the order to be sorted by rows http://select.cs.cmu.edu/code/graphlab/mmread.m
        I am not sure if it is easy to fix it in python or not. [...]
        You are definitely right the the standard describes column
        order and not row order.
        '''

        logger.debug("Loading matrix market matrix ")
        f = open(filename, "rb")
        r = mmread(f)
        m = np.array(r.ravel())
        m = m.reshape(r.shape, order='F')
        # print "mmread", r.ravel(), r.shape
        # print "trans", m.ravel(), m.shape
        # if m.shape[0] > 1:
        #     print m[1]
        f.close()
        return m
