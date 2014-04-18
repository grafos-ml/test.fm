"""
An interface for the model classes. It should provide automation for the score calculating
"""
cimport cython
import numpy as np
cimport numpy as np
ctypedef double dtype_t

cdef extern from "cblas.h":
    double cblas_ddot(int N, double *X, int incX, double *Y, int incY) nogil

cdef class IMatrixFactorization(object):
    """
    This Model assumes that the score is the matrix produt of the user row in the users matrix and the item row in
    the item matrix. The user and item matrixes will be stored in a Python list self.factors. The first matrix is for
    users and the second is for items. The both matrixes must be a numpy.array with shape (obj, factors)
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _get_score(self, int user, int n_users, int item, int n_items) nogil:
        # Multiply this vectors with blas
        return cblas_ddot(self._number_of_factors, &self.users[user-1], n_users, &self.items[item-1], n_items)

    def get_score(self, user, item):
        self._number_of_factors = <int>self.number_of_factors
        self.users = <double *>(<np.ndarray[double, ndim=2, mode="c"]>self.factors[0]).data
        self.items = <double *>(<np.ndarray[double, ndim=2, mode="c"]>self.factors[1]).data
        #for i in range(len(self.factors[0])):
        #    print len(self.factors[1]), i, self.items[i*self._number_of_factors], self.factors[1][i]
        #print(self.factors[0].shape[0])
        return self._get_score(self.data_map["user"][user], self.factors[0].shape[0], self.data_map["item"][item],
                               self.factors[1].shape[0])