"""
An interface for the model classes. It should provide automation for the score calculating
"""
cimport cython
from libc.stdlib cimport malloc, free, realloc, rand, RAND_MAX
import numpy as np
cimport numpy as np
from testfm.models import IModel

cdef extern from "cblas.h":
    double cblas_ddot(int N, double *X, int incX, double *Y, int incY) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float get_score(int number_of_factors, int number_of_contexts, double **factor_matrices, int *context) nogil:
    cdef int i, j
    cdef float factor, total = 0.
    for i in xrange(number_of_factors):
        factor = 1.
        for j in xrange(number_of_contexts):
            factor *= factor_matrices[j][context[j*2+1]*i + context[j*2] - 1]
        total += factor
    return total


class IFactorModel(IModel):
    """
    This Model assumes that the score is the matrix product of the user row in the users matrix and the item row in
    the item matrix. The user and item matrices will be stored in a Python list self.factors. The first matrix is for
    users and the second is for items. The both matrices must be a numpy.array with shape (obj, factors)
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_score(self, user, item, **context):
        user = self.data_map[self.get_user_column()][user], len(self.data_map[self.get_user_column()])
        item = self.data_map[self.get_item_column()][item], len(self.data_map[self.get_item_column()])
        cdef int number_of_contexts = len(self.get_context_columns())+2, i, number_of_factors = self.number_of_factors
        cdef double **factor_matrices = <double **>malloc(sizeof(double) *number_of_contexts)
        cdef float result
        if factor_matrices is NULL:
            raise MemoryError()
        cdef int *c_context = <int *>malloc(sizeof(int) * number_of_contexts * 2)
        if c_context is NULL:
            free(factor_matrices)
            raise MemoryError()
        try:
            p_context = [user, item] + [
                (self.data_map[c][context[c]], len(self.data_map[c])) for c in self.get_context_columns()]
            for i in xrange(number_of_contexts):
                factor_matrices[i] = <double *>(<np.ndarray[double, ndim=2, mode="c"]>self.factors[i]).data
                c_context[i*2], c_context[i*2+1] = p_context[i]

            with nogil:
                result = get_score(number_of_factors, number_of_contexts, factor_matrices, c_context)
            return result
        finally:
            free(c_context)
            free(factor_matrices)