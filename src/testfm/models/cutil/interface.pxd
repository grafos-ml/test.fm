"""
An interface for the model classes. It should provide automation for the score calculating
"""


cdef class IMatrixFactorization(object):

    cdef double *users
    cdef double *items
    cdef int _number_of_factors

    cdef double _get_score(self, int user, int n_users, int item, int n_items) nogil