"""
An interface for the model classes. It should provide automation for the score calculating
"""
cimport cython
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
from testfm.models.cutil.float_matrix cimport float_matrix, _float_matrix, fm_get, fm_set, fm_destroy
import pandas as pd


cdef class IModel:
    """
    Interface class for model
    """

    data_map = {}
    def __init__(self):
        self.data_map = {}


    @classmethod
    def param_details(cls):
        """
        Return a dictionary with the parameters for the set parameters and
        a tuple with min, max, step and default value.

        {
            'paramA': (min, max, step, default),
            'paramB': ...
            ...
        }
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        Set the parameters in the model.

        kwargs can have an arbitrary set of parameters
        """
        raise NotImplementedError

    @staticmethod
    def get_user_column():
        """
        Get the name of the user column in the pandas.DataFrame
        """
        return "user"

    @staticmethod
    def get_item_column():
        """
        Get the name of the item column in the pandas.DataFrame
        """
        return "item"

    @staticmethod
    def get_rating_column():
        """
        Get the name of the rating column in the pandas.DataFrame
        """
        return "rating"

    @staticmethod
    def get_context_columns():
        """
        Get a list of names of all the context column names for this model
        :return:
        """
        return []

    @classmethod
    def get_name(cls):
        """
        Get the informative name for the model.
        :return:
        """
        return cls.__name__

    def train(self, training_data):
        """
        Train the model with numpy array. The first column is for users, the second for item and the third for rating.

        :param training_data: A numpy array
        """
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, training_data):
        """
        Train the data from a pandas.DataFrame
        :param training_data: DataFrame a frame with columns 'user', 'item', 'contexts'..., 'rating'
        If rating don't exist it will be populated with 1 for all entries
        """
        columns = [self.get_user_column(), self.get_item_column()] + self.get_context_columns()
        data = []
        self.data_map = {}
        for column in columns:
            unique_data = training_data[column].unique()
            self.data_map[column] = pd.Series(xrange(len(unique_data)), unique_data)
            data.append(map(lambda x: self.data_map[column][x], training_data[column].values))
        data.append(training_data.get(self.get_rating_column(), np.ones((len(training_data),)).tolist()))
        self.train(np.array(data).transpose())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def users_size(self):
        """
        Return the number of users
        """
        return len(self.data_map[self.get_user_column()])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def items_size(self):
        """
        Return the number of items
        """
        return len(self.data_map[self.get_item_column()])

    def get_score(self, user, item, **context):
        """
        Return the score for user, item and evenctually a set of contexts
        """
        raise NotImplemented

    def number_of_context(self):
        """
        Return the number of factors
        """
        return 0

    #def __reduce__(self):
        #d = {}
        #d["c_factor"] = NULL
        #d["c_number_of_context"] = self.c_number_of_context
        #d["c_number_of_factors"] = self.c_number_of_factors
        #return IModel, (), d
    #    return IModel, (), {"data_map": self.data_map}

    #def __setstate__(self, d):
        #d["c_number_of_context"] = self.c_number_of_context
        #d["c_number_of_factors"] = self.c_number_of_factors
        #self.c_factors = self.get_factors()
        #self.c_factors = <float_matrix *>d["c_factor"]
        #self.c_number_of_context = d["c_number_of_context"]
        #self.c_number_of_factors = d["c_number_of_factors"]
    #    self.data_map = d["data_map"]

cdef class NOGILModel(IModel):
    """
    No gil interface. Implements a nogil get_score and nogil item_score
    """

    cdef float nogil_get_score(self, int user, int item, int extra_context, int *context) nogil:
        """
        Get the score without python GIL convention
        """
        cdef float result = 0.
        return result


cdef class IFactorModel(NOGILModel):
    """
    This Model assumes that the score is the matrix product of the user row in the users matrix and the item row in
    the item matrix. The user and item matrices will be stored in a Python list self.factors. The first matrix is for
    users and the second is for items. The both matrices must be a numpy.array with shape (obj, factors)
    """


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, n_factors=20, *args, **kwargs):
        """
        C-stage of the initiation of this model. It put c_factors to null
        """
        self.c_factors = NULL
        self.c_number_of_context = 0
        self.c_number_of_factors = n_factors

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __dealloc__(self):
        """
        Free the memory holding the c_factors
        """
        self.dealloc_factors()

    def fit(self, training_data):
        """
        Train the data from a pandas.DataFrame
        :param training_data: DataFrame a frame with columns 'user', 'item', 'contexts'..., 'rating'
        If rating don't exist it will be populated with 1 for all entries
        """
        self.dealloc_factors()
        super(IFactorModel, self).fit(training_data)
        self.c_number_of_contexts = 2+len(self.get_context_columns())
        self.c_factors = self.get_factors()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float_matrix *get_factors(self):
        """
        Put factor matrix data in c float_matrix
        :return:
        """
        cdef int i
        cdef float_matrix *factor_matrices = <float_matrix *>malloc(sizeof(float_matrix) * self.c_number_of_contexts)
        if factor_matrices is NULL:
            raise MemoryError()
        for i in range(self.c_number_of_contexts):
            factor_matrices[i] = <float_matrix>malloc(sizeof(_float_matrix))
            if factor_matrices[i] is NULL:
                raise MemoryError()
            factor_matrices[i].values = <float *>(<np.ndarray[float, ndim=2, mode="c"]>self.factors[i]).data
            factor_matrices[i].rows, factor_matrices[i].columns = self.factors[i].shape
            factor_matrices[i].size = factor_matrices[i].rows * factor_matrices[i].columns
            factor_matrices[i].transpose = \
                0 if self.factors[i].strides[0] > self.factors[i].strides[len(self.factors[i].strides)-1] else 1
        return factor_matrices

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void dealloc_factors(self) nogil:
        """
        Dealloc factors
        """
        cdef int i
        if self.c_factors is not NULL:
            for i in range(self.c_number_of_contexts):
                if self.c_factors[i] is not NULL:
                    self.c_factors[i].values = NULL
                    fm_destroy(self.c_factors[i])
            free(self.c_factors)
            self.c_factors = NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float nogil_get_score(IFactorModel self, int user, int item, int extra_context, int *context) nogil:
        """
        Get the score without python GIL convention
        """
        cdef int i, j
        cdef float factor, total = 0.
        if self.c_factors == NULL:
            with gil:
                self.c_number_of_factors = self.dimensions
                self.c_number_of_contexts = 2+len(self.get_context_columns())
                self.c_factors = self.get_factors()
        for i in xrange(self.c_number_of_factors):
            factor = fm_get(self.c_factors[0], user, i)
            factor *= fm_get(self.c_factors[1], item, i)
            for j in xrange(extra_context):
                #factor *= factor_matrices[j].values[context[j*2+1]*i + context[j*2]]
                if 0 <= context[j] < self.c_factors[j+2].rows:
                    factor *= fm_get(self.c_factors[j+2], context[j], i)
            total += factor
        return total


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_score(self, user, item, **context):
        cdef int c_user = self.data_map[self.get_user_column()][user], \
            c_item = self.data_map[self.get_item_column()][item], extra_context = len(context)
        cdef float result
        cdef int *c_context
        try:
            c_context = <int *>malloc(sizeof(int) * len(self.get_context_columns()))
            if c_context is NULL:
                raise MemoryError()
            for i, c in enumerate(self.get_context_columns()):
                c_context[i] = self.data_map[c][context[c]] if c in context else -1
            with nogil:
                result = self.nogil_get_score(c_user, c_item, extra_context, c_context)
            return result
        finally:
            #if factor_matrices is not NULL:
            #    for i in range(number_of_contexts):
            #        if factor_matrices[i] is not NULL:
            #            factor_matrices[i].values = NULL
            #            free(factor_matrices[i])
            #    free(factor_matrices)
            if c_context is not NULL:
                free(c_context)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.overflowcheck(False)
    @cython.cdivision(False)
    def get_not_mapped_recommendation(self, user, **context):
        """
        Return the recommendation as a numpy array
        """
        #cdef int c_user = user
        #cdef float_matrix rec = <float_matrix>malloc(sizeof(_float_matrix)), users, items
        #np_rec = np.zeros((len(self.data_map[self.get_item_column()]),), dtype=np.float32)

        #rec.rows = 1
        #rec.columns = len(self.data_map[self.get_item_column()])
        #rec.values = <float *>(<np.ndarray[float, ndim=2, mode="c"]>np_rec).data
        #if self.c_factors == NULL:
        #    self.c_number_of_factors = self.dimensions
        #    self.c_number_of_contexts = 2+len(self.get_context_columns())
        #    self.c_factors = self.get_factors()
        #users, items = self.c_factors[0], self.c_factors[1]
        #fm_static_multiply_row(users, c_user, items, rec)
        #return np_rec
        users, items = self.factors
        return np.squeeze(np.asarray(np.dot(users[user], items.transpose())))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.overflowcheck(False)
    @cython.cdivision(False)
    def get_recommendation(self, user, **context):
        return self.get_not_mapped_recommendation(self.data_map[self.get_user_column()][user], **context)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_user_factors(self, int user, list factors):
        cdef int i
        cdef float f
        for i, f in enumerate(factors):
            fm_set(self.c_factors[0], user, i, f)

    def set_params(self, n_factors, *args, **kwargs):
        """
        Set the parameters for the TensorCoFi
        """
        self.c_number_of_factors = n_factors
