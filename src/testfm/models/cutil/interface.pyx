"""
An interface for the model classes. It should provide automation for the score calculating
"""
cimport cython
from libc.stdlib cimport malloc, free  #, realloc, rand, RAND_MAX
from libc.stdio cimport printf
import numpy as np
cimport numpy as np
from testfm.models.cutil.float_matrix cimport float_matrix, _float_matrix, fm_get
import pandas as pd
from testfm.evaluation.cutil.measures cimport NOGILMeasure
from random import sample

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float get_score(int number_of_factors, int number_of_contexts, float_matrix *factor_matrices, int *context) nogil:
    cdef int i, j
    cdef float factor, total = 0.
    for i in xrange(number_of_factors):
        factor = 1.
        for j in xrange(number_of_contexts):
            #factor *= factor_matrices[j].values[context[j*2+1]*i + context[j*2]]
            factor *= fm_get(factor_matrices[j], context[j*2], i)
        total += factor
    return total

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float merge_max(float a, float b) nogil:
    return a if a > b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void merge_helper(float *input, int left, int right, float *scratch) nogil:
    #base case: one element
    if right == left + 1:
        return
    cdef int i = 0
    cdef int length = right - left
    cdef int midpoint_distance = length/2
    # l and r are to the positions in the left and right subarrays
    cdef int l = left, r = left + midpoint_distance

    # sort each subarray
    merge_helper(input, left, left + midpoint_distance, scratch)
    merge_helper(input, left + midpoint_distance, right, scratch)

    # merge the arrays together using scratch for temporary storage
    for i in range(length):
        # Check to see if any elements remain in the left array; if so, we check if there are any elements left in
        # the right array; if so, we compare them.  Otherwise, we know that the merge must use take the element
        # from the left array
        if l < left + midpoint_distance and (r == right or merge_max(input[l*2+1], input[r*2+1]) == input[l*2+1]):
            scratch[i*2], scratch[i*2+1] = input[l*2], input[l*2+1]
            l+=1
        else:
            scratch[i*2], scratch[i*2+1] = input[r*2], input[r*2+1]
            r+=1
    # Copy the sorted subarray back to the input
    for i in range(left, right):
        input[i*2], input[i*2+1] = scratch[i*2-left*2], scratch[(i*2-left*2)+1]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int mergesort(float *input, int size) nogil:
    cdef float *scratch = <float *>malloc(size * sizeof(float) * 2)
    if scratch is not NULL:
        merge_helper(input, 0, size, scratch)
        free(scratch);
        return 1
    return 0;

cdef class IModel:
    """
    Interface class for model
    """

    data_map = None

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
    def get_context_columns(self):
        """
        Get a list of names of all the context column names for this model
        :return:
        """
        raise NotImplemented

    def get_name(self):
        """
        Get the informative name for the model.
        :return:
        """
        return self.__class__.__name__

    def train(self, training_data):
        """
        Train the model with numpy array. The first column is for users, the second for item and the third for rating.

        :param training_data: A numpy array
        """
        raise NotImplemented

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

    def users_size(self):
        """
        Return the number of users
        """
        return len(self.data_map[self.get_user_column()])

    def items_size(self):
        """
        Return the number of items
        """
        return len(self.data_map[self.get_item_column()])


cdef class IFactorModel(IModel):
    """
    This Model assumes that the score is the matrix product of the user row in the users matrix and the item row in
    the item matrix. The user and item matrices will be stored in a Python list self.factors. The first matrix is for
    users and the second is for items. The both matrices must be a numpy.array with shape (obj, factors)
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float_matrix *get_factors(self):
        """
        Put factor matrix data in c float_matrix
        :return:
        """
        cdef int i, number_of_contexts = 2+len(self.get_context_columns())
        cdef float_matrix *factor_matrices = <float_matrix *>malloc(sizeof(float_matrix) * number_of_contexts)
        if factor_matrices is NULL:
            raise MemoryError()
        for i in range(number_of_contexts):
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
    cdef void dealloc_factors(self, float_matrix *factors):
        """
        Dealloc factors
        """
        cdef int i, number_of_contexts = 2+len(self.get_context_columns())
        if factors is not NULL:
            for i in range(number_of_contexts):
                if factors[i] is not NULL:
                    factors[i].values = NULL
                    free(factors[i])
                free(factors)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_score(self, user, item, **context):
        cdef int number_of_contexts = len(self.get_context_columns())+2, i, number_of_factors = self.number_of_factors
        cdef float_matrix *factor_matrices
        cdef float result
        cdef int *c_context
        try:
            user = self.data_map[self.get_user_column()][user], len(self.data_map[self.get_user_column()])
            item = self.data_map[self.get_item_column()][item], len(self.data_map[self.get_item_column()])
            factor_matrices = self.get_factors()
            c_context = <int *>malloc(sizeof(int) * number_of_contexts * 2)
            if c_context is NULL:
                raise MemoryError()
            p_context = [user, item] + [
                (self.data_map[c][context[c]], len(self.data_map[c])) for c in self.get_context_columns()]
            for i in range(number_of_contexts):
                c_context[i*2], c_context[i*2+1] = p_context[i]
            with nogil:
                result = get_score(number_of_factors, number_of_contexts, factor_matrices, c_context)
            return result
        finally:
            if factor_matrices is not NULL:
                for i in range(number_of_contexts):
                    if factor_matrices[i] is not NULL:
                        factor_matrices[i].values = NULL
                        free(factor_matrices[i])
                free(factor_matrices)
            if c_context is not NULL:
                free(c_context)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_score0(self, user, item, **context):
        user = self.data_map[self.get_user_column()][user], len(self.data_map[self.get_user_column()])
        item = self.data_map[self.get_item_column()][item], len(self.data_map[self.get_item_column()])
        cdef int number_of_contexts = len(self.get_context_columns())+2, i, number_of_factors = self.number_of_factors
        cdef float_matrix *factor_matrices
        cdef float result
        cdef int *c_context
        try:
            factor_matrices = <float_matrix *>malloc(sizeof(float_matrix) * number_of_contexts)
            if factor_matrices is NULL:
                raise MemoryError()
            c_context = <int *>malloc(sizeof(int) * number_of_contexts * 2)
            if c_context is NULL:
                raise MemoryError()
            p_context = [user, item] + [
                (self.data_map[c][context[c]], len(self.data_map[c])) for c in self.get_context_columns()]
            for i in range(number_of_contexts):
                factor_matrices[i] = <float_matrix>malloc(sizeof(_float_matrix))
                factor_matrices[i].values = <float *>(<np.ndarray[float, ndim=2, mode="c"]>self.factors[i]).data
                factor_matrices[i].rows, factor_matrices[i].columns = self.factors[i].shape
                factor_matrices[i].size = factor_matrices[i].rows * factor_matrices[i].columns
                factor_matrices[i].transpose = \
                    0 if self.factors[i].strides[0] > self.factors[i].strides[len(self.factors[i].strides)-1] else 1
                c_context[i*2], c_context[i*2+1] = p_context[i]

            with nogil:
                result = get_score(number_of_factors, number_of_contexts, factor_matrices, c_context)
            return result
        finally:
            if factor_matrices is not NULL:
                for i in range(number_of_contexts):
                    if factor_matrices[i] is not NULL:
                        factor_matrices[i].values = NULL
                        free(factor_matrices[i])
                free(factor_matrices)
            if c_context is not NULL:
                free(c_context)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def partial_measure(self, user, entries, all_items, non_relevant_count, measure):
        cdef float *ranked_list = NULL
        cdef int i, counter = <int>len(entries['item']), counter0, item
        cdef int number_of_factors = <int>self.number_of_factors
        cdef float_matrix *factor_matrices = NULL
        cdef NOGILMeasure c_measure
        cdef int nogil_flag = 0
        cdef int c_context[5]
        c_context[0], c_context[1], c_context[3] = self.data_map[self.get_user_column()][user], \
                                                   len(self.data_map[self.get_user_column()]), \
                                                   len(self.data_map[self.get_item_column()])
        cdef float result
        if non_relevant_count is None:
            f_ranked_list = [nr for nr in all_items if nr not in entries['item']]
        else:
            f_ranked_list = [nr for nr in sample(all_items, non_relevant_count)]
        counter0 = len(f_ranked_list)
        if isinstance(measure, NOGILMeasure):
            c_measure = <NOGILMeasure>measure
            nogil_flag = 1
        try:
            ranked_list = <float *>malloc(sizeof(float) * (counter+counter0) * 2)
            factor_matrices = self.get_factors()
            for i, item in enumerate(f_ranked_list):
                ranked_list[i*2], ranked_list[i*2+1] = 0., float(self.data_map[self.get_item_column()][item])
            for i, item in enumerate(entries['item'], start=counter0):
                ranked_list[i*2], ranked_list[i*2+1] = 1., float(self.data_map[self.get_item_column()][item])
            with nogil:
                for i in range(counter+counter0):
                    c_context[2] = <int>ranked_list[i*2+1]
                    ranked_list[i*2+1] = get_score(number_of_factors, 2, factor_matrices, c_context)

                if nogil_flag:
                    mergesort(ranked_list, counter+counter0)
                    result = c_measure.nogil_measure(ranked_list, counter+counter0)

            if nogil_flag == 0:
                ranked_list0 = []
                for i in range(counter+counter0):
                    ranked_list0.append((bool(ranked_list[i+2]), ranked_list[i*2+1]))
                #number of relevant items
                n = entries['item'].size
                #5. sort according to the score
                ranked_list0.sort(key=lambda x: x[1], reverse=True)
                result = measure.measure([ranked_list[i] for i in range(counter+counter0)], n=n)
                #6. evaluate according to each measure
            return {measure.name: result}
        finally:
            if ranked_list is not NULL:
                free(ranked_list)
            if factor_matrices is not NULL:
                for i in range(2):
                    if factor_matrices[i] is not NULL:
                        factor_matrices[i].values = NULL
                        free(factor_matrices[i])
                free(factor_matrices)