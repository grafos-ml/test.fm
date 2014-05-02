cimport cython
from libc.stdlib cimport malloc, free, realloc
from testfm.models.cutil.float_matrix import *
from testfm.models.cutil.float_matrix cimport *
from testfm.models.cutil.int_array import *
from testfm.models.cutil.int_array cimport *
from testfm.models.cutil.interface import IFactorModel
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    float log(float n) nogil
    float fabs(float score) nogil
    float copysign(float x, float y)

cdef extern from "cblas.h":
    void cblas_strsm(int order, int side, int uplo, int transa, int diag, int m, int n, float alpha, float *a, int ida,
                     float *b, int idb)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api float_matrix *tensorcofi_train(float_matrix data_array, int n_factors, int n_iterations, float c_lambda,
                                        float c_alpha, int n_dimensions, int *dimensions) nogil:
    """
    Train a set of float_matrices with tensor values for a set of contexts
    :param data_array:
    :return:
    """
    cdef int i, j, iteration, index, data_row, current_dimension, matrix_index, data_entry, data_column
    cdef float weight, score
    cdef int_array data_row_list

    # Initialize standard variables
    cdef float_matrix tmp = fm_new_init(n_factors, 1, 1.), tmp_transpose  # Temporary matrix (for each context)
    cdef float_matrix regularizer = fm_create_diagonal(n_factors, c_lambda)  # Regularizer is a lambda diagonal
    cdef float_matrix matrix_vector_product = fm_new_init(n_factors, 1, 0.)  # Matrix vector product
    cdef float_matrix one = fm_create_diagonal(n_factors, 1.), one_tmp  # Matrix one
    cdef float_matrix invertible = fm_new_init(n_factors, n_factors, 1.), \
         invertible_tmp = fm_new(n_factors, n_factors) # Invertible Matrix
    cdef float_matrix base, base_transpose, base_tmp
    cdef float_matrix *factors = <float_matrix *>malloc(sizeof(float_matrix) * n_dimensions)  # Factors
    cdef int_array **tensor = <int_array *>malloc(sizeof(int_array *) * n_dimensions)  # Tensor (array)
    for i in xrange(n_dimensions):  # Fill the tensor with information
        factors[i] = fm_create_random(n_factors, dimensions[i])
        tensor[i] = <int_array>malloc(sizeof(_int_array) * dimensions[i])
        for j in xrange(dimensions[i]):
            tensor[i][j] = ia_new()
        for data_row in xrange(data_array.rows):
            ia_add(tensor[i][fm_get(data_array, data_row, i)], data_row)  # Populate tensor
    # Tensor created
    # Factors created
    # Start Iteration ##################################################################################################

    for iteration in xrange(n_iterations):
        for current_dimension in xrange(n_dimensions):
            # Initiate base
            if n_dimensions == 2:
                base_transpose = fm_transpose(factors[1-current_dimension])  # New memory was allocated
                base = fm_multiply(factors[1-current_dimension], base_transpose)  # New memory was allocated
                fm_destroy(base_transpose)  # Memory from base_transpose released
            else:
                base = fm_new_init(n_factors, n_factors, 1.)  # New memory was allocated
                for matrix_index in xrange(n_dimensions):
                    if matrix_index != current_dimension:
                        base_transpose =  fm_transpose(factors[matrix_index])  # New memory was allocated
                        base_tmp = fm_multiply(factors[matrix_index], base_transpose)  # New memory was allocated
                        base = fm_static_element_wise_multiply(base, base_tmp, base)  # No new memory is allocated
                        fm_destroy(base_transpose)  # Memory from base_transpose released
                        fm_destroy(base_tmp)  # Memory from base_tmp released
            # Base created

            for data_entry in xrange(dimensions[current_dimension]):
                data_row_list = tensor[current_dimension][data_entry]
                for i in xrange(data_row_list.size):
                    # Initialize temporary matrix
                    #  No new memory is allocated
                    for j in xrange(tmp.size):
                        tmp.values[j] = 1.
                    # Done
                    for data_column in xrange(n_dimensions):
                        if data_column != current_dimension:
                            fm_static_multiply_column(tmp, factors[data_column],
                                                      fm_get(data_array, data_row, data_column),
                                                      tmp)  # No new memory is allocated
                        score = fm_get(data_array, data_row, data_array.columns-1)
                        weight = c_lambda * log(1.+abs(score))
                        # Start calculation of rank one update in invertible
                        tmp_transpose = fm_transpose(tmp)  # Memory allocated
                        fm_static_multiply(tmp, tmp_transpose, invertible_tmp)  # No new memory allocated
                        fm_static_multiply_scalar(invertible_tmp, weight, invertible_tmp)  # No new memory allocated
                        fm_static_add(invertible, invertible_tmp, invertible)  # No new memory allocated
                        fm_destroy(tmp_transpose)  # Free memory from tmp_transpose
                        # End calculation of rank one update in invertible

                        # Start calculate matrix vector product
                        fm_static_multiply_scalar(tmp, copysign(score, 1.) * (1.+weight), tmp)  # No new memory allocated
                        fm_static_add(matrix_vector_product, tmp, matrix_vector_product)  # No new memory allocated
                        # End calculate matrix vector product
                    fm_static_add(invertible, base, invertible)  # No new memory allocated
                    fm_static_element_wise_division(regularizer, dimensions[current_dimension],
                                                    regularizer)  # No new memory allocated
                    fm_static_add(invertible, regularizer, invertible)  # No new memory allocated
                    cblas_strsm(101, 141, 121, 112 if invertible.transpose else 111, 131, one.rows, one.column, 1.,
                                invertible.values, invertible.columns if invertible.transpose else invertible.rows,
                                one.values, one.columns if one.transpose else one.rows)  # Calculate the solution on one
                    one_tmp = fm_multiply(invertible, matrix_vector_product)  # New memory allocated
                    for i in xrange(dimensions[current_dimension]):
                        fm_set(factors[current_dimension], i, data_entry, fm_get(one_tmp, i, 0))
                    fm_destroy(one_tmp)

                    # Reset variables
                    fm_static_multiply_scalar(invertible, 0., invertible)
                    fm_static_multiply_scalar(matrix_vector_product, 0., matrix_index)
                    # End reset

    # Stop Iteration ###################################################################################################
    # Destroy the tensor and variables
    for i in xrange(n_dimensions):
        for j in xrange(dimensions[i]):
            ia_destroy(tensor[i][j])  # Destroy every int_array
        free(tensor[i])  # Destroy the array of int_array
    free(tensor)  # If I continue with the explanation it will be hilarious
    fm_destroy(tmp)  # Free tmp
    fm_destroy(regularizer)
    fm_destroy(matrix_vector_product)
    fm_destroy(one)
    fm_destroy(invertible)
    fm_destroy(invertible_tmp)
    fm_destroy()
    # Return the factors
    return factors


cdef class CTensorCoFi(IFactorModel):

    cdef int number_of_factors = 20
    cdef int number_of_iterations = 5
    cdef float constant_lambda = .05
    cdef float constant_alpha = 40
    cdef list context_columns = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(CTensorCoFi self, int n_factors=None, int n_iterations=None, float c_lambda=None, float c_alpha=None,
                 list other_context=None):
        """
        Constructor

        :param n_factors: Number of factors to the matrices
        :param n_iterations: Number of iteration in the matrices construction
        :param c_lambda: I came back when I find it out
        :param c_alpha: Constant important in weight calculation
        """
        self.set_params(n_factors, n_iterations, c_lambda, c_alpha)
        self.factors = []
        self.context_columns = other_context or self.context_columns

    @classmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def param_details(cls):
        """
        Return parameter details for n_factors, n_iterations, c_lambda and c_alpha
        """
        return {
            "n_factors": (10, 20, 2, 20),
            "n_iterations": (1, 10, 2, 5),
            "c_lambda": (.1, 1., .1, .05),
            "c_alpha": (30, 50, 5, 40)
        }

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_context_columns(CTensorCoFi self):
        """
        Get a list of names of all the context column names for this model
        :return:
        """
        return self.context_columns

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def train(CTensorCoFi self, data):
        """
        Train the model
        """
        cdef float_matrix *tensor
        cdef int i, j, row, column
        cdef float_matrix fm_data
        cdef int *dimensions
        try:
            fm_data = fm_new(data.shape[0], data.shape[1])
            if dimensions is NULL:
                raise MemoryError()
            dimensions = <int *>malloc(sizeof(int) * <int>len(self.data_map))
            if dimensions is NULL:
                raise MemoryError()
            for i in xrange(fm_data):
                row = i / fm_data.rows
                column = i % fm_data.rows
                fm_set(fm_data, row, column, data[row, column])
            for i in xrange(len(self.data_map)):
                dimensions[i] = <int>len(self.data_map)
            tensor = tensorcofi_train(fm_data, self.number_of_factors, self.number_of_iterations, self.constant_lambda,
                                      self.constant_alpha, <int>len(self.data_map))
            for i in xrange(len(self.data_map)):
                tmp = [[] for _ in xrange(tensor[i].rows)]
                for j in xrange(tensor[i].size):
                    row = j / tensor[i].rows
                    column = j % tensor[i].rows
                    tmp[row].append(fm_get(tensor[i], row, column))
                self.factors.append(np.array(tmp))
        finally:
            if dimensions is not NULL:
                free(dimensions)
            fm_destroy(fm_data)
            if tensor is not NULL:
                for i in xrange(len(self.data_map)):
                    fm_destroy(tensor[i])
                free(tensor)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_model(self):
        return self.factors

    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def online_user_factors(matrix_y, user_item_ids, p_param=10, lambda_param=0.01):
        """
        :param matrix_y: application matrix Y.shape = (#apps, #factors)
        :param user_item_ids: the rows that correspond to installed applications in Y matrix
        :param p_param: p parameter
        :param lambda_param: regularizer
        """
        y = matrix_y[user_item_ids]
        base1 = matrix_y.transpose().dot(matrix_y)
        base2 = y.transpose().dot(np.diag([p_param - 1] * y.shape[0])).dot(y)
        base = base1 + base2 + np.diag([lambda_param] * base1.shape[0])
        u_factors = np.linalg.inv(base).dot(y.transpose()).dot(np.diag([p_param] * y.shape[0]))
        u_factors = u_factors.dot(np.ones(y.shape[0]).transpose())
        return u_factors

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_params(CTensorCoFi self, int n_factors, int n_iterations, float c_lambda, float c_alpha):
        """
        Set the parameters for the TensorCoFi
        """
        self.number_of_factors = n_factors or self.number_of_factors
        self.number_of_iterations = n_iterations or self.number_of_iterations
        self.constant_lambda = c_lambda or self.constant_lambda
        self.constant_alpha = c_alpha or self.constant_alpha

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_name(CTensorCoFi self):
        return "TensorCoFi(n_factors=%s, n_iterations=%s, c_lambda=%s, c_alpha=%s)" % \
               (self.number_of_factors, self.number_of_iterations, self.constant_lambda, self.constant_alpha)