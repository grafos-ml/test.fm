
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from testfm.models.cutil.float_matrix cimport float_matrix, fm_create_diagonal, fm_new, fm_new_init, fm_create_random, \
    fm_get, fm_set, fm_destroy, fm_transpose, fm_multiply, fm_static_element_wise_multiply, fm_static_multiply_column, \
    fm_static_multiply, fm_static_multiply_scalar, fm_static_add, fm_solve

from testfm.models.cutil.int_array cimport *
from testfm.models.cutil.interface import IFactorModel
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double log(double n) nogil
    double fabs(double score) nogil
    double copysign(double x, float y) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
cdef api float_matrix *tensorcofi_train(float_matrix data_array, int n_factors, int n_iterations, float c_lambda,
                                        float c_alpha, int n_dimensions, int *dimensions) nogil:
    """
    Train a set of float_matrices with tensor values for a set of contexts
    :param data_array:
    :return:
    """
    cdef int i, j, k, iteration, index, data_row, current_dimension, matrix_index, data_entry, data_column
    cdef int *ipiv = <int *>malloc(sizeof(int) * n_factors)
    cdef float weight, score
    cdef int_array data_row_list = NULL

    # Initialize standard variables
    cdef float_matrix solution = NULL
    cdef float_matrix tmp = fm_new_init(n_factors, 1, 1.), tmp_transpose = NULL  # Temporary matrix (for each context)
    cdef float_matrix regularizer = fm_create_diagonal(n_factors, c_lambda)  # Regularizer is a lambda diagonal
    cdef float_matrix matrix_vector_product = fm_new_init(n_factors, 1, 0.)  # Matrix vector product
    cdef float_matrix one = fm_create_diagonal(n_factors, 1.), one_tmp = NULL # Matrix one
    cdef float_matrix invertible = fm_new_init(n_factors, n_factors, 0.), \
         invertible_tmp = fm_new(n_factors, n_factors) # Invertible Matrix
    cdef float_matrix base = NULL, base_transpose = NULL, base_tmp = NULL
    cdef float_matrix *factors = <float_matrix *>malloc(sizeof(float_matrix) * n_dimensions)  # Factors
    cdef int_array **tensor = <int_array **>malloc(sizeof(int_array *) * n_dimensions)  # Tensor (array)
    for i in range(n_dimensions):  # Fill the tensor with information
        factors[i] = fm_create_random(n_factors, dimensions[i])
        tensor[i] = <int_array *>malloc(sizeof(int_array) * dimensions[i])

        for j in range(dimensions[i]):
            tensor[i][j] = ia_new()
        for data_row in range(data_array.rows):
            ia_add(tensor[i][<int>fm_get(data_array, data_row, i)], data_row)  # Populate tensor
    # Tensor created
    # Factors created
    # Start Iteration ##################################################################################################

    for iteration in range(n_iterations):
        for current_dimension in range(n_dimensions):
            fm_destroy(base)
            # Initiate base
            if n_dimensions == 2:
                base_transpose = fm_transpose(factors[1-current_dimension])  # New memory was allocated
                base = fm_multiply(factors[1-current_dimension], base_transpose)  # New memory was allocated
                fm_destroy(base_transpose)  # Memory from base_transpose released
            else:
                base = fm_new_init(n_factors, n_factors, 1.)  # New memory was allocated
                for matrix_index in range(n_dimensions):
                    if matrix_index != current_dimension:
                        base_transpose =  fm_transpose(factors[matrix_index])  # New memory was allocated
                        base_tmp = fm_multiply(factors[matrix_index], base_transpose)  # New memory was allocated
                        base = fm_static_element_wise_multiply(base, base_tmp, base)  # No new memory is allocated
                        fm_destroy(base_transpose)  # Memory from base_transpose released
                        fm_destroy(base_tmp)  # Memory from base_tmp released
            # Base created

            for data_entry in range(dimensions[current_dimension]):
                data_row_list = tensor[current_dimension][data_entry]
                for i in range(ia_size(data_row_list)):
                    data_row = data_row_list.values[i]
                    # Initialize temporary matrix
                    #  No new memory is allocated
                    for j in range(tmp.size):
                        tmp.values[j] = 1.
                    # Done
                    for data_column in range(n_dimensions):
                        if data_column != current_dimension:
                            fm_static_multiply_column(tmp, factors[data_column],
                                                      <int>fm_get(data_array, data_row, data_column),
                                                      tmp)  # No new memory is allocated
                    score = fm_get(data_array, data_row, data_array.columns-1)
                    weight = c_lambda * log(1.+fabs(score))
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
                fm_static_multiply_scalar(regularizer, 1. / dimensions[current_dimension],
                                          regularizer)  # No new memory allocated
                fm_static_add(invertible, regularizer, invertible)  # No new memory allocated

                solution = fm_solve(invertible, one, ipiv)  # New memory allocated

                one_tmp = fm_multiply(solution, matrix_vector_product)  # New memory allocated
                for k in range(n_factors):
                    fm_set(factors[current_dimension], k, data_entry, fm_get(one_tmp, k, 0))
                fm_destroy(one_tmp)
                fm_destroy(solution)

                # Reset variables
                fm_static_multiply_scalar(one, 0., one)
                fm_static_multiply_scalar(regularizer, 0., regularizer)
                for k in range(one.rows):
                    fm_set(one, k, k, 1.)
                    fm_set(regularizer, k, k, c_lambda)
                fm_static_multiply_scalar(invertible, 0., invertible)
                fm_static_multiply_scalar(matrix_vector_product, 0., matrix_vector_product)
                # End reset

    # Stop Iteration ###################################################################################################
    # Destroy the tensor and variables
    for i in range(n_dimensions):
        #printf(">>> %d\n", i)
        for j in range(dimensions[i]):
            if tensor[i][j] is not NULL:
                #printf(">>> %d/%d\n", tensor[i][j]._size, tensor[i][j]._max_size)
                ia_destroy(tensor[i][j])  # Destroy every int_array
    for i in range(n_dimensions):
        free(tensor[i])  # Destroy the array of int_array
    free(tensor)  # If I continue with the explanation it will be hilarious
    free(ipiv)  # Free ipiv
    fm_destroy(tmp)  # Free tmp
    fm_destroy(regularizer)
    fm_destroy(matrix_vector_product)
    fm_destroy(one)
    fm_destroy(invertible)
    fm_destroy(invertible_tmp)
    fm_destroy(base)
    # Return the factors
    return factors


class CTensorCoFi(IFactorModel):

    number_of_factors = 20
    number_of_iterations = 5
    constant_lambda = .05
    constant_alpha = 40
    context_columns = []

    def __init__(self, n_factors=None, n_iterations=None, c_lambda=None, c_alpha=None, other_context=None):
        """
        Constructor

        :param n_factors: Number of factors to the matrices
        :param n_iterations: Number of iteration in the matrices construction
        :param c_lambda: I came back when I find it out
        :param c_alpha: Constant important in weight calculation
        """
        self.set_params(n_factors, n_iterations, c_lambda, c_alpha)
        self.factors = []
        self.context_columns = other_context or []

    @classmethod
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

    def get_context_columns(self):
        """
        Get a list of names of all the context column names for this model
        :return:
        """
        return self.context_columns


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.overflowcheck(False)
    @cython.cdivision(False)
    def train(self, data):
        """
        Train the model
        """
        cdef float_matrix *tensor = NULL
        cdef int i, j, row, column, number_of_factors = <int>self.number_of_factors, \
            number_of_iterations = <int>self.number_of_iterations
        cdef float constant_lambda = <float>self.constant_lambda, constant_alpha = <float>self.constant_alpha
        cdef float_matrix fm_data = NULL
        cdef int *dimensions = NULL
        cdef int number_of_dimensions = <int>len(self.data_map)
        try:
            fm_data = fm_new(data.shape[0], data.shape[1])
            if fm_data is NULL:
                raise MemoryError()
            dimensions = <int *>malloc(sizeof(int) * number_of_dimensions)
            if dimensions is NULL:
                raise MemoryError()
            for i in range(fm_data.size):
                row = i / fm_data.columns
                column = i % fm_data.columns
                fm_set(fm_data, row, column, <float>data[row, column])
            dimensions[0] = <int>self.users_size()
            dimensions[1] = <int>self.items_size()

            for i, in range(len(self.get_context_columns())):
                dimensions[i+2] = <int>len(self.data_map[self.get_context_columns()[i]])
            with nogil:
                tensor = tensorcofi_train(fm_data, number_of_factors, number_of_iterations, constant_lambda,
                                          constant_alpha, number_of_dimensions, dimensions)
            if tensor is NULL:
                raise RuntimeError
            for i in range(len(self.data_map)):
                tmp = np.empty(tensor[i].size, dtype=np.float32)
                for j in range(tensor[i].size):
                    tmp[j] = tensor[i].values[j]
                tmp.shape = tensor[i].rows, tensor[i].columns
                self.factors.append(tmp.transpose())
        finally:
            if dimensions is not NULL:
                free(dimensions)
            fm_destroy(fm_data)
            if tensor is not NULL:
                for i in range(number_of_dimensions):
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

    def set_params(self, int n_factors, int n_iterations, float c_lambda, float c_alpha):
        """
        Set the parameters for the TensorCoFi
        """
        self.number_of_factors = n_factors or 20
        self.number_of_iterations = n_iterations or 5
        self.constant_lambda = c_lambda or .05
        self.constant_alpha = c_alpha or 40.

    def get_name(self):
        return "CTensorCoFi(n_factors=%s, n_iterations=%s, c_lambda=%.2f, c_alpha=%d)" % \
               (self.number_of_factors, self.number_of_iterations, <float>self.constant_lambda, <int>self.constant_alpha)

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    #def get_score(self, user, item, **context):
    #    ret = self.factors[0][self.data_map[self.get_user_column()][user], :]
    #    ret = np.multiply(ret, self.factors[1][self.data_map[self.get_item_column()][item], :])
    #    for i, name in enumerate(self.get_context_columns(), start=2):
    #        ret = np.multiply(ret, self.factors[i][self.data_map[name][context[name]], :])
    #    return sum(ret)
