ctypedef public struct _float_matrix:
    float *values
    int rows
    int columns
    int size
    int transpose

ctypedef public _float_matrix *float_matrix

cdef class FloatMatrix:

    cdef float_matrix matrix

cdef api float_matrix fm_new(int rows, int columns) nogil
cdef api float_matrix fm_new_init(int rows, int columns, float fill_value) nogil
cdef api void fm_destroy(float_matrix self) nogil
cdef api float fm_get(float_matrix self, int row, int column) nogil
cdef api void fm_set(float_matrix self, int row, int column, float value) nogil
cdef api float_matrix fm_static_clone(float_matrix self, float_matrix clone_matrix) nogil
cdef api float_matrix fm_clone(float_matrix self) nogil
cdef api float_matrix fm_static_transpose(float_matrix self, float_matrix fm) nogil
cdef api float_matrix fm_transpose(float_matrix self) nogil
cdef api float_matrix fm_static_multiply_scalar(float_matrix self, float scalar, float_matrix fm) nogil
cdef api float_matrix fm_multiply_scalar(float_matrix self, float scalar) nogil
cdef api float_matrix fm_static_multiply(float_matrix self, float_matrix other, float_matrix result) nogil
cdef api float_matrix fm_multiply(float_matrix self, float_matrix other) nogil
cdef api float_matrix fm_static_multiply_column(float_matrix self, float_matrix other, int row,
                                                float_matrix result) nogil
cdef api float_matrix fm_multiply_column(float_matrix self, float_matrix other, int row) nogil
cdef api float_matrix fm_static_add(float_matrix self, float_matrix other, float_matrix result) nogil
cdef api float_matrix fm_add(float_matrix self, float_matrix other) nogil
cdef api float_matrix fm_static_element_wise_multiply(float_matrix self, float_matrix other, float_matrix result) nogil
cdef api float_matrix fm_element_wise_multiply(float_matrix self, float_matrix other) nogil
cdef api float_matrix fm_static_element_wise_division(float_matrix self, float_matrix other, float_matrix result) nogil
cdef api float_matrix fm_element_wise_division(float_matrix self, float_matrix other) nogil
cdef api float_matrix fm_diagonal(float_matrix self, float scalar) nogil
cdef api float_matrix fm_create_diagonal(int dim, float scalar) nogil
cdef api float_matrix fm_create_random(int rows, int columns) nogil
#cdef api void fm_print(float_matrix self) nogil
cdef api float_matrix fm_solve(float_matrix self, float_matrix result, int *ipiv) nogil