"""
cdef class FloatMatrix:

    cdef float *values
    cdef int rows
    cdef int columns
    cdef int size
    cdef short transpose

    cdef void _multiply_scalar(self, float scalar, FloatMatrix result) nogil
    cdef void _multiply(self, FloatMatrix other, FloatMatrix result) nogil
    cdef void _add(self, FloatMatrix other, FloatMatrix result) nogil
    cdef void _multiply_element_wise(self, FloatMatrix other, FloatMatrix result) nogil
    cdef float get(self, int row, int column) nogil
    cdef void set(self, int row, int column, float value) nogil
    cdef void _transpose(self, FloatMatrix fm) nogil
    cdef FloatMatrix clone(self)
"""

ctypedef public struct _float_matrix:
    float *values
    int rows
    int columns
    int size
    int transpose

ctypedef public _float_matrix *float_matrix

cdef class FloatMatrix:

    cdef float_matrix matrix