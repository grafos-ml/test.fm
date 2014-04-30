"""
FloatMatrix implementation for native no GIL matrix operations
"""
cimport cython
from libc.stdlib cimport malloc, free

cdef extern from "cblas.h":
    float cblas_scopy(int n, float *x, int incx, float *y, int incy) nogil  # For clone
    float cblas_sscal(int n, float a, float *x, int incx) nogil  # For multiply vector by real
    void cblas_sgemm(int order, int transa, int transb, int m, int n, int k, float alpha, float *a, int lda,
                     float *b, int ldb, float beta, float *c, int ldc) nogil  # For matrix multiplication

#ctypedef struct _float_matrix:
#    float *values
#    int rows
#    int columns
#    int size
#    short transpose
#
#ctypedef _float_matrix *float_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api float_matrix fm_new(int rows, int columns) nogil:
    """
    C constructor
    :param rows: Number of rows
    :param columns: Number of columns
    """
    cdef float_matrix self = <float_matrix>malloc(sizeof(_float_matrix))
    self.rows = rows
    self.columns = columns
    self.size = rows * columns
    self.transpose = 0
    self.values = <float *>malloc(sizeof(float) * self.size)
    return self

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api float_matrix fm_new_init(int rows, int columns, float fill_value) nogil:
    """
    C Constructor and initialize all values to "fill_value"
    :param rows:
    :param columns:
    :param fill_value:
    :return:
    """
    cdef float_matrix self = fm_new(rows, columns)
    cdef int i
    if fill_value == 0.:
        cblas_sscal(self.size, 0., self.values, 1)
    else:
        for i in xrange(self.size):
            self.values[i] = fill_value
    return self

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api void fm_destroy(float_matrix self) nogil:
    """
    Dealloc the C structures
    """
    if self.values is not NULL:
        free(self.values)
    if self is not NULL:
        free(self)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api float fm_get(float_matrix self, int row, int column) nogil:
    """
    Get a value from the matrix in the cell (row, column)
    :param row: Row index to get the item
    :param column: Column item to get the item
    :return: The float element in the specific cell
    """
    if self.transpose == 1:
        return self.values[column*self.rows + row]
    return self.values[row*self.columns + column]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef api void fm_set(float_matrix self, int row, int column, float value) nogil:
    """
    Set the specific cell to value
    :param row: Row index to get the item
    :param column: Column item to get the item
    :param value: The new value in the cell
    """
    if self.transpose == 1:
        self.values[column*self.rows + row] = value
    else:
        self.values[row*self.columns + column] = value

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api float_matrix fm_clone(float_matrix self) nogil:
    """
    Clone this matrix
    :param self:
    :return:
    """
    cdef float_matrix clone_matrix = fm_new(self.rows, self.columns)
    cblas_scopy(self.size, self.values, 1, clone_matrix.values, 1)
    return clone_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api float_matrix fm_transpose(float_matrix self) nogil:
    """
    Return a transpose of this float matrix
    :return:
    """
    cdef float_matrix fm = fm_clone(self)
    fm.transpose = 1 - self.transpose
    fm.rows = self.columns
    fm.columns = self.rows
    return fm

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api float_matrix fm_multiply_scalar(float_matrix self, float scalar) nogil:
    """
    Multiply all values in the matrix by a real number
    :param scalar: Real number to multiply this matrix
    """
    cdef float_matrix fm = fm_clone(self)
    cblas_sscal(fm.size, scalar, fm.values, 1)
    return fm

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api float_matrix fm_multiply(float_matrix self, float_matrix other) nogil:
    """
    Matrix multiplication
    :param other: The other matrix to multiply
    :return: A new multiplied matrix
    """
    #if self.columns != other.rows:
    #    raise LookupError("Matrix shapes are not compatible for multiplication (%d, %d) // (%d, %d)." % \
    #                      (self.rows, self.columns, other.rows, other.columns))
    cdef float_matrix result = fm_new(self.rows, other.columns)
    cblas_sgemm(101, 112 if self.transpose else 111, 112 if other.transpose else 111, self.rows, other.columns,
                self.columns, 1., self.values, self.rows if self.transpose else self.columns, other.values,
                other.rows if other.transpose else other.columns, 0., result.values,
                result.columns)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api float_matrix fm_add(float_matrix self, float_matrix other) nogil:
    """
    Add 2 matrices
    :param other: The other matrix to add
    :return:
    """
    #if self.rows != other.rows or self.columns != other.columns:
    #    raise LookupError("Matrix shapes are not compatible for multiplication (%d, %d) // (%d, %d)." % \
    #                      (self.rows, self.columns, other.rows, other.columns))
    cdef float_matrix result = fm_new(self.rows, self.columns)
    cdef int row, column, i
    for i in xrange(self.size):
        row = i / self.rows
        column = i % self.rows
        fm_set(result, row, column, fm_get(self, row, column) + fm_get(other, row, column))
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api float_matrix fm_element_wise_multiply(float_matrix self, float_matrix other) nogil:
    """
    Do the element wise multiplication in matrices
    :param other:
    :return:
    """
    #if self.rows != other.rows or self.columns != other.columns:
    #    raise LookupError("Matrix shapes are not compatible for multiplication (%d, %d) // (%d, %d)." % \
    #                      (self.rows, self.columns, other.rows, other.columns))
    #cdef FloatMatrix result = FloatMatrix(self.rows, self.columns)
    cdef float_matrix result = fm_new(self.rows, self.columns)
    cdef int row, column, i
    for i in xrange(self.size):
        row = i / self.rows
        column = i % self.rows
        fm_set(result, row, column, fm_get(self, row, column) * fm_get(other, row, column))
    return result


cdef class FloatMatrix:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(FloatMatrix self, rows=None, columns=None, initialize=True):
        """
        C constructor
        :param rows: Number of rows
        :param columns: Number of columns
        """
        if initialize:
            self.matrix = fm_new_init(rows, columns, 0.)
            if self.matrix is NULL or self.matrix.values is NULL:
                raise MemoryError

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __dealloc__(FloatMatrix self):
        """
        Dealloc the C structures
        """
        fm_destroy(self.matrix)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getitem__(FloatMatrix self, tuple place):
        """
        Get item
        :param tuple:
        :return:
        """
        cdef float_matrix matrix = self.matrix
        cdef int row, column
        row, column = place
        return fm_get(matrix, row, column)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __setitem__(FloatMatrix self, tuple place, float value):
        """
        set item
        :param tuple:
        :return:
        """
        cdef float_matrix matrix = self.matrix
        cdef int row, column
        row, column = place
        fm_set(matrix, row, column, value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def clone(FloatMatrix self):
        """
        Clone this Float Matrix
        :return:
        """
        clone_matrix = FloatMatrix(initialize=False)
        clone_matrix.matrix = fm_clone(self.matrix)
        return clone_matrix

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def transpose(FloatMatrix self):
        """
        Return a transpose of this float matrix
        :return:
        """
        transpose_fm = FloatMatrix(initialize=False)
        transpose_fm.matrix = fm_transpose(self.matrix)
        return transpose_fm

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __mul__(FloatMatrix self, other):
        """
        Return a transpose of this float matrix
        :return:
        """
        cdef FloatMatrix result
        cdef float_matrix self_matrix = self.matrix
        if isinstance(other, float) or isinstance(other, int):
            result = FloatMatrix(initialize=False)
            result.matrix = fm_multiply_scalar(self.matrix, <float>other)
        elif isinstance(other, FloatMatrix):
            if self.matrix.columns != (<FloatMatrix>other).matrix.rows:
                raise LookupError("Matrix shapes are not compatible for multiplication (%d, %d) // (%d, %d)." % \
                              (self.matrix.rows, self.matrix.columns, (<FloatMatrix>other).matrix.rows,
                               (<FloatMatrix>other).matrix.columns))
            result = FloatMatrix(initialize=False)
            result.matrix = fm_multiply(self_matrix, (<FloatMatrix>other).matrix)
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def element_wise_multiplication(FloatMatrix self, FloatMatrix other):
        """
        Do the element wise multiplication in matrices
        :param other:
        :return:
        """
        if self.matrix.rows != (<FloatMatrix>other).matrix.rows or \
                        self.matrix.columns != (<FloatMatrix>other).matrix.columns:
            raise LookupError("Matrix shapes are not compatible for element-wise multiplication (%d, %d) // (%d, %d)." % \
                              (self.matrix.rows, self.matrix.columns, (<FloatMatrix>other).matrix.rows,
                               (<FloatMatrix>other).matrix.columns))
        cdef FloatMatrix result = FloatMatrix(initialize=False)
        result.matrix = fm_element_wise_multiply(self.matrix, other.matrix)
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def  __add__(FloatMatrix self, FloatMatrix other):
        """
        Add 2 matrices
        :param other: The other matrix to add
        :return:
        """
        if self.matrix.rows != other.matrix.rows or self.matrix.columns != other.matrix.columns:
            raise LookupError("Matrix shapes are not compatible for multiplication (%d, %d) // (%d, %d)." % \
                              (self.matrix.rows, self.matrix.columns, other.matrix.rows, other.matrix.columns))
        cdef FloatMatrix result = FloatMatrix(initialize=False)
        result.matrix = fm_add(self.matrix, other.matrix)
        return result

    @property
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def rows(FloatMatrix self):
        return self.matrix.rows

    @property
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def columns(FloatMatrix self):
        return self.matrix.columns

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __len__(FloatMatrix self):
        return self.matrix.size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __str__(FloatMatrix self):
        return u"\n".join(u", ".join(unicode(self[i, j]) for j in xrange(self.columns)) for i in xrange(self.rows))
