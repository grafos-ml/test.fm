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
cdef api float_matrix new__float_matrix(int rows, int columns) nogil:
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
    multiply_scalar(self, 0.)
    return self

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api void destroy(float_matrix self) nogil:
    """
    Dealloc the C structures
    """
    if self.values is not NULL:
        free(self.values)
    if self is not NULL:
        free(self)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api float get(float_matrix self, int row, int column) nogil:
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
cdef api void set(float_matrix self, int row, int column, float value) nogil:
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
cdef api void transpose(float_matrix self, float_matrix fm) nogil:
    """
    Return a transpose of this float matrix
    :return:
    """
    #cdef FloatMatrix transpose = self.clone()
    fm.transpose = 1 - self.transpose
    fm.rows = self.columns
    fm.columns = self.rows

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api void multiply_scalar(float_matrix self, float scalar) nogil:
    """
    Multiply all values in the matrix by a real number
    :param scalar: Real number to multiply this matrix
    """
    #cdef FloatMatrix clone = self.clone()
    cblas_sscal(self.size, scalar, self.values, 1)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api void multiply(float_matrix self, float_matrix other, float_matrix result) nogil:
    """
    Matrix multiplication
    :param other: The other matrix to multiply
    :return: A new multiplied matrix
    """
    #if self.columns != other.rows:
    #    raise LookupError("Matrix shapes are not compatible for multiplication (%d, %d) // (%d, %d)." % \
    #                      (self.rows, self.columns, other.rows, other.columns))
    #cdef FloatMatrix result = FloatMatrix(self.rows, other.columns)
    cblas_sgemm(101, 112 if self.transpose else 111, 112 if other.transpose else 111, self.rows, other.columns,
                self.columns, 1., self.values, self.rows if self.transpose else self.columns, other.values,
                other.rows if other.transpose else other.columns, 0., result.values,
                result.columns)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api void add(float_matrix self, float_matrix other, float_matrix result) nogil:
    """
    Add 2 matrices
    :param other: The other matrix to add
    :return:
    """
    #if self.rows != other.rows or self.columns != other.columns:
    #    raise LookupError("Matrix shapes are not compatible for multiplication (%d, %d) // (%d, %d)." % \
    #                      (self.rows, self.columns, other.rows, other.columns))
    #cdef FloatMatrix result = FloatMatrix(self.rows, self.columns)
    cdef int row, column, i
    for i in xrange(self.size):
        row = i / self.rows
        column = i % self.rows
        set(result, row, column, get(self, row, column) + get(other, row, column))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api void multiply_element_wise(float_matrix self, float_matrix other, float_matrix result) nogil:
    """
    Do the element wise multiplication in matrices
    :param other:
    :return:
    """
    #if self.rows != other.rows or self.columns != other.columns:
    #    raise LookupError("Matrix shapes are not compatible for multiplication (%d, %d) // (%d, %d)." % \
    #                      (self.rows, self.columns, other.rows, other.columns))
    #cdef FloatMatrix result = FloatMatrix(self.rows, self.columns)
    cdef int row, column, i
    for i in xrange(self.size):
        row = i / self.rows
        column = i % self.rows
        set(result, row, column, get(self, row, column) * get(other, row, column))


cdef class FloatMatrix:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(FloatMatrix self, int rows, int columns):
        """
        C constructor
        :param rows: Number of rows
        :param columns: Number of columns
        """

        self.matrix = new__float_matrix(rows, columns)
        if self.matrix is NULL or self.matrix.values is NULL:
            raise MemoryError

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __dealloc__(FloatMatrix self):
        """
        Dealloc the C structures
        """
        destroy(self.matrix)

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
        return get(matrix, row, column)

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
        with nogil:
            set(matrix, row, column, value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def clone(FloatMatrix self):
        """

        :return:
        """
        clone_matrix = FloatMatrix(self.matrix.rows, self.matrix.columns)
        cblas_scopy(self.matrix.size, self.matrix.values, 1, clone_matrix.matrix.values, 1)
        return clone_matrix

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def transpose(FloatMatrix self):
        """
        Return a transpose of this float matrix
        :return:
        """
        cdef FloatMatrix transpose_matrix = self.clone()
        cdef float_matrix self_matrix = self.matrix, other_matrix = transpose_matrix.matrix
        transpose(self_matrix, other_matrix)
        return transpose_matrix

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __mul__(FloatMatrix self, other):
        """
        Return a transpose of this float matrix
        :return:
        """
        cdef FloatMatrix result
        cdef float_matrix self_matrix = self.matrix, other_matrix, result_matrix
        if isinstance(other, float) or isinstance(other, int):
            result = self.clone()
            result_matrix = result.matrix
            multiply_scalar(result_matrix, <float>other)
        elif isinstance(other, FloatMatrix):
            if self.matrix.columns != (<FloatMatrix>other).matrix.rows:
                raise LookupError("Matrix shapes are not compatible for multiplication (%d, %d) // (%d, %d)." % \
                              (self.matrix.rows, self.matrix.columns, (<FloatMatrix>other).matrix.rows,
                               (<FloatMatrix>other).matrix.columns))
            result = FloatMatrix(self.matrix.rows, (<FloatMatrix>other).matrix.columns)
            result_matrix = result.matrix
            other_matrix = (<FloatMatrix>other).matrix
            multiply(self_matrix, other_matrix, result_matrix)
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
        cdef FloatMatrix result = FloatMatrix(self.matrix.rows, self.matrix.columns)
        cdef float_matrix self_matrix = self.matrix, other_matrix = other.matrix, result_matrix = result.matrix
        multiply_element_wise(self_matrix, other_matrix, result_matrix)
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
        cdef FloatMatrix result = FloatMatrix(self.matrix.rows, self.matrix.columns)
        cdef float_matrix self_matrix = self.matrix, other_matrix = other.matrix, result_matrix = result.matrix
        add(self_matrix, other_matrix, result_matrix)
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
