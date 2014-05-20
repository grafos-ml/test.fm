"""
FloatMatrix implementation for native no GIL matrix operations
"""
cimport cython
from libc.stdlib cimport malloc, free, rand, RAND_MAX, exit as cexit
from libc.stdio cimport printf

cdef extern from "cblas.h":
    float cblas_scopy(int n, float *x, int incx, float *y, int incy) nogil  # For clone
    float cblas_sscal(int n, float a, float *x, int incx) nogil  # For multiply vector by real
    void cblas_sgemm(int order, int transa, int transb, int m, int n, int k, float alpha, float *a, int lda,
                     float *b, int ldb, float beta, float *c, int ldc) nogil  # For matrix multiplication
    #void cblas_symm(char *side, char *uplo, int m, int n, float alpha, float *a, int lda, float *b, int ldb, float beta,
    #                float *c, int ldc) nogil  # For matrix multiplication

cdef extern from "clapack.h":
    int clapack_sgesv(const int order, const int n, const int nrhs, float *a, const int lda, int *ipiv, float *b,
                      const int ldb) nogil


@cython.overflowcheck(False)
@cython.cdivision(False)
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


cdef api void fm_destroy(float_matrix self) nogil:
    """
    Dealloc the C structures
    """
    if self is not NULL:
        if self.values is not NULL:
            free(self.values)
        free(self)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
cdef api float fm_get(float_matrix self, int row, int column) nogil:
    """
    Get a value from the matrix in the cell (row, column)
    :param row: Row index to get the item
    :param column: Column item to get the item
    :return: The float element in the specific cell
    """
    #if 0 > row or row >= self.rows or 0 > column or column >= self.columns:
    #    raise IndexError
    if self.transpose == 1:
        return <float>self.values[column*self.rows + row]
    return <float>self.values[row*self.columns + column]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
cdef api void fm_set(float_matrix self, int row, int column, float value) nogil:
    """
    Set the specific cell to value
    :param row: Row index to get the item
    :param column: Column item to get the item
    :param value: The new value in the cell
    """
    #printf("(%d/%d, %d/%d) = %f\n", row, self.rows, column, self.columns, value)
    if self.transpose == 1:
        #if column*self.rows + row >= self.size:
        #    cexit(155)
        self.values[column*self.rows + row] = value
    else:
        #if row*self.columns + column >= self.size:
        #    cexit(154)
        self.values[row*self.columns + column] = value


cdef api float_matrix fm_static_clone(float_matrix self, float_matrix clone_matrix) nogil:
    """
    Clone this matrix
    :param self:
    :return:
    """
    cblas_scopy(self.size, self.values, 1, clone_matrix.values, 1)
    return clone_matrix


cdef api float_matrix fm_clone(float_matrix self) nogil:
    """
    Clone this matrix
    :param self:
    :return:
    """
    cdef float_matrix clone_matrix = fm_new(self.rows, self.columns)
    return fm_static_clone(self, clone_matrix)


@cython.overflowcheck(False)
@cython.cdivision(False)
cdef api float_matrix fm_static_transpose(float_matrix self, float_matrix fm) nogil:
    """
    Return a transpose of this float matrix
    :return:
    """
    fm.transpose = 1 - self.transpose
    cdef int rows, columns
    rows, columns = self.rows, self.columns
    fm.rows, fm.columns = columns, rows
    return fm


cdef api float_matrix fm_transpose(float_matrix self) nogil:
    """
    Return a transpose of this float matrix
    :return:
    """
    cdef float_matrix fm = fm_clone(self)
    return fm_static_transpose(self, fm)


cdef api float_matrix fm_static_multiply_scalar(float_matrix self, float scalar, float_matrix fm) nogil:
    """
    Multiply all values in the matrix by a real number
    :param scalar: Real number to multiply this matrix
    """
    cblas_sscal(fm.size, scalar, fm.values, 1)
    return fm


cdef api float_matrix fm_multiply_scalar(float_matrix self, float scalar) nogil:
    """
    Multiply all values in the matrix by a real number
    :param scalar: Real number to multiply this matrix
    """
    cdef float_matrix fm = fm_clone(self)
    return fm_static_multiply_scalar(self, scalar, fm)


cdef api float_matrix fm_static_multiply(float_matrix self, float_matrix other, float_matrix result) nogil:
    """
    Matrix multiplication
    :param other: The other matrix to multiply
    :return: A new multiplied matrix
    """
    cblas_sgemm(101, 112 if self.transpose else 111, 112 if other.transpose else 111, self.rows, other.columns,
                self.columns, 1., self.values, self.rows if self.transpose else self.columns, other.values,
                other.rows if other.transpose else other.columns, 0., result.values, result.columns)
    return result


cdef api float_matrix fm_multiply(float_matrix self, float_matrix other) nogil:
    """
    Matrix multiplication
    :param other: The other matrix to multiply
    :return: A new multiplied matrix
    """
    cdef float_matrix result = fm_new(self.rows, other.columns)
    return fm_static_multiply(self, other, result)


@cython.overflowcheck(False)
@cython.cdivision(False)
cdef api float_matrix fm_static_multiply_column(float_matrix self, float_matrix other, int row,
                                                float_matrix result) nogil:
    """
    Matrix multiplication by other column
    :param other: The other matrix to multiply. It should have the same number of rows than self but only one column
    :return: A new multiplied matrix
    """
    cdef int i, r, c
    for i in xrange(self.size):
        r = i / self.columns
        c = i % self.columns
        fm_set(result, r, c, fm_get(self, r, c) * fm_get(other, r, row))
    return result


cdef api float_matrix fm_multiply_column(float_matrix self, float_matrix other, int row) nogil:
    """
    Matrix multiplication
    :param other: The other matrix to multiply
    :return: A new multiplied matrix
    """
    cdef float_matrix result = fm_new(self.rows, self.columns)
    return fm_static_multiply_column(self, other, row, result)


@cython.overflowcheck(False)
@cython.cdivision(False)
cdef api float_matrix fm_static_add(float_matrix self, float_matrix other, float_matrix result) nogil:
    """
    Add 2 matrices
    :param other: The other matrix to add
    :return:
    """
    cdef int row, column, i
    for i in xrange(self.size):
        row = i / self.columns
        column = i % self.columns
        fm_set(result, row, column, fm_get(self, row, column) + fm_get(other, row, column))
    return result


cdef api float_matrix fm_add(float_matrix self, float_matrix other) nogil:
    """
    Add 2 matrices
    :param other: The other matrix to add
    :return:
    """
    cdef float_matrix result = fm_new(self.rows, self.columns)
    return fm_static_add(self, other, result)


@cython.overflowcheck(False)
@cython.cdivision(False)
cdef api float_matrix fm_static_element_wise_multiply(float_matrix self, float_matrix other, float_matrix result) nogil:
    """
    Do the element wise multiplication in matrices
    :param other:
    :return:
    """
    cdef int row, column, i
    for i in xrange(self.size):
        row = i / self.columns
        column = i % self.columns
        fm_set(result, row, column, fm_get(self, row, column) * fm_get(other, row, column))
    return result


cdef api float_matrix fm_element_wise_multiply(float_matrix self, float_matrix other) nogil:
    """
    Do the element wise multiplication in matrices
    :param other:
    :return:
    """
    cdef float_matrix result = fm_new(self.rows, self.columns)
    return fm_static_element_wise_multiply(self, other, result)


@cython.overflowcheck(False)
@cython.cdivision(False)
cdef api float_matrix fm_static_element_wise_division(float_matrix self, float_matrix other, float_matrix result) nogil:
    """
    Do the element wise multiplication in matrices
    :param other:
    :return:
    """
    cdef int row, column, i
    for i in xrange(self.size):
        row = i / self.columns
        column = i % self.columns
        fm_set(result, row, column, fm_get(self, row, column) / fm_get(other, row, column))
    return result


cdef api float_matrix fm_element_wise_division(float_matrix self, float_matrix other) nogil:
    """
    Do the element wise multiplication in matrices
    :param other:
    :return:
    """
    cdef float_matrix result = fm_new(self.rows, self.columns)
    return fm_static_element_wise_division(self, other, result)


cdef api float_matrix fm_diagonal(float_matrix self, float scalar) nogil:
    """
    Make self diagonal. Make sure the matrix is square
    :param other:
    :return:
    """
    self = fm_static_multiply_scalar(self, 0., self)
    cdef int i
    for i in xrange(self.rows):
        fm_set(self, i, i, scalar)
    return self


cdef api float_matrix fm_create_diagonal(int dim, float scalar) nogil:
    """
    Do the element wise multiplication in matrices
    :param dim: Size of side of the matrix
    :return:
    """
    cdef float_matrix fm = fm_new(dim, dim)
    return fm_diagonal(fm, scalar)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
cdef api float_matrix fm_create_random(int rows, int columns) nogil:
    """
    Create a matrix with random numbers between 0 and 1
    """
    cdef float_matrix fm = fm_new(rows, columns)
    cdef int i
    for i in xrange(fm.size):
        fm.values[i] = rand() / <float>RAND_MAX
    return fm

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef api void fm_print(float_matrix self) nogil:
#    """
#    Print the matrix
#    """
#    cdef int row, column
#    for row in range(self.rows):
#        if row != 0:
#            printf("\n")
#        for columns in range(self.columns):
#            if columns != 0:
#                printf(", ")
#            printf("%f", fm_get(self, row, columns))


cdef api float_matrix fm_solve(float_matrix self, float_matrix result, int *ipiv) nogil:
    """
    Solve the system SELF * X = RESULT
    :return The solution for this system
    """
    cdef float_matrix solution = fm_clone(result)
    clapack_sgesv(101, self.columns, solution.columns, self.values, self.rows, ipiv,
                  solution.values, solution.rows)
    return solution


cdef class FloatMatrix:

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

    def __dealloc__(FloatMatrix self):
        """
        Dealloc the C structures
        """
        fm_destroy(self.matrix)

    def __getitem__(FloatMatrix self, tuple place):
        """
        Get item
        :param tuple:
        :return:
        """
        row, column = place
        if 0 <= row < self.rows and 0 <= column < self.columns:
            return fm_get(self.matrix, <int>row, <int>column)
        else:
            raise IndexError, "%s cell doesn't exist. Matrix dimension is (%d, %d)" % (place, self.rows, self.columns)

    def __setitem__(FloatMatrix self, tuple place, float value):
        """
        set item
        :param tuple:
        :return:
        """
        row, column = place
        if 0 <= row < self.rows and 0 <= column < self.columns:
            fm_set(self.matrix, <int>row, <int>column, value)
        else:
            raise IndexError, "%s cell doesn't exist. Matrix dimension is (%d, %d)" % (place, self.rows, self.columns)

    def clone(FloatMatrix self):
        """
        Clone this Float Matrix
        :return:
        """
        clone_matrix = FloatMatrix(initialize=False)
        clone_matrix.matrix = fm_clone(self.matrix)
        return clone_matrix

    def transpose(FloatMatrix self):
        """
        Return a transpose of this float matrix
        :return:
        """
        transpose_fm = FloatMatrix(initialize=False)
        transpose_fm.matrix = fm_transpose(self.matrix)
        return transpose_fm

    @property
    def is_transpose(FloatMatrix self):
        """
        Return true if self is transpose
        :return:
        """
        return bool(self.matrix.transpose)

    def __mul__(FloatMatrix self, other):
        """
        Return a transpose of this float matrix
        :return:
        """
        cdef FloatMatrix result
        if isinstance(other, float) or isinstance(other, int):
            result = FloatMatrix(initialize=False)
            result.matrix = fm_multiply_scalar(self.matrix, <float>other)
        elif isinstance(other, FloatMatrix):
            if self.matrix.columns != (<FloatMatrix>other).matrix.rows:
                raise LookupError("Matrix shapes are not compatible for multiplication (%d, %d)//(%d, %d)." % \
                              (self.matrix.rows, self.matrix.columns, (<FloatMatrix>other).matrix.rows,
                               (<FloatMatrix>other).matrix.columns))
            result = FloatMatrix(initialize=False)
            result.matrix = fm_multiply(self.matrix, (<FloatMatrix>other).matrix)
        return result

    def __div__(FloatMatrix self, FloatMatrix other):
        """
        Return a division of this float matrix with other
        :return:
        """
        cdef FloatMatrix result = FloatMatrix(initialise=False)
        result.matrix = fm_element_wise_division(self.matrix, other.matrix)

        return result

    def element_wise_multiplication(FloatMatrix self, FloatMatrix other):
        """
        Do the element wise multiplication in matrices
        :param other:
        :return:
        """
        if self.matrix.rows != other.matrix.rows or self.matrix.columns != other.matrix.columns:
            raise LookupError("Matrix shapes are not compatible for element-wise multiplication (%d, %d)//(%d, %d)." % \
                              (self.matrix.rows, self.matrix.columns, other.matrix.rows, other.matrix.columns))
        cdef FloatMatrix result = FloatMatrix(initialize=False)
        result.matrix = fm_element_wise_multiply(self.matrix, other.matrix)
        return result

    def column_multiplication(FloatMatrix self, FloatMatrix other, int column):
        """
        Do the element wise multiplication in matrices
        :param other:
        :return:
        """
        if self.matrix.rows != other.matrix.rows:
            raise LookupError("Matrix shapes are not compatible for column multiplication (%d, %d)//(%d, %d)." % \
                              (self.matrix.rows, self.matrix.columns, (<FloatMatrix>other).matrix.rows,
                               (<FloatMatrix>other).matrix.columns))
        cdef FloatMatrix result = FloatMatrix(initialize=False)
        result.matrix = fm_multiply_column(self.matrix, other.matrix, column)
        return result

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
    def rows(FloatMatrix self):
        return self.matrix.rows

    @property
    def columns(FloatMatrix self):
        return self.matrix.columns

    def __len__(FloatMatrix self):
        return self.matrix.size

    def __str__(FloatMatrix self):
        return u"\n".join(u", ".join(unicode(self[i, j]) for j in xrange(self.columns)) for i in xrange(self.rows))

    @staticmethod
    def one(int rows, int columns):
        """
        Create a float matrix of ones
        :param rows:
        :param columns:
        :return:
        """
        cdef FloatMatrix fm = FloatMatrix(initialize=False)
        fm.matrix = fm_new_init(rows, columns, 1.)
        return fm

    @staticmethod
    def eye(int dim):
        """
        Create a diagonal matrix
        :param dim: Dimension of a side of the matrix
        :return:
        """
        cdef FloatMatrix fm = FloatMatrix(initialize=False)
        fm.matrix = fm_create_diagonal(dim, 1.)
        return fm

    @staticmethod
    def random(int rows, int columns):
        """
        Create a matrix with pseudo random values between 0 and 1
        :param dim: Dimension of a side of the matrix
        :return:
        """
        cdef FloatMatrix fm = FloatMatrix(initialize=False)
        fm.matrix = fm_create_random(rows, columns)
        return fm
