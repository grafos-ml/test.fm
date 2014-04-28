cimport cython
from libc.stdlib cimport malloc, free, realloc, rand, RAND_MAX
#import numpy as np
#cimport numpy as np
from testfm.models.cutil.interface import IMatrixFactorization

cdef extern from "math.h":
    float log(float n) nogil
    float fabs(float score) nogil
    float copysign(float x, float y)

cdef extern from "cblas.h":
    float cblas_sdot(int n, float *x, int inc_x, float *x, int inc_x) nogil
    void cblas_scal(float alpha, float *x) nogil
    void cblas_sgemm(char *side, char *uplo, int m, int n, int k, float alpha, float *a, int lda, float *b, int ldb,
                     float beta, float *c, int ldc)



cdef class FloatMatrix:

    cdef float *values
    cdef int rows
    cdef int columns
    cdef int size


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void __cinit__(self, int rows, int columns) nogil:
        self.size = rows * columns
        self.rows = rows
        self.columns = columns
        self.values = malloc(sizeof(float) * self.size)
        if self.values is NULL:
            raise MemoryError()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void multiply_scalar(self, float scalar) nogil:
        cblas_scal(scalar, &self.values)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void multiply(self, FloatMatrix matrix_a, FloatMatrix matrix_b) nogil:
        cblas_sgemm("n", "n", matrix_a.rows, matrix_b.columns, matrix_a.columns, 1., &matrix_a.values, 1,
                    &matrix_b.values, 1, 0., &self.values, 1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void add(self, FloatMatrix matrix_a, FloatMatrix matrix_b) nogil:
        cdef int i
        for i in xrange(self.size):
            self.values[i] = matrix_a.values[i] + matrix_b.values[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void multiply_by_elem(self, FloatMatrix matrix_a, FloatMatrix matrix_b) nogil:
        cdef int i
        for i in xrange(self.size):
            self.values[i] = matrix_a.values[i] * matrix_b.values[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float get(self, int row, int column) nogil:
        return self.values[row*self.rows + column]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set(self, int row, int column, float value) nogil:
        self.values[row*self.rows + column] = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void free(self) nogil:
        free(self.values)

cdef class IntList:
    """
    This list is not organized
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void __cinit__(self) nogil:
            self.max_size = 10
            self.size = 0
            self.values = malloc(sizeof(int)*self.max_size)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int get(self, int index) nogil:
        """
        Get the value in the index of this list
        """
        if not 0 < index < self.size:
            raise IndexError
        return self.values[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void add(self, int value) nogil:
        """
        Add a new value to this list
        """
        if self.size == self.max_size:
            self.max_size += 10
            self.values = realloc(self.values, sizeof(int)*self.max_size)
        self.values[self.size] = value
        self.size += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int pop(self, int index) nogil:
        """
        Removes and return the element in the index
        """
        cdef int result = self.get(index)
        self.size -= 1
        self.values[index] = self.values[self.size]
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void free(self) nogil:
        free(self.values)


cdef class LowTensorCoFi(IMatrixFactorization):
    """
    Cython implementation of tensorCoFi algorithm based on the java version from Alexandros Karatzoglou
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void __cinit__(self, int n_factors=20, int n_iterations=5, double c_lambda=.05, int c_alpha=40):
        """
        Constructor

        :param n_factors: Number of factors to the matrices
        :param n_iterations: Number of iteration in the matrices construction
        :param c_lambda: I came back when I find it out
        :param c_alpha: Constant important in weight calculation
        """
        self.number_of_factors = n_factors
        self.number_of_iterations = n_iterations
        self.constant_lambda = c_lambda
        self.constant_alpha = c_alpha

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def train(self, data):
        """
        Train the model using this data numpy array
        :param data: Numpy array
        :return:
        """
        cdef FloatMatrix c_data, tmp, regularizer, matrix_vector, one, invertible, base
        cdef FloatMatrix *factors
        cdef int *dimensions
        c_data, dimensions, factors, tmp, regularizer, matrix_vector_prod, one, invertible, base = \
            self.allocate_resources(data)
        with nogil:
            self.train_model(c_data, dimensions, factors, tmp, regularizer, matrix_vector_prod, one, invertible, base)
            self.free_resources(c_data, dimensions, factors, tmp, regularizer, matrix_vector_prod, one, invertible,
                                base)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def allocate_resources(self, data):
        """
        Allocate the necessary resources for the execution
        :param data:
        :return:
        """
        cdef int i, j
        cdef float f
        cdef FloatMatrix c_data = FloatMatrix(data.shape[0], data.shape[1])
        for i, f in enumerate(data.data):  # Can be avoided
            c_data.values[i] = f
        self.number_of_dimensions = <int>len(self.dimensions)
        cdef int dimensions[self.number_of_dimensions]
        cdef FloatMatrix factors[self.number_of_dimensions]
        for i in xrange(self.number_of_dimensions):
            dimensions[i] = <int> self.dimensions[i]
            factors[i] = FloatMatrix(self.number_of_factors, dimensions[i])
            for j in xrange(dimensions[i]*self.number_of_factors):
                factors[i].values[j] = <double>rand() / <double>RAND_MAX

        # Matrix of ones with dimension equal to number of factors
        cdef FloatMatrix tmp = FloatMatrix(self.number_of_factors, 1)

        # Matrix of regularizer, one, invertible and base
        cdef FloatMatrix regularizer = FloatMatrix(self.number_of_factors, self.number_of_factors)
        cdef FloatMatrix one = FloatMatrix(self.number_of_factors, self.number_of_factors)
        cdef FloatMatrix invertible = FloatMatrix(self.number_of_factors, self.number_of_factors)
        cdef FloatMatrix base = FloatMatrix(self.number_of_factors, self.number_of_factors)
        regularizer.multiply_scalar(0.)
        one.multiply_scalar(0.)
        invertible.multiply_scalar(0.)
        for i in xrange(self.number_of_factors):
            regularizer.set(i, i, self.constant_lambda)
            one.set(i, i, 1.)

        # Matrix of the vector product
        cdef FloatMatrix matrix_vector_prod = FloatMatrix(self.number_of_factors)
        matrix_vector_prod.multiply_scalar(0.)

        return c_data, dimensions, factors, tmp, regularizer, matrix_vector_prod, one, invertible, base

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void free_resources(self, FloatMatrix data, int *dimensions, FloatMatrix *factors, FloatMatrix tmp,
                             FloatMatrix regularizer, FloatMatrix matrix_vector_prod, FloatMatrix one,
                             FloatMatrix invertible, FloatMatrix base) nogil:
        """
        Free the allocated resources
        :param factors:
        :param tmp:
        :param regularizer:
        :param matrix_vector_prod:
        :param one:
        :param invertible:
        :param base:
        :return:
        """
        cdef int i
        for i in xrange(self.number_of_dimensions):
            factors[i].free()
        free(factors)
        tmp.free()
        regularizer.free()
        matrix_vector_prod.free()
        one.free()
        invertible.free()
        base.free()
        data.free()
        free(dimensions)

    cdef void train_model(self, FloatMatrix data, int *dimensions, FloatMatrix *factors, FloatMatrix tmp,
                          FloatMatrix regularizer, FloatMatrix matrix_vector_prod, FloatMatrix one,
                          FloatMatrix invertible, FloatMatrix base) nogil:
        """
        Train the model
        :param factors:
        :param tmp:
        :param regularizer:
        :param matrix_vector_prod:
        :param one:
        :param invertible:
        :param base:
        :return:
        """
        # Start initiate variables
        cdef IntList *tensor[self.number_of_dimensions]  # TO FREE
        cdef IntList data_row_list
        cdef int i, j, index, data_row, iteration, current_dimension, matrix_index, data_entry, data_col
        cdef FloatMatrix tmp_fm[self.number_of_dimensions]  # TO FREE
        for i in xrange(tmp_fm):
            tmp_fm[i] = FloatMatrix(dimensions[i], dimensions[i])
        cdef float weight
        for i in xrange(self.number_of_dimensions):
            tensor[i] = malloc(sizeof(IntList)*dimensions[i])
            for j in xrange(dimensions[i]):
                tensor[i][j] = IntList()
            for data_row in xrange(data.size):
                index = data.get(data_row, i)
                tensor[i][index].set(data_row)
        # End initiate variables

        # Start Iteration
        for iteration in xrange(self.number_of_iterations):
            for current_dimension in xrange(self.number_of_dimensions):
                if self.number_of_dimensions == 2:
                    base.multiply(factors[1-current_dimension], factors[1-current_dimension])
                else:
                    # Initiate base
                    for i in xrange(self.number_of_factors ** 2):
                        base.values[i] = 1.
                    for matrix_index in xrange(self.number_of_dimensions):
                        if matrix_index != current_dimension:
                            tmp_fm[matrix_index].multiply(factors[matrix_index], factors[matrix_index])
                            base.multiply_by_elem(base, factors[matrix_index])
                ############
                for data_entry in xrange(dimensions[current_dimension]):
                    data_row_list = tensor[current_dimension][data_entry]
                    for i in xrange(data_row_list.size):
                        data_row = data_row_list.values[i]
                        for j in xrange(tmp.rows*tmp.columns):
                            tmp.values[j] = 1.
                        for data_col in xrange(self.number_of_dimensions):
                            if data_col != current_dimension:
                                tmp.multiply(factors[data_col][])




for (int iter = 0; iter < this.iter; iter++) {

    for (int currentDimension = 0; currentDimension < this.dimensions.length; currentDimension++) {

        for (int dataEntry = 1; dataEntry <= dimensions[currentDimension]; dataEntry++) {
            dataRowList = tensor.get(currentDimension).get(dataEntry);
                for(int dataRow : dataRowList) {
                    temp = temp.mul((float) 0.0).addi((float) 1.0);
                    for (int dataCol = 0; dataCol < dimensions.length; dataCol++)
                        if (dataCol != currentDimension)
                            temp = temp.muliColumnVector(
                                this.Factors.get(dataCol).getColumn((int) dataArray.get(dataRow, dataCol)-1));
                    float score = dataArray.get(dataRow, dataArray.columns - 1);
                    weight =  1.0f + this.p * (float)Math.log(1.0f + (float)(Math.abs(score)));
                    invertible = invertible.rankOneUpdate((float) (weight - 1.0), temp);
                    matrixVectorProd = matrixVectorProd.addColumnVector(temp.mul((float) Math.signum(score)*weight));
                }
            invertible = invertible.addi(base);
            regularizer = regularizer.mul((float) 1.0 / (float) dimensions[currentDimension]);
            invertible = invertible.addi(regularizer);
            try {
                invertible = Solve.solveSymmetric(invertible, one);
            } catch (Exception e) {
                System.out.print(invertible.toString());
                e.printStackTrace();
            }
            this.Factors.get(currentDimension).putColumn(dataEntry-1, invertible.mmul(matrixVectorProd));

            invertible = invertible.mul((float) 0.0);
            matrixVectorProd = matrixVectorProd.mul((float) 0.0);
        }
    }

