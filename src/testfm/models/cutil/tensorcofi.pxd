

cdef class FloatMatrix:

    cdef float *values
    cdef int rows
    cdef int columns
    cdef int size

    cdef void __cinit__(self, int rows, int columns) nogil
    cdef void multiply_scalar(self, float scalar) nogil
    cdef void multiply(self, FloatMatrix matrix_a, FloatMatrix matrix_b) nogil
    cdef void add(self, FloatMatrix matrix_a, FloatMatrix matrix_b) nogil
    cdef void multiply_by_elem(self, FloatMatrix matrix_a, FloatMatrix matrix_b) nogil
    cdef float get(self, int row, int column) nogil
    cdef void set(self, int row, int column, float value) nogil
    cdef void free(self) nogil

cdef class IntList:

    cdef int *values
    cdef int size
    cdef int max_size

    cdef void __cinit__(self) nogil
    cdef int get(self, int index) nogil
    cdef void add(self, int value) nogil
    cdef int pop(self, int index) nogil
    cdef void free(self) nogil

cdef class CTensorCoFi:
    """
    Python implementation of tensorCoFi algorithm based on the java version from Alexandros Karatzoglou
    """

    cdef int number_of_factors
    cdef int number_of_iterations
    cdef float constant_lambda
    cdef int constant_alpha
    cdef list dimensions
    cdef int number_of_dimensions

    cdef void train_model(self, FloatMatrix data, int *dimensions, FloatMatrix *factors, FloatMatrix tmp,
                          FloatMatrix regularizer, FloatMatrix matrix_vector_prod, FloatMatrix one,
                          FloatMatrix invertible, FloatMatrix base) nogil
    cdef void free_resources(self, FloatMatrix data, int *dimensions, FloatMatrix *factors, FloatMatrix tmp,
                             FloatMatrix regularizer, FloatMatrix matrix_vector_prod, FloatMatrix one,
                             FloatMatrix invertible, FloatMatrix base) nogil