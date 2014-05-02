
cimport cython
from libc.stdlib cimport malloc, free, realloc


ctypedef public struct _int_array:
    int *values
    int max_size
    int size

ctypedef public _int_array *int_array


@cython.boundscheck(False)
@cython.wraparound(False)
cdef api int_array ia_new() nogil:
    """
    Create a new int_array
    :return:
    """
    cdef int_array ia = <int_array>malloc(sizeof(_int_array))
    ia.max_size = 10
    ia.size = 0
    ia.values = <int *>malloc(sizeof(float)*ia.max_size)
    return ia


@cython.boundscheck(False)
@cython.wraparound(False)
cdef api void ia_destroy(int_array self) nogil:
    """
    Destroy this int array
    :param self:
    :return:
    """
    if self.values is not NULL:
        free(self.values)
    if self is not NULL:
        free(self)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef api void ia_add(int_array self, int value) nogil:
    """
    Add a new value to the array
    :param value:
    :return:
    """
    if self.size >= self.max_size:
        self.max_size += 10
        self.values = <int *>realloc(self.values, sizeof(int)*self.max_size)
    self.values[self.size] = value
    self.size += 1


cdef api int ia_get(int_array self, int index) nogil:
    """
    Get element in index
    :param self:
    :param index:
    :return:
    """
    if  0 <= index <= self.size:
        return self.values[index]