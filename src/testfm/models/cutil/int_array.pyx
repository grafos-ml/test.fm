
cimport cython
from libc.stdlib cimport malloc, free, realloc


cdef api int_array ia_new() nogil:
    """
    Create a new int_array
    :return:
    """
    cdef int_array ia = <int_array>malloc(sizeof(_int_array))
    ia._max_size = 10
    ia.values = <int *>malloc(sizeof(int) * ia._max_size)
    ia._size = 0
    return ia


cdef api void ia_destroy(int_array self) nogil:
    """
    Destroy this int array
    :param self:
    :return:
    """
    if self is not NULL:
        if self.values is not NULL:
            free(self.values)
        free(self)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef api int ia_add(int_array self, int value) nogil:
    """
    Add a new value to the array
    :param value:
    :return:
    """
    cdef int *tmp
    if self._size >= self._max_size:
        self._max_size += 20
        tmp = <int *>realloc(self.values, sizeof(int)*self._max_size)
        if tmp is NULL:
            self._max_size -= 20
            return 1
        self.values = tmp
    self.values[self._size] = value
    self._size += 1
    return 0


cdef api int ia_size(int_array self) nogil:
    return self._size
