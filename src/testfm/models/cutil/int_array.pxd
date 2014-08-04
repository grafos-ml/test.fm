

ctypedef public struct _int_array:
    int *values
    int _max_size
    int _size

ctypedef public _int_array *int_array


cdef api int_array ia_new() nogil
cdef api void ia_destroy(int_array self) nogil
cdef api int ia_add(int_array self, int value) nogil
cdef api int ia_size(int_array self) nogil