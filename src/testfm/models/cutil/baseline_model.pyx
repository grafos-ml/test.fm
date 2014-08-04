cimport cython
from libc.stdlib cimport rand, RAND_MAX
from testfm.models.cutil.interface cimport NOGILModel

cdef class NOGILRandomModel(NOGILModel):
    """
    Random model
    """

    @cython.cdivision(False)
    cdef float nogil_get_score(NOGILRandomModel self, int user, int item, int extra_context, int *context) nogil:
        return rand() / <float>RAND_MAX

    def get_name(self):
        return "Random"