from testfm.models.cutil.float_matrix cimport float_matrix

cdef class IModel:
    """
    Interface for a generic model
    """

cdef class NOGILModel(IModel):
    """
    No gil interface. Implements a nogil get_score and nogil item_score
    """
    cdef float nogil_get_score(self, int user, int item, int extra_context, int *context) nogil

cdef class IFactorModel(NOGILModel):
    """
    Interface/abstract class for factor matrix based models
    """

    cdef float_matrix *c_factors
    cdef int c_number_of_contexts
    cdef int c_number_of_factors

    cdef float_matrix *get_factors(self)
    cdef void dealloc_factors(IFactorModel self) nogil