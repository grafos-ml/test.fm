"""
Mesure and evaluator interface in c
"""

__author__ = "joaonrb"

cdef class NOGILMeasure:
    """
    Implementation of Mean Average Precision.
    """
    cdef float nogil_measure(self, float *ranked_list, int list_size) nogil