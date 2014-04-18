"""
Mesure and evaluator interface in c
"""

__author__ = "joaonrb"

cdef class MAPMeasure:
    """
    Implementation of Mean Average Precision.
    """
    cdef float _measure(self, float *ranked_list, int list_size) nogil