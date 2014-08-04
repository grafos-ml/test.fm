"""
Created on 20 January 2014

Evaluation measures for the list wise (i.e., precision) and point wise (i.e., RMSE) recommendations. All of the measures
will get a SORTED list of items with ground truth for the user and will compute their measures based on it.

The error measure gets a SORTED (descending order or prediction) list. The 0 element of the list has highest prediction
and is top of the recommendation list. The ground true can be either float or the boolean. If it is float, it
means the rating and the relevance threshold should be provided in order to compute the ranking measure. If it is
boolean, True=relevant, False=irrelevant. [(True, 0.9), (False, 0.55), (True, 0.4), (True, 0.2)]

Now we can compute the ranking error measure. For example Precision@2 = 0.5.

.. moduleauthor:: joaonrb <joaonrb@gmail.com>
"""
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf


cdef class NOGILMeasure:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float nogil_measure(self, float *ranked_list, int list_size) nogil:
        cdef float result = 0.
        return result

cdef class MAPMeasure(NOGILMeasure):

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float nogil_measure(self, float *ranked_list, int list_size) nogil:
        """
        Each class of the Measure has to implement this method. It basically
        knows how to compute the performance measure such as MAP, P@k, etc.

        recs - a list of tuples of ground truth and score.
            Example: [(True, 0.92), (False, 0.55), (True, 0.41), (True, 0.2)]
        """
        cdef float map_measure = 0.
        cdef float relevant = 0.
        cdef int i
        for i in range(list_size):
            #printf(">>>%f %f\n", ranked_list[i], ranked_list[i+1])
            if ranked_list[i*2] == 1.:
                relevant += 1.
                map_measure += (relevant / (i+1.))
        return 0.0 if relevant == 0. else (map_measure/relevant)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def measure(self, recs, n=None):
        """
        Example of how to use map and the input format.
        >>> mapm = MAPMeasure()
        >>> mapm.measure([])
        nan
        #this is a perfect ranking of a single relevant item
        >>> mapm.measure([(True, 0.00)])
        1.0
        >>> mapm.measure([(False, 0.00)])
        0.0
        >>> mapm.measure([(False, 0.01), (True, 0.00)])
        0.5
        The following example is taken from wikipedia (as far as I remember)
        >>> mapm.measure([(False, 0.9), (True, 0.8), (False, 0.7), (False, 0.6), (True, 0.5), (True, 0.4), \
        (True, 0.3), (False, 0.2), (False, 0.1), (False, 0)])
        0.4928571428571428
        """
        #cdef float result
        #cdef int l = len(recs)
        if not isinstance(recs, list) or len(recs) < 1:
            return float("nan")
        #try:
        #    crec = <float *>malloc(len(recs)*sizeof(float)*2)
        #    if crec is NULL:
        #        raise MemoryError()
        #    for i in range(len(recs)):
        #        crec[i*2], crec[i*2+1] = float(recs[i][0]), recs[i][1]
        #    with nogil:
        #        result = self.nogil_measure(crec, l)
        #    return result
        #finally:
        #    free(crec)
        map_measure = 0.
        relevant = 0.

        for i, (ground_truth, prediction) in enumerate(recs):
            if ground_truth is True:
                relevant += 1.
                map_measure += relevant / (i+1)

        return 0.0 if relevant == 0 else map_measure/relevant

    @property
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def name(self):
        return u"MAPMeasure"

