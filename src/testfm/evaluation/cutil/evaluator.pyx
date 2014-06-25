cimport cython
from cython.parallel cimport parallel, prange
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from testfm.evaluation.cutil.measures cimport NOGILMeasure
from testfm.models.cutil.interface cimport NOGILModel
import random


cdef float merge_max(float a, float b) nogil:
    return a if a > b else b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
cdef void merge_helper(float *input, int left, int right, float *scratch) nogil:
    #base case: one element
    if right == left + 1:
        return
    cdef int i = 0
    cdef int length = right - left
    cdef int midpoint_distance = length/2
    # l and r are to the positions in the left and right subarrays
    cdef int l = left, r = left + midpoint_distance

    # sort each subarray
    merge_helper(input, left, left + midpoint_distance, scratch)
    merge_helper(input, left + midpoint_distance, right, scratch)

    # merge the arrays together using scratch for temporary storage
    for i in range(length):
        # Check to see if any elements remain in the left array; if so, we check if there are any elements left in
        # the right array; if so, we compare them.  Otherwise, we know that the merge must use take the element
        # from the left array
        if l < left + midpoint_distance and (r == right or merge_max(input[l*2+1], input[r*2+1]) == input[l*2+1]):
            scratch[i*2], scratch[i*2+1] = input[l*2], input[l*2+1]
            l+=1
        else:
            scratch[i*2], scratch[i*2+1] = input[r*2], input[r*2+1]
            r+=1
    # Copy the sorted subarray back to the input
    for i in range(left, right):
        input[i*2], input[i*2+1] = scratch[i*2-left*2], scratch[(i*2-left*2)+1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
cdef int mergesort(float *input, int size) nogil:
    cdef float *scratch = <float *>malloc(size * sizeof(float) * 2)
    if scratch is not NULL:
        merge_helper(input, 0, size, scratch)
        free(scratch)
        return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
cdef int is_in(int value, int size, int *int_list) nogil:
    """
    Check if value in tuple list.
    :param value:
    :param tuple_list:
    :param size:
    :return:
    """
    cdef int i
    for i in range(size):
        if value == int_list[i+1]:
            #with gil:
            #    print "<%d == %d>" % (value, int_list[i+1])
            return 1
    return 0



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
def evaluate_model(factor_model, testing_data, measures, all_items, non_relevant_count, k):
    """
    Try to apply native multi threading to evaluation. It can put the score calculation into threading if the model
    supports nogil and the measure if the measure type supports nogil.

    :param factor_model: ModelInterface  an instance of ModelInterface
    :param measures: list of measure we want to compute (instances of)
    :param all_items: list of items available in the data set (used for negative sampling). If set to None, then
        testing items are used for this
    :param non_relevant_count: int number of non relevant items to add to the list for performance evaluation
    :return: list of score corresponding to measures
    """
    cdef int i, j, user, item, c_nrc = non_relevant_count or len(all_items)

    cdef int *c_all_items = NULL
    cdef int *size_of_user_items = NULL
    cdef int size_of_items, size_of_users
    cdef int **c_grouped = NULL
    try:
        k = k or -1
        grouped = testing_data.groupby('user')
        all_items = testing_data.item.unique() if all_items is None else all_items
        size_of_items = len(all_items)
        size_of_users = len(grouped)
        # compute

        ################################################################################################################
        c_all_items = <int *>malloc(sizeof(int) * size_of_items)  # Define all_items in c
        if c_all_items is NULL:
            raise MemoryError
        c_grouped = <int **>malloc(sizeof(int *) * size_of_users)  # Define a list of lists for user's items
        if c_grouped is NULL:
            raise MemoryError
        for i in range(size_of_users):
            c_grouped[i] = NULL
        size_of_user_items = <int *>malloc(sizeof(int) * size_of_users)  # Define the number of items per user
        if size_of_user_items is NULL:
            raise MemoryError
        for i, (user, items) in enumerate(grouped):
            size_of_user_items[i] = len(items)
            c_grouped[i] = <int *>malloc(sizeof(int) * (size_of_user_items[i] + 1))
            if c_grouped[i] is NULL:
                raise MemoryError
            c_grouped[i][0] = factor_model.data_map[factor_model.get_user_column()][user]
            for j, item in enumerate(items["item"], start=1):
                c_grouped[i][j] = factor_model.data_map[factor_model.get_item_column()][item]
        for i, item in enumerate(all_items):
            c_all_items[i] = factor_model.data_map[factor_model.get_item_column()][item]

        ################################################################################################################
        nogil_measures = []
        gil_measures = []
        for m in measures:
            if isinstance(m, NOGILMeasure):
                nogil_measures.append(m)
            else:
                gil_measures.append(m)
        if isinstance(factor_model, NOGILModel):
            results = evaluate_full_threading(factor_model, size_of_users, size_of_user_items, c_grouped,
                                              nogil_measures, size_of_items, c_all_items, c_nrc, k)
            results += evaluate_model_only_threading(factor_model, size_of_users, size_of_user_items, c_grouped,
                                                     gil_measures, size_of_items, c_all_items, c_nrc, k)
        #else:
        #    results = evaluate_measure_only_threading(factor_model, size_of_users, size_of_user_items, c_grouped,
        #                                              nogil_measures, size_of_items, c_all_items, c_nrc, k)
        #    results += evaluate_no_threading(factor_model, size_of_users, size_of_user_items, c_grouped, gil_measures,
        #                                     size_of_items, c_all_items,  c_nrc, k)
        return results
    finally:
        if c_all_items is not NULL:
            free(c_all_items)
        if c_grouped is not NULL:
            for i in range(size_of_users):
                if c_grouped[i] is not NULL:
                    free(c_grouped[i])
            free(c_grouped)
        if size_of_user_items is not NULL:
            free(size_of_user_items)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
cdef list evaluate_full_threading(NOGILModel factor_model, int size_of_users, int *size_of_user_items, int **c_grouped,
                                  list nogil_measures, int size_of_items, int *c_all_items, int non_relevant_count,
                                  int k):
    """
    Evaluate using multi thread for both scoring and measure
    :param factor_model:
    :param grouped:
    :param nogil_measures:
    :param all_items:
    :param non_relevant_count:
    :param k:
    :return:
    """
    if len(nogil_measures) == 0:
        return []
    cdef int i, j, n_measures = len(nogil_measures), n = size_of_users * n_measures, c_nrc
    cdef dict measures = {}
    cdef float fi
    cdef NOGILMeasure m
    cdef list result = []
    with nogil, parallel():
        for i in prange(n, schedule="guided"):
            with gil:
                m = nogil_measures[i / size_of_users]
            j =  i % size_of_users
            c_nrc = non_relevant_count if size_of_items - non_relevant_count >= size_of_user_items[j] \
                    else size_of_items - size_of_user_items[j]
            fi = measure_full_nogil(factor_model, size_of_user_items[j], c_grouped[j], m, size_of_items, c_all_items, 
                                    c_nrc, k)

            with gil:
                try:
                    measures[m.name].append(fi)
                except KeyError:
                    measures[m.name] = [fi]
    for m in nogil_measures:
        #print measures
        result.append(sum(measures[m.name]))
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
cdef float measure_full_nogil(NOGILModel factor_model, int size_of_user_items, int *user_items, NOGILMeasure measure,
                              int size_of_all_items, int *all_items, int non_relevant_count, int k) nogil:
    """
    Evaluate some user according some measure using full nogil threading
    :param factor_model:
    :param size_of_user_items:
    :param user_items:
    :param measure:
    :param size_of_all_items:
    :param all_items:
    :param non_relevant_count:
    :param k:
    :return:
    """
    cdef int i, j, secure_counter = 0, total = 0, total_of_items = non_relevant_count + size_of_user_items
    cdef float *ranked_list = <float *>malloc(sizeof(float) * total_of_items * 2)
    cdef float result
    with gil:
        items = random.sample([all_items[item] for item in range(size_of_all_items) 
                              if not is_in(all_items[item], size_of_user_items, user_items)], non_relevant_count)
    if ranked_list is NULL:
        return -1.
    for i in range(size_of_user_items):
        #if i+1 > size_of_user_items:
        #    with gil:
        #        raise Exception
       ranked_list[i*2], ranked_list[i*2+1] = 1., factor_model.nogil_get_score(user_items[0], user_items[i+1], 0, NULL)
    for j in range(non_relevant_count):
        total = i+j+1
        with gil:
            j = items.pop()
        ranked_list[total*2], ranked_list[total*2+1] = 0., factor_model.nogil_get_score(user_items[0], j, 0, NULL)
    #if total != size_of_user_items+non_relevant_count:
    #    with gil:
    #        print([(ranked_list[i*2], ranked_list[i*2+1]) for i in range(size_of_user_items+non_relevant_count)])
    #        raise Exception("%d, %d" % (total, size_of_user_items+non_relevant_count))
    # ->
    #for i in range(size_of_all_items):
    #    if total >= total_of_items:
    #        break
    #    if is_in(all_items[i], size_of_user_items, user_items):
    #        ranked_list[total*2], ranked_list[total*2+1] = 1., factor_model.nogil_get_score(user_items[0], all_items[i],
    #                                                                                        0, NULL)
    #        total += 1
    #    elif secure_counter < non_relevant_count:
    #        ranked_list[total*2], ranked_list[total*2+1] = 0., factor_model.nogil_get_score(user_items[0], all_items[i],
    #                                                                                        0, NULL)
    #        secure_counter += 1
    #        total += 1
    # ->
    mergesort(ranked_list, total+1)
    #with gil:
    #    print([(ranked_list[i*2], ranked_list[i*2+1]) for i in range(size_of_user_items+non_relevant_count)])
    result = measure.nogil_measure(ranked_list, (total+1) if k <= 0 else (k+1))
    free(ranked_list)
    return result

###################################################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
cdef list evaluate_model_only_threading(NOGILModel factor_model, int size_of_users, int *size_of_user_items,
                                        int **c_grouped, list gil_measures, int size_of_items, int *c_all_items,
                                        int non_relevant_count, int k):
    """
    Evaluate using multi thread for scoring but not measure
    :param factor_model:
    :param grouped:
    :param nogil_measures:
    :param all_items:
    :param non_relevant_count:
    :param k:
    :return:
    """
    # TODO -> is not fixed
    if len(gil_measures) == 0:
        return []
    cdef int i, j, n_measures = len(gil_measures), n = size_of_users * n_measures, c_nrc
    cdef dict measures = {}
    cdef float fi
    cdef NOGILMeasure m
    cdef list result = []
    with nogil, parallel():
        for i in prange(n, schedule="guided"):
            with gil:
                m = gil_measures[i / size_of_users]
            j =  i % size_of_users
            c_nrc = \
                non_relevant_count if size_of_items - non_relevant_count >= size_of_user_items[j] else size_of_items - size_of_user_items[j]
            fi = measure_model_nogil(factor_model, size_of_user_items[j], c_grouped[j], m, size_of_items,
                                    c_all_items, c_nrc, k)


            with gil:
                measures[i / size_of_users] = measures.get(i / size_of_users, 0.) + fi
    for i in range(len(measures)):
        result.append(measures[i] / size_of_users)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
cdef float measure_model_nogil(NOGILModel factor_model, int size_of_user_items, int *user_items, measure,
                               int size_of_all_items, int *all_items, int non_relevant_count, int k) nogil:
    """
    Evaluate some user according some measure using full nogil threading
    :param factor_model:
    :param size_of_user_items:
    :param user_items:
    :param measure:
    :param size_of_all_items:
    :param all_items:
    :param non_relevant_count:
    :param k:
    :return:
    """
    # TODO -> is not fixed
    cdef int i, j, secure_counter = 0, total = 0, \
        total_of_items = (non_relevant_count + size_of_user_items) if non_relevant_count else size_of_all_items
    cdef float *ranked_list = <float *>malloc(sizeof(float) * total_of_items * 2)
    cdef float result
    if ranked_list is NULL:
        return -1.
    for i in range(size_of_all_items):
        if total >= total_of_items:
            break
        if is_in(all_items[i], size_of_user_items, user_items):
            ranked_list[total+2], ranked_list[total*2+1] = 1., factor_model.nogil_get_score(user_items[0], all_items[i],
                                                                                            0, NULL)
            total += 1
        elif secure_counter < non_relevant_count:
            ranked_list[total*2], ranked_list[total*2+1] = 0., factor_model.nogil_get_score(user_items[0], all_items[i],
                                                                                            0, NULL)
            secure_counter += 1
            total += 1
    mergesort(ranked_list, total)
    with gil:
        result = measure.measure([(bool(ranked_list[i*2]), ranked_list[i*2+1]) for i in range(total_of_items)],
                                                                                        total if k <= 0 else k)
    free(ranked_list)
    return result