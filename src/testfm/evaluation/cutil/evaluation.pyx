from random import sample

cdef float measure(int **ranked_list, int list_size) nogil:
    cdef float map_measure = 0.
    cdef float relevant = 0.
    cdef int i
    for i in range(list_size):
        if ranked_list[i][0] != 0:
            relevant += 1.
            map_measure += relevant / (i + 1)
    return 0.0 if relevant == 0 else map_measure/relevant

cdef partial_measure(user, entries, factor_model, all_items, non_relevant_count, measure, k=None):
    if non_relevant_count is None:
        # Add all items except relevant
        ranked_list = [(False, factor_model.get_score(user, nr)) for nr in all_items if nr not in entries['item']]
    else:
        #2. inject #non_relevant random items
        ranked_list = [(False, factor_model.get_score(user, nr)) for nr in sample(all_items, non_relevant_count)]
        #2. add all relevant items from the testing_data
    ranked_list += [(True, factor_model.get_score(user, i)) for i in entries['item']]

        #shuffle(ranked_list)  # Just to make sure we don't introduce any bias (AK: do we need this?)

    #number of relevant items
    n = entries['item'].size
    #5. sort according to the score
    ranked_list.sort(key=lambda x: x[1], reverse=True)

    #6. evaluate according to each measure
    return measure.measure(ranked_list[:k], n=n)