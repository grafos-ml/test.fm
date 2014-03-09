__author__ = 'linas'

from pandas import DataFrame

def not_new_items_filter(training, testing):
    '''
    Given two data frames, it filters from the testing frame all the non-new items.
    A new item is the one that was not seen in the training set.

    >>> training = DataFrame({'user' : [1, 1, 1, 2], 'item' : [1, 2, 3, 4]})
    >>> testing = DataFrame({'user' : [1, 1, 1, 2], 'item' : [3, 1, 4, 4]})
    >>> testing = not_new_items_filter(training, testing)
    >>> len(testing)#only user 1 has item 4 not seen in training set of user 1
    1
    >>> int(testing['item'])
    4
    '''

    test_users = testing.user.unique()
    for u in test_users:
        training_items_of_u =  training[training.user == u].item.unique()
        testing = testing[(testing.user != u) | ((testing.user == u) & ~testing.item.isin(training_items_of_u))]
    return testing
if __name__ == '__main__':
    import doctest
    doctest.testmod()
