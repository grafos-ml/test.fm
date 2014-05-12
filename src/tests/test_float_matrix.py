"""
Module for nose testing the float matrix developed in C
"""
__author__ = "joaonrb"

from testfm.models.cutil.float_matrix import FloatMatrix


class TestObjectOperations:
    """
    Test the object operations, such as get and set
    """

    @staticmethod
    def test_set_and_get():
        """
        Test set and get a float from the matrix
        """
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a = FloatMatrix(3, 3)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                a[i, j] = value  # Try to set value to some row and column
        # Is expected that the matrix is like:
        #   1.0 2.0 3.0
        #   4.0 5.0 6.0
        #   7.0 8.0 9.0
        assert a[0, 0] == 1. and isinstance(a[0, 0], float), "Element (0, 0) is not 1 or not a float"
        assert a[0, 1] == 2. and isinstance(a[0, 1], float), "Element (0, 1) is not 2 or not a float"
        assert a[0, 2] == 3. and isinstance(a[0, 2], float), "Element (0, 2) is not 3 or not a float"
        assert a[1, 0] == 4. and isinstance(a[1, 0], float), "Element (1, 0) is not 4 or not a float"
        assert a[1, 1] == 5. and isinstance(a[1, 1], float), "Element (1, 1) is not 5 or not a float"
        assert a[1, 2] == 6. and isinstance(a[1, 2], float), "Element (1, 2) is not 6 or not a float"
        assert a[2, 0] == 7. and isinstance(a[2, 0], float), "Element (2, 0) is not 7 or not a float"
        assert a[2, 1] == 8. and isinstance(a[2, 1], float), "Element (2, 1) is not 8 or not a float"
        assert a[2, 2] == 9. and isinstance(a[2, 2], float), "Element (2, 2) is not 9 or not a float"



class TestBasicOperations:
    """
    Test for the basic operations in the matrix
    """

    @staticmethod
    def test_transpose():
        """
        Test the tranpose fo th matrix
        :return:
        """
        pass