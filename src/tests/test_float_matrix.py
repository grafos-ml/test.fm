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
    def test_size_of_symmetric():
        """
        [FloatMatrix Basic Operations] Test number of rows and columns in a symmetric matrix
        """
        a = FloatMatrix(3, 3)
        assert a.rows == 3, "Number of rows(%d) should be 3" % a.rows
        assert a.columns == 3, "Number of columns(%d) should be 3" % a.columns


    @staticmethod
    def test_set_and_get_in_symmetric():
        """
        [FloatMatrix Basic Operations] Test set and get a float from the symmetric matrix
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
        assert a[0, 0] == 1. and isinstance(a[0, 0], float), "Element (0, 0) is not 1.0 or not a float"
        assert a[0, 1] == 2. and isinstance(a[0, 1], float), "Element (0, 1) is not 2.0 or not a float"
        assert a[0, 2] == 3. and isinstance(a[0, 2], float), "Element (0, 2) is not 3.0 or not a float"
        assert a[1, 0] == 4. and isinstance(a[1, 0], float), "Element (1, 0) is not 4.0 or not a float"
        assert a[1, 1] == 5. and isinstance(a[1, 1], float), "Element (1, 1) is not 5.0 or not a float"
        assert a[1, 2] == 6. and isinstance(a[1, 2], float), "Element (1, 2) is not 6.0 or not a float"
        assert a[2, 0] == 7. and isinstance(a[2, 0], float), "Element (2, 0) is not 7.0 or not a float"
        assert a[2, 1] == 8. and isinstance(a[2, 1], float), "Element (2, 1) is not 8.0 or not a float"
        assert a[2, 2] == 9. and isinstance(a[2, 2], float), "Element (2, 2) is not 9.0 or not a float"

    @staticmethod
    def test_set_and_get_out_of_bounds_in_symmetric():
        """
        [FloatMatrix Basic Operations] Test set and get out of the matrix bounds
        """
        a = FloatMatrix(3, 3)
        try:
            c = a[3, 2]
        except IndexError:
            pass
        else:
            assert True, "Allow out of bounds for rows (%f)" % c

        try:
            c = a[2, 3]
        except IndexError:
            pass
        else:
            assert True, "Allow out of bounds for columns (%f)" % c
        try:
            a[3, 2] = 3
        except IndexError:
            pass
        else:
            assert True, "Allow set in out of bounds for rows"

        try:
            a[2, 3] = 3
        except IndexError:
            pass
        else:
            assert True, "Allow set out of bounds for columns"

    @staticmethod
    def test_size_of_non_symmetric():
        """
        [FloatMatrix Basic Operations] Test number of rows and columns in a symmetric matrix
        """
        a = FloatMatrix(3, 2)
        assert a.rows == 3, "Number of rows(%d) should be 3" % a.rows
        assert a.columns == 2, "Number of columns(%d) should be 2" % a.columns


    @staticmethod
    def test_set_and_get_in_non_symmetric():
        """
        [FloatMatrix Basic Operations] Test set and get a float from the symmetric matrix
        """
        matrix = [[1, 2], [3, 4], [5, 6]]
        a = FloatMatrix(3, 2)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                a[i, j] = value  # Try to set value to some row and column
        # Is expected that the matrix is like:
        #   1.0 2.0
        #   3.0 4.0
        #   5.0 6.0
        assert a[0, 0] == 1. and isinstance(a[0, 0], float), "Element (0, 0) is not 1.0 or not a float"
        assert a[0, 1] == 2. and isinstance(a[0, 1], float), "Element (0, 1) is not 2.0 or not a float"
        assert a[1, 0] == 3. and isinstance(a[1, 0], float), "Element (1, 0) is not 3.0 or not a float"
        assert a[1, 1] == 4. and isinstance(a[1, 1], float), "Element (1, 1) is not 4.0 or not a float"
        assert a[2, 0] == 5. and isinstance(a[2, 0], float), "Element (2, 0) is not 5.0 or not a float"
        assert a[2, 1] == 6. and isinstance(a[2, 1], float), "Element (1, 1) is not 6.0 or not a float"

    @staticmethod
    def test_set_and_get_out_of_bounds_in_symmetric():
        """
        [FloatMatrix Basic Operations] Test set and get out of the matrix bounds
        """
        a = FloatMatrix(3, 2)
        try:
            c = a[3, 1]
        except IndexError:
            pass
        else:
            assert True, "Allow out of bounds for rows (%f)" % c

        try:
            c = a[1, 3]
        except IndexError:
            pass
        else:
            assert True, "Allow out of bounds for columns (%f)" % c
        try:
            a[3, 1] = 3
        except IndexError:
            pass
        else:
            assert True, "Allow set in out of bounds for rows"

        try:
            a[1, 3] = 3
        except IndexError:
            pass
        else:
            assert True, "Allow set out of bounds for columns"

    @staticmethod
    def test_clone():
        """
        [FloatMatrix Basic Operations] Test the matrix is cloned
        """
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a = FloatMatrix(3, 3)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                a[i, j] = value  # Try to set value to some row and column
        cloned_a = a.clone()
        b = cloned_a.element_wise_multiplication(a)
        assert b[0, 0] == a[0, 0] ** 2, "Element (0, 0) is not working correctly"
        assert b[0, 1] == a[0, 1] ** 2, "Element (0, 1) is not working correctly"
        assert b[0, 2] == a[0, 2] ** 2, "Element (0, 2) is not working correctly"
        assert b[1, 0] == a[1, 0] ** 2, "Element (1, 0) is not working correctly"
        assert b[1, 1] == a[1, 1] ** 2, "Element (1, 1) is not working correctly"
        assert b[1, 2] == a[1, 2] ** 2, "Element (1, 2) is not working correctly"
        assert b[2, 0] == a[2, 0] ** 2, "Element (2, 0) is not working correctly"
        assert b[2, 1] == a[2, 1] ** 2, "Element (2, 1) is not working correctly"
        assert b[2, 2] == a[2, 2] ** 2, "Element (2, 2) is not working correctly"


class TestMatrixOperations:
    """
    Test for the basic operations in the matrix
    """

    @staticmethod
    def test_transpose():
        """
        [FloatMatrix Matrix Operations] Test the transpose for the matrix
        """
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a = FloatMatrix(3, 3)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                a[i, j] = value  # Try to set value to some row and column
        # Is expected that the matrix a is like:
        #   1.0 2.0 3.0
        #   4.0 5.0 6.0
        #   7.0 8.0 9.0
        b = a.transpose()
        # Is expected that the matrix b is like:
        #   1.0 4.0 7.0
        #   2.0 5.0 8.0
        #   3.0 6.0 9.0
        assert b.is_transpose, "B is not considering itself transpose"
        assert b[0, 0] == 1., "Element (0, 0) is not 1.0"
        assert b[0, 1] == 4., "Element (0, 1) is not 4.0"
        assert b[0, 2] == 7., "Element (0, 2) is not 7.0"
        assert b[1, 0] == 2., "Element (1, 0) is not 2.0"
        assert b[1, 1] == 5., "Element (1, 1) is not 5.0"
        assert b[1, 2] == 8., "Element (1, 2) is not 8.0"
        assert b[2, 0] == 3., "Element (2, 0) is not 3.0"
        assert b[2, 1] == 6., "Element (2, 1) is not 6.0"
        assert b[2, 2] == 9., "Element (2, 2) is not 9.0"