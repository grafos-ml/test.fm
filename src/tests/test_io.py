__author__ = 'linas'

import unittest
from testfm.fmio.load_model import Load_Okapi


class TestLSI(unittest.TestCase):

    def setUp(self):
        self.okapi = Load_Okapi()


    def test_parseline(self):
        """
        Using stupid syntax +"\t"+ just because pycharm can not handle tabs...
        @return:
        """
        l1 = self.okapi._parse_line("6523 0\t[0.838237; 0.508268]")
        self.assertEqual((0, (6523, [0.838237, 0.508268])), l1)
        l2 = self.okapi._parse_line("5794 1\t[0.090743; 0.334450]")
        self.assertEqual((1, (5794, [0.090743, 0.334450])), l2)

if __name__ == '__main__':
    unittest.main()
