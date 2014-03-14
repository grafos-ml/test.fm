# -*- coding: utf-8 -*-
"""
Created on 23 January 2014

Load models from a stdin or the file.

.. moduleauthor:: Linas
"""

__author__ = 'linas'


import re
from testfm.models.fm_loaded import FactorModel
non_decimal = re.compile(r'[^\d.]+')


class Load_Okapi(object):

    def get_model(self, file_path):
        """
        Loads the model from okapi style file.
        >>> from pkg_resources import resource_filename
        >>> import testfm
        >>> path = resource_filename(testfm.__name__, "data/okapi.tsv")
        >>> okapi = Load_Okapi()
        >>> model = okapi.get_model(path)
        >>> len(model._users)
        4
        >>> len(model._items)
        2
        >>> model.getScore(30331, 6731)
        0.11887863463700001

        @param file_path: str that shows the path to load the file
        @return:
        """
        data = self._parse_file(file_path)

        # 0 for user factors and 1 for item factors
        factors = {0: {}, 1: {}}
        for node_type, (data_key, data_value) in data:
            try:
                factors[node_type][data_key] = data_value
            except KeyError:
                pass

        fm = FactorModel(userf=factors[0], itemf=factors[1])
        return fm

    def _parse_file(self, file_path):
        """
        parses file into list of tuples with the model
        >>> from pkg_resources import resource_filename
        >>> import testfm
        >>> path = resource_filename(testfm.__name__, "data/okapi.tsv")
        >>> okapi = Load_Okapi()
        >>> data = okapi._parse_file(path)
        >>> len(data)
        6

        @param file_path: path to the file
        @return:
        """
        f = open(file_path, 'r')
        return [self._parse_line(line) for line in f]

    def _parse_line(self, line):
        """
        Parses line of format
        6523 0	[0.838237; 0.508268]
        2066 1	[0.973420; 0.453210]
        28462 0	[0.025059; 0.737217]
        The first number is the id of user or item, second is the type of node,
        then tab and then factors.

        :param line: str containing a string with a model
        :return: (0, (6523, [0.838237, 0.508268]) a tuple (nodetype, (id: factors)), where nodetype 0=user, 1=item, \
        factors is list of floats.
        """
        node, model = line.split('\t')
        node_id, node_type = node.split()
        return int(node_type), (int(node_id), [float(non_decimal.sub('', f))
                                               for f in model.split(";")])

if __name__ == '__main__':

    import doctest
    doctest.testmod()
