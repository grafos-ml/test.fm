__author__ = 'mumas'

import unittest
import pandas as pd
import numpy as np
from math import sqrt

from testfm.evaluation.evaluator import Evaluator
from testfm.models.baseline_model import ConstantModel
class TestEvaluator(unittest.TestCase):

    def test_rmse(self):
        eval = Evaluator()

        model = ConstantModel(constant=3.0)
        testing = pd.DataFrame([{'user':10, 'item':100, 'rating':3},
                                {'user':10,'item':110,'rating':1},
                                {'user':12,'item':100,'rating':4},
        ])
        rmse = eval.evaluate_model_rmse(model, testing)
        self.assertEqual(sqrt((0+4+1)/3.0), rmse)
        eval.close()
