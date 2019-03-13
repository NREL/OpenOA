import unittest
import numpy as np
import pandas as pd
from numpy import testing as nptest

from operational_analysis.toolkits.machine_learning_setup import *
from operational_analysis.toolkits.power_curve import logistic5param


class TestMLToolkit(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

        self.x1 = pd.Series(np.random.random(1000) * 30)
        self.x2 = pd.Series(np.random.random(1000) * 1)

        # Simple test case with a 1d power curve as a noisy logistic 5 param
        params = [1300, -7, 11, 2, 0.5]
        noise = 0.1
        self.y_1d = pd.Series(logistic5param(self.x1, *params) + np.random.random(1000) * noise)

        # 2D test case where the 1d logistic is scaled by a 2nd variable.
        self.y_2d = self.y_1d * self.x2

    def test_etr(self):
        # Create test data using logistic5param form
        #ml = MachineLearningSetup("etr")

        #ml.hyper_optimize(self.x1, self.y_1d)
        #ml.hyper_optimize(self.x1.values.reshape(1, -1), self.y_1d)

        #print ml.opt_hyp

        #y_pred = ml.random_search.predict(self.x1)

        #nptest.assert_allclose(self.y_1d, y_pred, rtol=1, atol=noise * 2, err_msg="Power curve did not properly fit.")
        pass


    def tearDown(self):
        pass
