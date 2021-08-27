import sys
import unittest

import numpy as np
import pandas as pd
from numpy import testing as nptest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from operational_analysis.toolkits.machine_learning_setup import MachineLearningSetup


class TestMLToolkit(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

        # Create 2 random features, 1st to mimic wind speed, the other air density
        self.x1 = pd.Series(np.random.random(1000) * 30)  # Wind speed
        self.x2 = pd.Series(np.random.random(1000) * 360)  # Wind direction
        self.x3 = pd.Series(1 + np.random.random(1000) * 0.2)  # Air density

        # Group features together
        self.X = pd.DataFrame(data={"ws": self.x1, "wd": self.x2, "dens": self.x3})

        # Create simple power relationship with feature variables
        self.y = self.x3 * np.power(self.x1, 3) * np.log(self.x2) / 6  # Units of kW

    def test_algorithms(self):
        # Test different algorithms hyperoptimization and fitting results
        # Hyperparameter optimization is based on randomized grid search, so pass criteria is not stringent
        np.random.seed(42)

        # Specify expected mean power, R2 and RMSE from the fits
        required_metrics = {
            "etr": (0.999852, 130.0),
            "gbm": (0.999999, 30.0),
            "gam": (0.983174, 1330.0),
        }

        # Loop through algorithms
        for a in required_metrics.keys():
            ml = MachineLearningSetup(a)  # Setup ML object

            # Perform randomized grid search only once for efficiency
            ml.hyper_optimize(self.X, self.y, n_iter_search=1, report=False, cv=KFold(n_splits=2))

            # Predict power based on model results
            y_pred = ml.random_search.predict(self.X)

            # Compute performance metrics which we'll test
            corr = np.corrcoef(self.y, y_pred)[
                0, 1
            ]  # Correlation between predicted and actual power
            rmse = np.sqrt(
                mean_squared_error(self.y, y_pred)
            )  # RMSE between predicted and actual power

            # Mean power in GW is within 3 decimal places
            nptest.assert_approx_equal(
                self.y.sum() / 1e6,
                y_pred.sum() / 1e6,
                significant=3,
                err_msg="Sum of predicted and actual power for {} not close enough".format(a),
            )

            # Test correlation of model fit
            nptest.assert_approx_equal(
                corr,
                required_metrics[a][0],
                significant=4,
                err_msg="Correlation between {} features and response is wrong".format(a),
            )

            # Test RMSE of model fit
            self.assertLess(rmse, required_metrics[a][1], "RMSE of {} fit is too high".format(a))

    def tearDown(self):
        pass
