import unittest

import numpy as np
import numpy.testing as npt

from operational_analysis.methods.eya_gap_analysis import EYAGapAnalysis


class EYAGAPAnalysis(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Set up operational results data
        oa_data = [448.0, 0.0493, 0.012, 477.8]
        # AEP (GWh/yr), availability loss (fraction), electrical loss (fraction), turbine ideal energy (GWh/yr)

        # Set up EYA estimates
        eya_data = [467.0, 597.14, 0.062, 0.024, 0.037, 0.011, 0.087]
        # AEP (GWh/yr), gross energy (GWh/yr), availability loss (fraction), electrical loss (fraction),
        # turbine performance loss (fraction), blade degradation loss (fraction), wake loss (fraction)

        # Creat gap analysis method object and run
        self.analysis = EYAGapAnalysis(
            plant="NA", eya_estimates=eya_data, oa_results=oa_data, make_fig=False
        )
        self.analysis.run()

    def test_eya_gap_analysis_results(self):

        # Check that the compiled gap analysis results match as expected
        expected_compiled_data = [467.0, -41.441648, 6.59437, 6.230899, 9.61638]
        actual_compiled_data = self.analysis._compiled_data
        npt.assert_array_almost_equal(expected_compiled_data, actual_compiled_data, decimal=3)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
