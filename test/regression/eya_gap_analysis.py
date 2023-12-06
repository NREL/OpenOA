import unittest

import numpy as np
import numpy.testing as npt

from openoa.analysis.eya_gap_analysis import EYAGapAnalysis


from test.conftest import project_ENGIE, example_data_path_str  # isort: skip


class EYAGAPAnalysis(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Set up operational results data
        oa_data = dict(
            aep=448.0,
            availability_losses=0.0493,
            electrical_losses=0.012,
            turbine_ideal_energy=477.8,
        )
        # AEP (GWh/yr), availability loss (fraction), electrical loss (fraction), turbine ideal energy (GWh/yr)

        # Set up EYA estimates
        eya_data = dict(
            aep=467.0,
            gross_energy=597.14,
            availability_losses=0.062,
            electrical_losses=0.024,
            turbine_losses=0.037,
            blade_degradation_losses=0.011,
            wake_losses=0.087,
        )

        self.project = project_ENGIE.prepare(example_data_path_str, use_cleansed=False)

        # AEP (GWh/yr), gross energy (GWh/yr), availability loss (fraction), electrical loss (fraction),
        # turbine performance loss (fraction), blade degradation loss (fraction), wake loss (fraction)

        # Creat gap analysis method object and run
        self.analysis = self.project.EYAGapAnalysis(
            eya_estimates=eya_data, oa_results=oa_data
        )  # make_fig=False)
        self.analysis.run()

    def test_eya_gap_analysis_results(self):
        # Check that the compiled gap analysis results match as expected
        expected_compiled_data = [467.0, -41.441648, 6.59437, 6.230899, 9.61638]
        actual_compiled_data = self.analysis.compiled_data
        npt.assert_array_almost_equal(expected_compiled_data, actual_compiled_data, decimal=3)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
