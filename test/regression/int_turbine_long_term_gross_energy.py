import random
import unittest
from test import example_data_path_str

import numpy as np
import numpy.testing as npt
from examples.project_ENGIE import Project_Engie

from operational_analysis.methods.turbine_long_term_gross_energy import TurbineLongTermGrossEnergy


def reset_prng():
    np.random.seed(42)
    random.seed(42)

class TestLongTermGrossEnergy(unittest.TestCase):
    def setUp(self):
        reset_prng()
        # Set up data to use for testing (ENGIE data)
        self.project = Project_Engie(example_data_path_str)
        self.project.prepare()

        self.analysis = TurbineLongTermGrossEnergy(self.project, UQ=False)

        self.analysis.run(
            reanal_subset=["era5", "merra2"],
            max_power_filter=0.85,
            wind_bin_thresh=1.0,
            correction_threshold=0.9,
            enable_plotting=False,
        )

    def test_longterm_gross_energy_results(self):
        reset_prng()
        # Test not-UQ case, mean value
        res = self.analysis._plant_gross.mean()
        check = 12.84266859
        npt.assert_almost_equal(res / 1e6, check)

    def tearDown(self):
        pass


class TestLongTermGrossEnergyUQ(unittest.TestCase):
    def setUp(self):
        reset_prng()
        # Set up data to use for testing (TurbineExampleProject)
        self.project = Project_Engie(example_data_path_str)
        self.project.prepare()

        self.analysis_uq = TurbineLongTermGrossEnergy(self.project, UQ=True, num_sim=5)
        self.analysis_uq.run(enable_plotting=False, reanal_subset=["era5", "merra2"])

    def test_longterm_gross_energy_results(self):
        reset_prng()
        # Test UQ case, mean value
        res_uq = self.analysis_uq._plant_gross.mean()
        check_uq = 13.550325
        npt.assert_almost_equal(res_uq / 1e6, check_uq)

        # Test UQ case, stdev
        res_std_uq = self.analysis_uq._plant_gross.std()
        check_std_uq = 0.13636826
        npt.assert_almost_equal(res_std_uq / 1e6, check_std_uq)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
