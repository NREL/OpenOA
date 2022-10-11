import random
import unittest

import numpy as np
import numpy.testing as npt
from examples import project_ENGIE, example_data_path_str

from openoa.analysis import TurbineLongTermGrossEnergy


def reset_prng():
    np.random.seed(42)
    random.seed(42)


class TestLongTermGrossEnergy(unittest.TestCase):
    def setUp(self):
        reset_prng()

        # Set up data to use for testing (ENGIE data)
        self.project = project_ENGIE.prepare(example_data_path_str)
        self.project.analysis_type.append("TurbineLongTermGrossEnergy")
        self.project.validate()

        self.analysis = TurbineLongTermGrossEnergy(
            self.project,
            UQ=False,
            max_power_filter=0.85,
            wind_bin_threshold=1.0,
            correction_threshold=0.9,
        )

        self.analysis.run(reanalysis_subset=["era5", "merra2"])

    def test_longterm_gross_energy_results(self):
        reset_prng()
        # Test not-UQ case, mean value
        res = self.analysis.plant_gross.mean()
        check = 12.84266859
        npt.assert_almost_equal(res / 1e6, check)

    def tearDown(self):
        pass


class TestLongTermGrossEnergyUQ(unittest.TestCase):
    def setUp(self):
        reset_prng()

        # Set up data to use for testing (TurbineExampleProject)
        self.project = project_ENGIE.prepare(example_data_path_str)
        self.project.analysis_type.append("TurbineLongTermGrossEnergy")
        self.project.validate()

        self.analysis_uq = TurbineLongTermGrossEnergy(self.project, UQ=True, num_sim=5)
        self.analysis_uq.run(reanalysis_subset=["era5", "merra2"])

    def test_longterm_gross_energy_results(self):
        reset_prng()

        # Test UQ case, mean value
        res_uq = self.analysis_uq.plant_gross.mean()
        check_uq = 13.550325
        npt.assert_almost_equal(res_uq / 1e6, check_uq)

        # Test UQ case, stdev
        res_std_uq = self.analysis_uq.plant_gross.std()
        check_std_uq = 0.13636826
        npt.assert_almost_equal(res_std_uq / 1e6, check_std_uq)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
