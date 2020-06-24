import unittest

import numpy as np
import numpy.testing as npt

from operational_analysis.methods.turbine_long_term_gross_energy import TurbineLongTermGrossEnergy
from examples.project_ENGIE import Project_Engie

class TestLongTermGrossEnergy(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Set up data to use for testing (ENGIE data)
        self.project = Project_Engie('./examples/data/la_haute_borne')
        self.project.prepare()
        
        self.analysis = TurbineLongTermGrossEnergy(self.project, UQ = False)
        
        self.analysis.run(reanal_subset = ['era5', 'merra2'],
                                                   max_power_filter = 0.85,
                                                   wind_bin_thresh = 1.,
                                                   correction_threshold = 0.9,
                                                   enable_plotting = False)

    def test_longterm_gross_energy_results(self):

        # Test not-UQ case
        res = self.analysis._plant_gross
        check = np.ones_like(res)*13.7
        npt.assert_almost_equal(res/1e6, check, decimal=1)
        
    def tearDown(self):
        pass


class TestLongTermGrossEnergyUQ(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Set up data to use for testing (TurbineExampleProject)
        self.project = Project_Engie('./examples/data/la_haute_borne')
        self.project.prepare()

        self.analysis_uq = TurbineLongTermGrossEnergy(self.project, UQ = True, num_sim = 100)
        self.analysis_uq.run(enable_plotting = False, reanal_subset = ['era5', 'merra2'])

    def test_longterm_gross_energy_results(self):

        # Test UQ case, mean value
        res_uq = self.analysis_uq._plant_gross.mean()
        check_uq = 13.7
        npt.assert_almost_equal(res_uq/1e6, check_uq, decimal=1)

        # Test UQ case, stdev
        res_std_uq = self.analysis_uq._plant_gross.std()
        check_std_uq = 0.10
        npt.assert_almost_equal(res_std_uq/1e6, check_std_uq, decimal=1)
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
