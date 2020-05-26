import unittest

import numpy as np
import numpy.testing as npt

from operational_analysis.methods.turbine_long_term_gross_energy import TurbineLongTermGrossEnergy
from examples.turbine_analysis.turbine_project import TurbineExampleProject


class TestPandasPrufPlantAnalysis(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Set up data to use for testing (TurbineExampleProject)
        self.project = TurbineExampleProject('./examples/turbine_analysis/data')
        self.project.prepare()
        
        self.analysis = TurbineLongTermGrossEnergy(self.project, UQ = False)
        self.analysis.run(reanal_subset = ['erai', 'merra2', 'ncep2'], enable_plotting = False)

        self.analysis_uq = TurbineLongTermGrossEnergy(self.project, UQ = True, num_sim = 100)
        self.analysis_uq.run(reanal_subset = ['erai', 'merra2', 'ncep2'], enable_plotting = False)

    def test_longterm_gross_energy_results(self):

        # Test not-UQ case
        res = self.analysis._plant_gross
        check = np.ones_like(res)*1.35
        npt.assert_almost_equal(res/1000000, check, decimal=1)

        # Test not-UQ case, mean value
        res_uq = self.analysis_uq._plant_gross.mean()
        check_uq = np.ones_like(res)*1.35
        npt.assert_almost_equal(res_uq/1000000, check_uq, decimal=1)

        # Test not-UQ case, stdev
        res_std_uq = self.analysis_uq._plant_gross.std()
        check_std_uq = 0.05
        npt.assert_almost_equal(res_std_uq/1000000, check_std_uq, decimal=1)
        
    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
