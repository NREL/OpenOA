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
        self.analysis = TurbineLongTermGrossEnergy(self.project)
        self.analysis.run()

    def test_longterm_gross_energy_results(self):
        res = self.analysis._summary_results.values.astype("float")
        check = np.ones_like(res)*1.35
        npt.assert_almost_equal(res/1000000, check, decimal=1)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
