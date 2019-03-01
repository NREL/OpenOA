import unittest

import numpy as np

from operational_analysis.methods.turbine_long_term_gross_energy import TurbineLongTermGrossEnergy
from examples.turbine_analysis.turbine_project import TurbineExampleProject

#import pdb


class TestPandasPrufPlantAnalysis(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

        # Set up data to use for testing (TurbineExampleProject)
        self.project = TurbineExampleProject('./examples/turbine_analysis/data')
        self.project.prepare()

        #pdb.set_trace()

        self.analysis = TurbineLongTermGrossEnergy(self.project)
        self.analysis.run()


    def test_longterm_gross_energy_results(self):
        pass


    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
