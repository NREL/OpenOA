import unittest

import numpy as np
import numpy.testing as npt

from operational_analysis.methods.electrical_losses import ElectricalLosses
from examples.turbine_analysis.turbine_project import TurbineExampleProject

class TestElectricalLosses(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        
        # Set up data to use for testing (TurbineExampleProject)
        self.project = TurbineExampleProject('./examples/turbine_analysis/data')
        self.project.prepare()
        
        # Synthesize two turbines from example turbine data and create new scada dataframe
        scada_1_df = self.project._scada.df[['id', 'energy_kwh']]
        scada_temp = self.project._scada.df
        scada_temp['id'] = 'T1'
        scada_temp['energy_kwh'] = scada_1_df['energy_kwh'] * np.random.randint(80, 120, scada_1_df.shape[0])/100
        scada_2_df = scada_temp[['id', 'energy_kwh']]

        self.project._scada.df = scada_1_df.append(scada_2_df)
        self.project._num_turbines = 2
        
        # Synthesize metered data from scada dataframe
        meter_df = scada_1_df['energy_kwh'].to_frame()
        meter_df['energy_kwh'] = meter_df['energy_kwh'] * 2 * np.random.randint(98, 100, scada_1_df.shape[0])/100
        self.project._meter.df = meter_df
        self.project._meter_freq = '10T'
                
        # Creat electrical loss method object and run
        self.analysis = ElectricalLosses(self.project)
        self.analysis.run()

    def test_electrical_losses_results(self):
        
        # Check that the computed electrical losses are as expected
        expected_losses = 0.0123478
        actual_compiled_data = self.analysis._electrical_losses
        npt.assert_array_almost_equal(expected_losses, actual_compiled_data, decimal = 4)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
