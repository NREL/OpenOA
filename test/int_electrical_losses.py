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
        
        # SCADA DATA
        # Synthesize two turbines from example turbine data and create new scada dataframe
        # First turbine is from the real data
        scada_1_df = self.project._scada.df[['id', 'energy_kwh']]
        # Create second turbine
        scada_temp = self.project._scada.df
        scada_temp['id'] = 'T1'
        # Normal distribution centered on 1 -> second turbine has same average energy production as turbine 1
        scada_temp['energy_kwh'] = scada_1_df['energy_kwh'] * np.random.randint(80, 120, scada_1_df.shape[0])/100
        scada_2_df = scada_temp[['id', 'energy_kwh']]
        # Put two turbines together
        self.project._scada.df = scada_1_df.append(scada_2_df)
        self.project._num_turbines = 2
        
        # METER DATA
        # Synthesize metered data from scada dataframe
        meter_df = scada_1_df['energy_kwh'].to_frame()
        # Meter data as a fraction of twice (because there are 2 turbines) the scada data
        meter_df['energy_kwh'] = meter_df['energy_kwh'] * 2 * np.random.randint(98, 100, scada_1_df.shape[0])/100
        self.project._meter.df = meter_df
        self.project._meter_freq = '10T'

        # Create electrical loss method object and run
        # NO UQ case
        self.analysis = ElectricalLosses(self.project, UQ = 'N')
        self.analysis.run()

        # Create electrical loss method object and run
        # WITH UQ
        self.analysis_uq = ElectricalLosses(self.project, UQ = 'Y', num_sim = 3000)
        self.analysis_uq.run()

    def test_electrical_losses_results(self):
        
        # Check that the computed electrical loss means and their std are as expected
        expected_losses = 0.0126
        expected_losses_std = 0
        expected_losses_uq = 0.0126
        expected_losses_uq_std = 0.007
        actual_compiled_data = self.analysis._electrical_losses.mean()
        actual_compiled_data_std = self.analysis._electrical_losses.std()
        actual_compiled_data_uq = self.analysis_uq._electrical_losses.mean()
        actual_compiled_data_uq_std = self.analysis_uq._electrical_losses.std()
        
        npt.assert_array_almost_equal(expected_losses, actual_compiled_data, decimal = 3)
        npt.assert_array_almost_equal(expected_losses_std, actual_compiled_data_std, decimal = 3)
        npt.assert_array_almost_equal(expected_losses_uq, actual_compiled_data_uq, decimal = 3)
        npt.assert_array_almost_equal(expected_losses_uq_std, actual_compiled_data_uq_std, decimal = 3)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
