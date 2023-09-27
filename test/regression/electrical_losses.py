import unittest

import numpy as np
import numpy.testing as npt

from openoa.analysis.electrical_losses import ElectricalLosses


from test.conftest import project_ENGIE, example_data_path_str  # isort: skip


class TestElectricalLosses(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Set up data to use for testing (ENGIE data)
        self.project = project_ENGIE.prepare(example_data_path_str, use_cleansed=False)
        self.project.analysis_type.append("ElectricalLosses")
        self.project.validate()

        # Create electrical loss method object and run
        # NO UQ case
        self.analysis = ElectricalLosses(
            self.project, UQ=False, uncertainty_correction_threshold=0.95
        )
        self.analysis.run()

    def testelectrical_losses_results(self):
        # Check that the computed electrical loss means and their std are as expected
        expected_losses = 0.02
        expected_losses_std = 0
        actual_compiled_data = self.analysis.electrical_losses.mean()
        actual_compiled_data_std = self.analysis.electrical_losses.std()

        npt.assert_array_almost_equal(expected_losses, actual_compiled_data, decimal=3)
        npt.assert_array_almost_equal(expected_losses_std, actual_compiled_data_std, decimal=3)

    def tearDown(self):
        pass


class TestElectricalLossesUQ(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Set up data to use for testing (ENGIE data)
        self.project = project_ENGIE.prepare(example_data_path_str, use_cleansed=False)
        self.project.analysis_type.append("ElectricalLosses")
        self.project.validate()

        # Create electrical loss method object and run
        # WITH UQ
        self.analysis_uq = ElectricalLosses(
            self.project, UQ=True, num_sim=3000, uncertainty_correction_threshold=(0.9, 0.995)
        )
        self.analysis_uq.run()

    def testelectrical_losses_results(self):
        # Check that the computed electrical loss means and their std are as expected
        expected_losses_uq = 0.02
        expected_losses_uq_std = 0.0069
        actual_compiled_data_uq = self.analysis_uq.electrical_losses.mean()
        actual_compiled_data_uq_std = self.analysis_uq.electrical_losses.std()

        npt.assert_array_almost_equal(expected_losses_uq, actual_compiled_data_uq, decimal=3)
        npt.assert_array_almost_equal(
            expected_losses_uq_std, actual_compiled_data_uq_std, decimal=3
        )

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
