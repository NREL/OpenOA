import unittest

from examples import project_ENGIE
from openoa import PlantData
from test import example_data_path_str


class TestPlantData(unittest.TestCase):
    """
    TestPlantData

    Tests the construction, validation, and methods of PlantData using La Haute Born wind plant data
    """

    @classmethod
    def setUpClass(cls):
        cls.scada_df, cls.meter_df, cls.curtail_df, cls.asset_df, cls.reanalysis_dict = project_ENGIE.prepare(
            path=example_data_path_str,
            return_value="dataframes"
        )

    def setUp(self):
        """
        Create the plantdata object
        """
        self.plant = PlantData(
            analysis_type=None,  # No validation desired at this point in time
            metadata=example_data_path_str+"/../plant_meta.yml",
            scada=self.scada_df,
            meter=self.meter_df,
            curtail=self.curtail_df,
            asset=self.asset_df,
            reanalysis=self.reanalysis_dict,
        )

    def test_validatePlantForAEP(self):
        """
        The example plant should validate for MonteCarloAEP analysis type
        """
        self.plant.analysis_type = "MonteCarloAEP"
        self.plant.validate()

    def test_doesNotValidateForAll(self):
        """
        The example plant should not validate for MonteCarloAEP analysis type
        """
        with self.assertRaises(ValueError):
            self.plant.analysis_type = "all"
            self.plant.validate()