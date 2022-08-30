import tempfile
import unittest
from pathlib import Path

from examples import project_ENGIE
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from openoa import PlantData
from openoa.plant import ANALYSIS_REQUIREMENTS


example_data_path = Path(__file__).parents[2].resolve() / "examples" / "data" / "la_haute_borne"
example_data_path_str = str(example_data_path)


class TestPlantData(unittest.TestCase):
    """
    TestPlantData

    Tests the construction, validation, and methods of PlantData using La Haute Born wind plant data
    """

    @classmethod
    def setUpClass(cls):
        (
            cls.scada_df,
            cls.meter_df,
            cls.curtail_df,
            cls.asset_df,
            cls.reanalysis_dict,
        ) = project_ENGIE.prepare(path=example_data_path_str, return_value="dataframes")

    def setUp(self):
        """
        Create the plantdata object
        """
        self.plant = PlantData(
            analysis_type=None,  # No validation desired at this point in time
            metadata=example_data_path_str + "/../plant_meta.yml",
            scada=self.scada_df,
            meter=self.meter_df,
            curtail=self.curtail_df,
            asset=self.asset_df,
            reanalysis=self.reanalysis_dict,
        )

    def test_analysis_type_values(self):
        """
        Test the acceptance of the valid inputs, except None
        """
        valid = [*ANALYSIS_REQUIREMENTS] + ["all", None]

        self.plant.analysis_type = valid
        self.assertTrue(self.plant.analysis_type == valid)

        # Test a couple of edge cases to show that only valid inputs are allowed
        # Add a mispelling
        with self.assertRaises(ValueError):
            self.plant.analysis_type = "Montecarloaep"

        # Add a completely wrong value
        with self.assertRaises(ValueError):
            self.plant.analysis_type = "this is wrong"

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

    def test_update_columns(self):
        """
        Tests that the column names are successfully mapped to the standardized names.
        """
        # Put the plant analysis type back in working order
        self.plant.analysis_type = "MonteCarloAEP"
        self.plant.validate()

        assert_array_equal(
            self.plant.scada.columns.values, self.plant.metadata.scada.col_map.keys()
        )
        assert_array_equal(
            self.plant.meter.columns.values, self.plant.metadata.meter.col_map.keys()
        )
        assert_array_equal(
            self.plant.asset.columns.values, self.plant.metadata.asset.col_map.keys()
        )
        assert_array_equal(
            self.plant.curtail.columns.values, self.plant.metadata.curtail.col_map.keys()
        )
        for name in self.plant.reanalysis:
            assert_array_equal(
                self.plant.reanalysis[name].columns.values,
                self.plant.metadata.reanalysis[name].col_map.keys(),
            )

    def test_toCSV(self):
        """
        Save this plant to a temporary directory, load it in, and make sure the data matches.
        """
        # Save
        data_path = tempfile.mkdtemp()
        self.plant.to_csv(save_path=data_path, with_openoa_col_names=True)

        # Load
        plant_loaded = PlantData(
            metadata=f"{data_path}/metadata.yml",
            scada=f"{data_path}/scada.csv",
            meter=f"{data_path}/meter.csv",
            curtail=f"{data_path}/curtail.csv",
            status=f"{data_path}/scada.csv",
            asset=f"{data_path}/asset.csv",
            reanalysis={
                "era5": f"{data_path}/reanalysis_era5.csv",
                "merra2": f"{data_path}/reanalysis_merra2.csv",
            },
        )

        assert_frame_equal(
            self.plant.scada,
            plant_loaded.scada,
            "Scada dataframe did not survive CSV save/loading process",
        )
        # TODO, add more comprehensive checking.
        # - Check metadata
        # - Check all DFs equal
