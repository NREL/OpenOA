import copy
import tempfile
import unittest
from pathlib import Path

import yaml
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from openoa import PlantData
from openoa.schema import ANALYSIS_REQUIREMENTS, ReanalysisMetaData
from openoa.schema.schema import create_schema, create_analysis_schema


from test.conftest import project_ENGIE, example_data_path_str  # isort: skip


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
        ) = project_ENGIE.prepare(
            path=example_data_path_str, return_value="dataframes", use_cleansed=False
        )

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

        # Ensure that after validating a non-None analysis type, that the None is removed
        assert None not in self.plant.analysis_type

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

        # Get the OpenOA standardized column names where the default isn't used
        scada_original = set((v for k, v in self.plant.metadata.scada.col_map.items() if k != v))
        assert len(scada_original.intersection(self.plant.scada.columns)) == 0

        meter_original = set((v for k, v in self.plant.metadata.meter.col_map.items() if k != v))
        assert len(meter_original.intersection(self.plant.meter.columns)) == 0

        asset_original = set((v for k, v in self.plant.metadata.asset.col_map.items() if k != v))
        assert len(asset_original.intersection(self.plant.asset.columns)) == 0

        curtail_original = set(
            (v for k, v in self.plant.metadata.curtail.col_map.items() if k != v)
        )
        assert len(curtail_original.intersection(self.plant.curtail.columns)) == 0

        for name in self.plant.reanalysis:
            re_original = set(
                (v for k, v in self.plant.metadata.reanalysis[name].col_map.items() if k != v)
            )
            assert len(re_original.intersection(self.plant.reanalysis[name].columns)) == 0

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
            asset=f"{data_path}/asset.csv",
            reanalysis={
                "era5": f"{data_path}/reanalysis_era5.csv",
                "merra2": f"{data_path}/reanalysis_merra2.csv",
            },
        )

        assert_frame_equal(
            self.plant.scada,
            plant_loaded.scada,
            "SCADA dataframe did not survive CSV save/loading process",
        )
        assert_frame_equal(
            self.plant.meter,
            plant_loaded.meter,
            "Meter dataframe did not survive CSV save/loading process",
        )
        assert_frame_equal(
            self.plant.curtail,
            plant_loaded.curtail,
            "Curtail dataframe did not survive CSV save/loading process",
        )
        assert_frame_equal(
            self.plant.asset,
            plant_loaded.asset,
            "Asset dataframe did not survive CSV save/loading process",
        )
        assert_frame_equal(
            self.plant.reanalysis["era5"],
            plant_loaded.reanalysis["era5"],
            "ERA5 dataframe did not survive CSV save/loading process",
        )
        assert_frame_equal(
            self.plant.reanalysis["merra2"],
            plant_loaded.reanalysis["merra2"],
            "MERRA2 dataframe did not survive CSV save/loading process",
        )


class TestPlantDatPartial(unittest.TestCase):
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
        ) = project_ENGIE.prepare(
            path=example_data_path_str, return_value="dataframes", use_cleansed=False
        )

    def setUp(self):
        """
        Create the plantdata object
        """
        with open(example_data_path_str + "/../plant_meta.yml", "r") as f:
            meta_partial = yaml.safe_load(f)
        meta_partial.pop("reanalysis")
        self.plant = PlantData(
            analysis_type=None,  # No validation desired at this point in time
            metadata=meta_partial,
            scada=self.scada_df,
            meter=self.meter_df,
            curtail=self.curtail_df,
            asset=self.asset_df,
        )

    def test_reanalysis_defaults(self):
        assert self.plant.metadata.reanalysis == {"product": ReanalysisMetaData()}

    def test_reanalysis_missing_metadata(self):
        """Tests that when there are missing products in the reanalysis metadata, that
        a KeyError is raised early.
        """
        with open(example_data_path_str + "/../plant_meta.yml", "r") as f:
            metadata = yaml.safe_load(f)

        # Raised when all missing
        meta_partial = copy.deepcopy(metadata)
        meta_partial.pop("reanalysis")
        with self.assertRaises(KeyError):
            PlantData(scada=self.scada_df, meter=self.meter_df, reanalysis=self.reanalysis_dict)

        # Raised when there is only some missing
        meta_partial = copy.deepcopy(metadata)
        meta_partial["reanalysis"].pop("merra2")
        with self.assertRaises(KeyError):
            PlantData(scada=self.scada_df, meter=self.meter_df, reanalysis=self.reanalysis_dict)


class TestSchema(unittest.TestCase):
    def setUp(self):
        schema_path = Path(__file__).resolve().parents[2] / "openoa/schema"

        with open(schema_path / "full_schema.yml", "r") as f:
            self.full_schema = yaml.safe_load(f)

        with open(schema_path / "base_electrical_losses_schema.yml", "r") as f:
            self.el_schema = yaml.safe_load(f)

        with open(schema_path / "base_monte_carlo_aep_schema.yml", "r") as f:
            self.mc_aep_schema = yaml.safe_load(f)

        with open(schema_path / "base_tie_schema.yml", "r") as f:
            self.tie_schema = yaml.safe_load(f)

        with open(schema_path / "scada_wake_losses_schema.yml", "r") as f:
            self.wake_schema = yaml.safe_load(f)

    def test_full_schema(self):
        full_schema = create_schema()
        assert self.full_schema == full_schema

    def test_analysis_schemas(self):
        # A direct comparison is not possible because the frequency ordering is different
        # between the two dictionaries, so only check for matching required data types

        el_schema = create_analysis_schema("ElectricalLosses")
        assert self.el_schema.keys() == el_schema.keys()
        for key, dict in el_schema.items():
            # Check that the correct required columns are pulled
            assert self.el_schema[key].keys() == dict.keys()
            # Check for matching frequencies
            assert set(dict["frequency"]) == set(self.el_schema[key]["frequency"])

        mc_aep_schema = create_analysis_schema("MonteCarloAEP")
        assert self.mc_aep_schema.keys() == mc_aep_schema.keys()
        for key, dict in mc_aep_schema.items():
            # Check that the correct required columns are pulled
            assert self.mc_aep_schema[key].keys() == dict.keys()
            # Check for matching frequencies
            assert set(dict["frequency"]) == set(self.mc_aep_schema[key]["frequency"])

        tie_schema = create_analysis_schema("TurbineLongTermGrossEnergy")
        assert self.tie_schema.keys() == tie_schema.keys()
        for key, dict in tie_schema.items():
            # Check that the correct required columns are pulled
            assert self.tie_schema[key].keys() == dict.keys()
            # Check for matching frequencies
            if key == "asset":
                assert "frequency" not in self.tie_schema[key]
                continue
            assert set(dict["frequency"]) == set(self.tie_schema[key]["frequency"])

        wake_schema = create_analysis_schema("WakeLosses-scada")
        assert self.wake_schema.keys() == wake_schema.keys()
        for key, dict in wake_schema.items():
            # Check that the correct required columns are pulled
            assert self.wake_schema[key].keys() == dict.keys()
            # Check for matching frequencies
            if key == "asset":
                assert "frequency" not in self.wake_schema[key]
                continue
            assert set(dict["frequency"]) == set(self.wake_schema[key]["frequency"])

    def test_combined_schema(self):
        analysis_types = [
            "ElectricalLosses",
            "MonteCarloAEP",
            "TurbineLongTermGrossEnergy",
            "WakeLosses-scada",
            "StaticYawMisalignment",
        ]
        combined_schema = create_analysis_schema(analysis_types=analysis_types)

        correct_schema = {
            "scada": {
                "asset_id": {"name": "asset_id", "dtype": "str", "units": None},
                "WTUR_W": {"name": "WTUR_W", "dtype": "float", "units": "kW"},
                "WMET_HorWdSpd": {"name": "WMET_HorWdSpd", "dtype": "float", "units": "m/s"},
                "WMET_HorWdDir": {"name": "WMET_HorWdDir", "dtype": "float", "units": "m/s"},
                "WMET_HorWdDirRel": {"name": "WMET_HorWdDirRel", "dtype": "float", "units": "deg"},
                "WROT_BlPthAngVal": {"name": "WROT_BlPthAngVal", "dtype": "float", "units": "deg"},
                "frequency": [
                    "h",
                    "min",
                    "s",
                    "ms",
                    "us",
                    "ns",
                ],
            },
            "reanalysis": {
                "WMETR_HorWdSpd": {"name": "WMETR_HorWdSpd", "dtype": "float", "units": "m/s"},
                "WMETR_HorWdDir": {"name": "WMETR_HorWdDir", "dtype": "float", "units": "deg"},
                "WMETR_AirDen": {"name": "WMETR_AirDen", "dtype": "float", "units": "kg/m^3"},
                "frequency": [
                    "h",
                    "min",
                    "s",
                    "ms",
                    "us",
                    "ns",
                ],
            },
            "meter": {
                "MMTR_SupWh": {"name": "MMTR_SupWh", "dtype": "float", "units": "kWh"},
                "frequency": ["min", "MS", "ME", "D", "ns", "W", "us", "min", "s", "h", "ms"],
            },
            "curtail": {
                "IAVL_ExtPwrDnWh": {"name": "IAVL_ExtPwrDnWh", "dtype": "float", "units": "kWh"},
                "IAVL_DnWh": {"name": "IAVL_DnWh", "dtype": "float", "units": "kWh"},
                "frequency": ["min", "MS", "ME", "D", "ns", "W", "us", "min", "s", "h", "ms"],
            },
            "asset": {
                "latitude": {"name": "latitude", "dtype": "float", "units": "WGS84"},
                "longitude": {"name": "longitude", "dtype": "float", "units": "WGS84"},
                "rated_power": {"name": "rated_power", "dtype": "float", "units": "kW"},
            },
        }

        # A direct comparison is not possible because the frequency ordering is different
        # between the two dictionaries.
        # Check for matching required data types
        assert correct_schema.keys() == combined_schema.keys()
        for key, dict in combined_schema.items():
            # Check that the correct required columns are pulled
            assert correct_schema[key].keys() == dict.keys()
            # Check for matching frequencies
            if key == "asset":
                assert "frequenct" not in correct_schema[key]
                continue
            assert set(dict["frequency"]) == set(correct_schema[key]["frequency"])
