"""Methods to generate YAML and JSON schema files"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any
from pathlib import Path

import yaml
from attrs import Attribute, asdict

from openoa.schema.metadata import (
    AssetMetaData,
    MeterMetaData,
    SCADAMetaData,
    TowerMetaData,
    StatusMetaData,
    CurtailMetaData,
    ReanalysisMetaData,
    determine_analysis_requirements,
)


HERE = Path(__file__).resolve().parent

meta_class_map = {
    "scada": SCADAMetaData,
    "meter": MeterMetaData,
    "tower": TowerMetaData,
    "status": StatusMetaData,
    "curtail": CurtailMetaData,
    "asset": AssetMetaData,
    "reanalysis": ReanalysisMetaData,
}


def _attrs_meta_filter(inst: Attribute, value: Any) -> bool:
    """Filters out any unnecessary components of the metadata classes.

    Args:
        inst (Attribute): The class attribute (``attrs.field``) being checked
        value (Any): The actual values in the field.

    Returns:
        bool: False, if should not be serialized, and True, if it should be serialized.
    """
    if inst.name in ("col_map", "name", "col_map_reversed"):
        return False
    if inst is None or value is None:
        return False
    return True


def _attrs_meta_serializer(inst: type, field: Attribute, value: Any) -> Any:
    """Custom serialization for attrs dataclass fields.

    Args:
        inst (type): The class object.
        field (Attribute): The ``attrs.field`` information.
        value (Any): The actual values in the :py:attr:`field`.

    Returns:
        Any: Reformatted data.
    """
    if field is None:
        return value
    if field.name == "dtypes":
        value = {k: str(v).replace("<class '", "").replace("'>", "") for k, v in value.items()}
        return value
    return value


def create_schema() -> dict:
    """Creates a dictionary of the metadata input requirements.

    Returns:
        dict: The compiled metadata dictionary specifying the required data definitions.
    """
    schema = {name: {} for name in meta_class_map}
    for name, meta in meta_class_map.items():
        meta_dict = asdict(
            meta(), filter=_attrs_meta_filter, value_serializer=_attrs_meta_serializer
        )
        for key, value in meta_dict.items():
            if key in ("dtypes", "units"):
                continue
            if key == "frequency":
                schema[name][key] = meta_dict[key]
                continue
            schema[name][key] = {
                "name": meta_dict[key],
                "dtype": meta_dict["dtypes"][key],
                "units": meta_dict["units"][key],
            }
    return schema


def create_analysis_schema(analysis_types: str | list[str]) -> dict:
    """Creates a dictionary of the metadata input requirements.

    Returns:
        dict: The compiled metadata dictionary specifying the required data definitions.
    """
    schema = create_schema()
    schema_copy = deepcopy(schema)
    column_requirements, frequency_requirements = determine_analysis_requirements(
        which="both", analysis_type=analysis_types
    )
    for name, meta in schema_copy.items():
        if name not in column_requirements:
            schema.pop(name)
            continue
        for col in meta:
            if col == "frequency":
                schema[name][col] = list(frequency_requirements[name])
                continue
            if col not in column_requirements[name]:
                schema[name].pop(col)
    return schema


if __name__ == "__main__":
    # Get the schemas
    full_schema = create_schema()
    base_mc_aep_schema = create_analysis_schema("MonteCarloAEP")
    temp_mc_aep_schema = create_analysis_schema("MonteCarloAEP-temp")
    wd_mc_aep_schema = create_analysis_schema("MonteCarloAEP-wd")
    temp_wd_mc_aep_schema = create_analysis_schema("MonteCarloAEP-temp-wd")
    scada_wake_schema = create_analysis_schema("WakeLosses-scada")
    tower_wake_schema = create_analysis_schema("WakeLosses-tower")
    base_tie_schema = create_analysis_schema("TurbineLongTermGrossEnergy")
    base_electric_schema = create_analysis_schema("ElectricalLosses")
    base_yaw_misalignment_schema = create_analysis_schema("StaticYawMisalignment")

    # Save the analysis schemass
    with open(HERE / "full_schema.yml", "w") as f:
        yaml.dump(full_schema, f, default_flow_style=False, sort_keys=False)
    with open(HERE / "full_schema.json", "w") as f:
        json.dump(full_schema, f, sort_keys=False, indent=2)

    with open(HERE / "base_monte_carlo_aep_schema.yml", "w") as f:
        yaml.dump(base_mc_aep_schema, f, default_flow_style=False, sort_keys=False)
    with open(HERE / "base_monte_carlo_aep_schema.json", "w") as f:
        json.dump(base_mc_aep_schema, f, sort_keys=False, indent=2)

    with open(HERE / "temperature_monte_carlo_aep_schema.yml", "w") as f:
        yaml.dump(temp_mc_aep_schema, f, default_flow_style=False, sort_keys=False)
    with open(HERE / "temperature_monte_carlo_aep_schema.json", "w") as f:
        json.dump(temp_mc_aep_schema, f, sort_keys=False, indent=2)

    with open(HERE / "temperature_wind_direction_monte_carlo_aep_schema.yml", "w") as f:
        yaml.dump(temp_wd_mc_aep_schema, f, default_flow_style=False, sort_keys=False)
    with open(HERE / "temperature_wind_direction_monte_carlo_aep_schema.json", "w") as f:
        json.dump(temp_wd_mc_aep_schema, f, sort_keys=False, indent=2)

    with open(HERE / "wind_direction_monte_carlo_aep_schema.yml", "w") as f:
        yaml.dump(wd_mc_aep_schema, f, default_flow_style=False, sort_keys=False)
    with open(HERE / "wind_direction_monte_carlo_aep_schema.json", "w") as f:
        json.dump(wd_mc_aep_schema, f, sort_keys=False, indent=2)

    with open(HERE / "scada_wake_losses_schema.yml", "w") as f:
        yaml.dump(scada_wake_schema, f, default_flow_style=False, sort_keys=False)
    with open(HERE / "scada_wake_losses_schema.json", "w") as f:
        json.dump(scada_wake_schema, f, sort_keys=False, indent=2)

    with open(HERE / "tower_wake_losses_schema.yml", "w") as f:
        yaml.dump(tower_wake_schema, f, default_flow_style=False, sort_keys=False)
    with open(HERE / "tower_wake_losses_schema.json", "w") as f:
        json.dump(tower_wake_schema, f, sort_keys=False, indent=2)

    with open(HERE / "base_tie_schema.yml", "w") as f:
        yaml.dump(base_tie_schema, f, default_flow_style=False, sort_keys=False)
    with open(HERE / "base_tie_schema.json", "w") as f:
        json.dump(base_tie_schema, f, sort_keys=False, indent=2)

    with open(HERE / "base_electrical_losses_schema.yml", "w") as f:
        yaml.dump(base_electric_schema, f, default_flow_style=False, sort_keys=False)
    with open(HERE / "base_electrical_losses_schema.json", "w") as f:
        json.dump(base_electric_schema, f, sort_keys=False, indent=2)

    with open(HERE / "base_yaw_misalignmental_losses_schema.yml", "w") as f:
        yaml.dump(base_yaw_misalignment_schema, f, default_flow_style=False, sort_keys=False)
    with open(HERE / "base_yaw_misalignmental_losses_schema.json", "w") as f:
        json.dump(base_yaw_misalignment_schema, f, sort_keys=False, indent=2)
