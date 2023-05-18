"""Methods to generate YAML and JSON schema files"""

from __future__ import annotations

import json
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
)


HERE = Path(__file__).resolve().parent


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
    meta_classes = (
        SCADAMetaData,
        MeterMetaData,
        TowerMetaData,
        StatusMetaData,
        CurtailMetaData,
        AssetMetaData,
        ReanalysisMetaData,
    )
    schema = {}
    for meta in meta_classes:
        cls = meta()
        meta_dict = asdict(cls, filter=_attrs_meta_filter, value_serializer=_attrs_meta_serializer)
        schema[cls.name] = {"""broken out by column into name, dtype, and units"""}
        for key, value in meta_dict:
            if key in ("dtypes", "units", "frequency"):
                continue
            schema[cls.name][key] = {
                "name": meta_dict[key],
                "dtype": meta_dict["dtypes"][key],
                "units": meta_dict["units"][key],
            }
            schema[cls.name].pop("units")
            schema[cls.name].pop("dtypes")
    return schema


def create_analysis_schema(analysis_types: str | list[str]) -> dict:
    """Creates a dictionary of the metadata input requirements.

    Returns:
        dict: The compiled metadata dictionary specifying the required data definitions.
    """
    meta_classes = (
        SCADAMetaData,
        MeterMetaData,
        TowerMetaData,
        StatusMetaData,
        CurtailMetaData,
        AssetMetaData,
        ReanalysisMetaData,
    )
    schema = {}
    for meta in meta_classes:
        cls = meta()
        meta_dict = asdict(cls, filter=_attrs_meta_filter, value_serializer=_attrs_meta_serializer)
        schema[cls.name] = {"""broken out by column into name, dtype, and units"""}
    return schema


# TODO: bring metadata components of plant.py into schema/
# TODO: replace ID naming to asset_id


if __name__ == "__main__":
    # Get the schemas
    full_schema = create_schema()

    # Save the analysis schemass
    with open(HERE / "full_schema.yml", "w") as f:
        yaml.dump(full_schema, f, default_flow_style=False, sort_keys=False)

    with open(HERE / "full_schema.json", "w") as f:
        json.dump(full_schema, f, sort_keys=False, indent=2)
