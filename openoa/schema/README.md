# Schema Reference

This is a quick reference guide to navigate the JSON and YAML data specifications for
what types of data, and their units, are expected to be used for various analysis types. These
specifications don't have to match exactly to your data, but are here to indicate the default
values of the internal data schema, and demonstrate the expectations of input data.

Each of the keys (e.g., scada) represents a field for `PlantData` with each nested dictionary
representing the following:

- name: the default column name
- dtype: the data type `PlantData` will convert that column to, if it's not already of that type
- units: the expected units of the input data (there are no checks for this)

As an example for the first column entry and its associated frequency in the full schema,

```yaml
scada:
  time:
    name: time
    dtype: numpy.datetime64
    units: datetim64[ns]
  ...
  frequency: 10min
```

the expected matching input for this column would be the following to indicate in the SCADA data
that the time column is still the default value, and that the frequency is still the default value
of 10 minutes.

```python
metadata = {"scada": {"time": "time", "frequency": "10min", ...}}
```

For a less trivial example, we can bring in our data's actual column naming convention by simply
indicating that the time column in the SCADA data should be mapped from "datetime_mst".
Additionally, we're indicate that our data have an hourly frequency.

```python
metadata = {"scada": {"time": "datetime_mst", "frequency": "h", ...}}
```

## Full Schema

[JSON](full_schema.json) [YAML](full_schema.yml)

The complete data specification of any data that might be expected to be accessed
throughout the OpenOA codebase.

## Base Electrical Losses

[JSON](base_electrical_losses_schema.json) [YAML](base_electrical_losses_schema.yml)

The minimum data specification to be able to run an electrical losses analysis.

## Base Monte Carlo AEP

[JSON](base_monte_carlo_aep_schema.json) [YAML](base_monte_carlo_aep_schema.yml)

The minimum data specification for the base Monte Carlo AEP analysis.

### Monte Carlo AEP with Temperature

[JSON](temperature_monte_carlo_aep_schema.json) [YAML](temperature_monte_carlo_aep_schema.yml)

Used for when `reg_temperature=True`.

### Monte Carlo AEP with Wind Direction

[JSON](wind_direction_monte_carlo_aep_schema.json) [YAML](wind_direction_monte_carlo_aep_schema.yml)

Used for when `reg_wind_direction=True`.

### Monte Carlo AEP with Temperature and Wind Direction

[JSON](temperature_wind_direction_monte_carlo_aep_schema.json) [YAML](temperature_wind_direction_monte_carlo_aep_schema.yml)

Used for when `reg_temperature=True` and `reg_wind_direction=True`.

## Base Turbine Ideal Energy

[JSON](base_tie_schema.json) [YAML](base_tie_schema.yml)

The minimum data specification for the base turbine ideal energy analysis.

## Wake Losses

### Wake Losses using SCADA Wind Variables

[JSON](scada_wake_losses_schema.json) [YAML](scada_wake_losses_schema.yml)

The default wake loss schema, using SCADA windspeed and wind direction inputs.

### Wake Losses using Tower Wind Variables

[JSON](tower_wake_losses_schema.json) [YAML](tower_wake_losses_schema.yml)

The modified wake loss schema, using met tower data for windspeed and wind direction inputs.

## Base Static Yaw Misalignment

[JSON](base_yaw_misalignmental_losses_schema.json) [YAML](base_yaw_misalignmental_losses_schema.yml)

The minimum data specification for the base static yaw misalignment analysis.
