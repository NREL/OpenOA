# Schema Reference

This is a quick reference guide to navigate the JSON and YAML data specifications for
what types of data, and their units, are expected to be used for various analysis types. These
specifications don't have to match exactly to your data, but are here to indicate the default
values of the internal data schema, and demonstrate the expectations of input data.

Each of the keys (e.g., scada) represents a field for `PlantData` with each nested dictionary
representing the following:

- name: the default column name
- dtype: the data type `PlantData will convert that column to, if it's not already of that type
- units: the expected units of the input data (there are no checks for this)

As an example for the first column entry and its associated frequency in the full schema,

```yaml
scada:
  time:
    name: time
    dtype: numpy.datetime64
    units: datetim64[ns]
  ...
  frequency: 10T
```

the expected matching input for this column would be the following to indicate in the SCADA data
that the time column is still the default value, and that the frequency is still the default value
of 10 minutes.

```python
metadata = {"scada": {"time": "time", "frequency": "10T", ...}}
```

For a less trivial example, we can bring in our data's actual column naming convention by simply
indicating that the time column in the SCADA data should be mapped from "datetime_mst".
Additionally, we're indicate that our data have an hourly frequency.

```python
metadata = {"scada": {"time": "datetime_mst", "frequency": "H", ...}}
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

The minimum data specification for the base Monte Carlo AEP analysis. More robust
schema is on the way for the variations on the base analysis.

## Base Turbine Ideal Energy

[JSON](base_tie_schema.json) [YAML](base_tie_schema.yml)

The minimum data specification for the base turbine ideal energy analysis. More robust
schema is on the way for the variations on the base analysis.

## Base Wake Losses

[JSON](base_wake_losses_schema.json) [YAML](base_wake_losses_schema.yml)

The minimum data specification for the base wake losses analysis. More robust
schema is on the way for the variations on the base analysis.
