import pandas as pd
import pytest

from operational_analysis.toolkits import reanalysis_downloading as rd


def test_get_dataset_names():
    dataset_names = {"merra2": "nasa_merra2_global", "era5": "ecmwf_era5_v2"}

    assert rd._get_dataset_names("merra2") == dataset_names["merra2"]
    assert rd._get_dataset_names("era5") == dataset_names["era5"]

    # Check that minor typos are handled
    assert rd._get_dataset_names("MERRA2") == dataset_names["merra2"]
    assert rd._get_dataset_names(" era5 ") == dataset_names["era5"]

    # Check that invalid dataset names are caught
    with pytest.raises(KeyError):
        rd._get_dataset_names("ERAI")


def test_default_var_dicts_planetos():
    var_dict_merra2 = {"U50M": "u_ms", "V50M": "v_ms", "T2M": "temperature_K", "PS": "surf_pres_Pa"}

    var_dict_era5 = {
        "eastward_wind_at_100_metres": "u_ms",
        "northward_wind_at_100_metres": "v_ms",
        "air_temperature_at_2_metres": "temperature_K",
        "surface_air_pressure": "surf_pres_Pa",
    }

    assert rd._get_default_var_dicts_planetos("merra2") == var_dict_merra2
    assert rd._get_default_var_dicts_planetos("era5") == var_dict_era5

    # Check that minor typos are handled
    assert rd._get_default_var_dicts_planetos("MERRA2") == var_dict_merra2
    assert rd._get_default_var_dicts_planetos(" era5 ") == var_dict_era5

    # Check that invalid dataset names are caught
    with pytest.raises(ValueError):
        rd._get_default_var_dicts_planetos("ERAI")


def test_get_start_end_dates_planetos():
    # Define data set start and end dates from PlanetOS (actual values as of November 2021)
    start_date_ds_merra2 = pd.to_datetime("1980-01-01 00:30:00")
    end_date_ds_merra2 = pd.to_datetime("2021-08-31 23:30:00")

    start_date_ds_era5 = pd.to_datetime("1979-01-01 00:00:00")
    end_date_ds_era5 = pd.to_datetime("2021-10-28 18:00:00")

    # Number of years of data to download
    num_years = 20

    # First test when start and end dates are both defined for merra2
    # Test dates that are within bounds. The end date should have 1 hour added to it.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date="1980-01-01 00:30:00",
        end_date="2021-08-31 23:30",
        num_years=num_years,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("1980-01-01 00:30"),
        pd.to_datetime("2021-09-01 00:30"),
    )

    # Test end date that is out of bounds. The end date should be the data set end date with 1 hour added to it.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date="1980-01-01 00:30:00",
        end_date="2021-08-31 23:31",
        num_years=num_years,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("1980-01-01 00:30"),
        pd.to_datetime("2021-09-01 00:30"),
    )

    # Test start date that is out of bounds.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date="1980-01-01 00:29",
        end_date="2021-08-31 23:30",
        num_years=num_years,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("1980-01-01 00:30"),
        pd.to_datetime("2021-09-01 00:30"),
    )

    # Test start date that is out of bounds. Since the minute values of the new start date changes, the end date should
    # be adjusted to match the new start date minute value so that an integer number of hours is still requested.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date="1980-01-01 00:00",
        end_date="2021-08-31 23:00",
        num_years=num_years,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("1980-01-01 00:30"),
        pd.to_datetime("2021-09-01 00:30"),
    )

    # Test start and end dates that are out of bounds. The end date should be the data set end date with 1 hour added
    # to it.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date="1980-01-01 00:29:00",
        end_date="2021-09-01 00:00",
        num_years=num_years,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("1980-01-01 00:30"),
        pd.to_datetime("2021-09-01 00:30"),
    )

    # Now test when a start date is defined, but the end date is undefined for merra2
    # Test dates that are within bounds. The end date should be 20 years after the start date. Even though 20 years
    # after the start date is after the data set end date, it is still allowed because the last time specified is not
    # actually retrieved from the data set.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date="2001-09-01 00:30",
        end_date=None,
        num_years=20,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("2001-09-01 00:30"),
        pd.to_datetime("2021-09-01 00:30"),
    )

    # Test end date that is out of bounds. The end date should be the data set end date with 1 hour added to it.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date="2001-09-01 01:00",
        end_date=None,
        num_years=20,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("2001-09-01 01:00"),
        pd.to_datetime("2021-09-01 00:30"),
    )

    # Test start date that is out of bounds. The start and end dates should be the data set start date and 20 years
    # after the data set start date.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date="1980-01-01 00:29",
        end_date=None,
        num_years=20,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("1980-01-01 00:30"),
        pd.to_datetime("2000-01-01 00:30"),
    )

    # Test start and end dates that are out of bounds. The start and end dates should be the data set start date and 1
    # hour after the data set end date.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date="1980-01-01 00:29",
        end_date=None,
        num_years=42,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("1980-01-01 00:30"),
        pd.to_datetime("2021-09-01 00:30"),
    )

    # Now test when an end date is defined, but the start date is undefined for merra2
    # Test dates that are within bounds. The end date should be 1 hour after the specified end date and the start date
    # should be 20 years before that.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date=None,
        end_date="2021-08-31 23:30",
        num_years=20,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("2001-09-01 00:30"),
        pd.to_datetime("2021-09-01 00:30"),
    )

    # Test end date that is out of bounds. The end date should be the data set end date with 1 hour added to it and the
    # start date should be 20 years before that.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date=None,
        end_date="2021-09-01 00:00",
        num_years=20,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("2001-09-01 00:30"),
        pd.to_datetime("2021-09-01 00:30"),
    )

    # Test start date that is out of bounds. The start and end dates should be the data set start date and 20 years
    # after the data set start date.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date=None,
        end_date="1999-12-31 23:00",
        num_years=20,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("1980-01-01 00:30"),
        pd.to_datetime("2000-01-01 00:30"),
    )

    # Test start and end dates that are out of bounds. The start and end dates should be the data set start date and 1
    # hour after the data set end date.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date=None,
        end_date="2021-09-01 00:00",
        num_years=42,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("1980-01-01 00:30"),
        pd.to_datetime("2021-09-01 00:30"),
    )

    # Now test when neither the start or end dates are defined for era5.
    # First test when start date is in bounds. The end date should be 1 hour after the end of the last full month in
    # the data set and the start date should be 20 years before that.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date=None,
        end_date=None,
        num_years=num_years,
        start_date_ds=start_date_ds_era5,
        end_date_ds=end_date_ds_era5,
    )
    assert start_end_dates_new == (
        pd.to_datetime("2001-10-01 00:00"),
        pd.to_datetime("2021-10-01 00:00"),
    )

    # Now test when start date is out of bounds. The end date should be 1 hour after the end of the last full month in
    # the data set and the start date should be the start date of the data set.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date=None,
        end_date=None,
        num_years=43,
        start_date_ds=start_date_ds_era5,
        end_date_ds=end_date_ds_era5,
    )
    assert start_end_dates_new == (
        pd.to_datetime("1979-01-01 00:00"),
        pd.to_datetime("2021-10-01 00:00"),
    )

    # Now test when start date is in bounds for era5. The end date should be 1 hour after the end of the last full
    # month in the data set and the start date should be 20 years before that. Since the end date of the era5 data set
    # is the end of a full month, the end date should be the start of the next month.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date=None,
        end_date=None,
        num_years=num_years,
        start_date_ds=start_date_ds_merra2,
        end_date_ds=end_date_ds_merra2,
    )
    assert start_end_dates_new == (
        pd.to_datetime("2001-09-01 00:00"),
        pd.to_datetime("2021-09-01 00:00"),
    )

    # Last, test when start or end dates are on February 29 of a leap year for era5.
    # First, test an end date that is Feb. 29 of a leap year with a start date 5 years prior. The new start date should
    # be automatically changed to Feb. 28 of that year since it is not a leap year.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date=None,
        end_date="2016-02-29 12:00",
        num_years=5,
        start_date_ds=start_date_ds_era5,
        end_date_ds=end_date_ds_era5,
    )
    assert start_end_dates_new == (
        pd.to_datetime("2011-02-28 13:00"),
        pd.to_datetime("2016-02-29 13:00"),
    )

    # Now test a start date that is Feb. 29 of a leap year with an end date 5 years after. The new end date should be
    # automatically changed to Feb. 28 of that year since it is not a leap year.
    start_end_dates_new = rd._get_start_end_dates_planetos(
        start_date="2016-02-29 12:00",
        end_date=None,
        num_years=5,
        start_date_ds=start_date_ds_era5,
        end_date_ds=end_date_ds_era5,
    )
    assert start_end_dates_new == (
        pd.to_datetime("2016-02-29 12:00"),
        pd.to_datetime("2021-02-28 12:00"),
    )
