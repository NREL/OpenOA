import sys
from pathlib import Path

import pytest


examples_folder = Path(__file__).resolve().parents[1]
sys.path.append(examples_folder)
from examples import project_ENGIE, example_data_path_str  # noqa: disable=E402


ROOT = Path(__file__).parent


def pytest_addoption(parser):
    parser.addoption("--unit", action="store_true", default=False, help="run tests in test/unit/.")
    parser.addoption(
        "--regression", action="store_true", default=False, help="run tests in test/regression/."
    )


def pytest_configure(config):
    # Check for the options
    unit = config.getoption("--unit")
    regression = config.getoption("--regression")

    # Provide the appropriate directories
    regression_tests = list((ROOT / "regression").iterdir())
    unit_tests = list((ROOT / "unit").iterdir())

    # If both or neither, run them all, otherwise run just the appropriate subset
    if regression and unit or (not regression and not unit):
        config.args = unit_tests + regression_tests
    elif regression:
        config.args = regression_tests
    elif unit:
        config.args = unit_tests
