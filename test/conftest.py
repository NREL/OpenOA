import sys
from pathlib import Path


examples_folder = Path(__file__).resolve().parents[1]
sys.path.append(examples_folder)
from examples import project_ENGIE, example_data_path_str  # noqa: disable=E402
