"""Quality control methods to help users check their data for irregularities."""

from abc import abstractmethod
from typing import List, Tuple, Union
from datetime import time, datetime

import pytz
import h5pyd
import numpy as np
import pandas as pd
import dateutil
import matplotlib.pyplot as plt
from pyproj import Proj
from dateutil import tz

from openoa import logging, logged_method_call
from openoa.utils import timeseries


Number = Union[int, float]
logger = logging.getLogger(__name__)
