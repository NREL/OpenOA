# This class contains analytical routines for estimating static yaw misalignment for a list of
# specified wind turbines. For a series of wind speed bins, the yaw misalignment is estimated for
# each turbine by identifying the difference between the wind vane angle where power performance
# (either power or normalized coefficient of power) is maximized and the mean wind vane angle for
# the turbine. The wind vane angle where power is maximized is found by fitting a cosine exponent
# curve to the binned power performance values as a function of wind vane angle. The offset of the
# best-fit cosine curve is treated as the wind vane angle where power is maximized. In addition to
# static yaw misalignment estimates for each wind speed bin, the mean yaw misalignment value
# averaged over all wind speed bins is calculated.
#
# Many parts of this method are based on or inspired by the following publications:

# 1. Bao, Y., Yang, Q., Fu, L., Chen, Q., Cheng, C., and Sun, Y. Identification of Yaw Error
#    Inherent Misalignment for Wind Turbine Based on SCADA Data: A Data Mining Approach. Proc. 12th
#    Asian Control Conference (ASCC), Kitakyushu, Japan, June 9-12 (2019). 1095-1100.
# 2. Xue, J. and Wang, L. Online data-driven approach of yaw error estimation and correction of
#    horizontal axis wind turbine. *IET J. Eng.* 2019(18):4937–4940 (2019).
#    https://doi.org/10.1049/joe.2018.9293.
# 3. Astolfi, D., Castellani, F., and Terzi, L. An Operation Data-Based Method for the Diagnosis of
#    Zero-Point Shift of Wind Turbines Yaw Angle. *J. Solar Energy Engineering* 142(2):024501
#    (2020). https://doi.org/10.1115/1.4045081.
# 4. Jing, B., Qian, Z., Pei, Y., Zhang, L., and Yang, T. Improving wind turbine efficiency through
#    detection and calibration of yaw misalignment. *Renewable Energy* 160:1217-1227 (2020).
#    https://doi.org/10.1016/j.renene.2020.07.063.
# 5. Gao, L. and Hong, J. Data-driven yaw misalignment correction for utility-scale wind turbines.
#    *J. Renewable Sustainable Energy* 13(6):063302 (2021). https://doi.org/10.1063/5.0056671.

# WARNING: This is a relatively simple method that has not yet been validated using data from wind
# turbines with known static yaw misalignments. Therefore, the results should be treated with
# caution. One known issue is that the method currently relies on nacelle wind speed measurements
# to determine the power performance as a function of wind vane angle. If the measured wind speed
# is affected by the amount of yaw misalignment, potential biases can exist in the estimated static
# yaw misalignment values.

from __future__ import annotations

from copy import deepcopy

import attrs
import numpy as np
import pandas as pd
import numpy.typing as npt
from tqdm import tqdm
from attrs import field, define
from scipy.optimize import curve_fit

from openoa.plant import PlantData
from openoa.utils import plot, filters
from openoa.schema import FromDictMixin, ResetValuesMixin
from openoa.logging import logging, logged_method_call
from openoa.analysis._analysis_validators import validate_UQ_input, validate_half_closed_0_1_right


logger = logging.getLogger(__name__)
NDArrayFloat = npt.NDArray[np.float64]
plot.set_styling()


def cos_curve(x, A, Offset, cos_exp):
    """Computes a cosine exponent curve as a function of yaw misalignment for curve fitting.

    Args:
        x (:obj:`float`): The yaw misalignment input in degrees.
        A (:obj:`float`): The amplitude of the cosine exponent curve.
        Offset (:obj:`float`): The yaw misaligment offset at which the cosine exponent curve is
            maximized in degrees.
        cos_exp (:obj:`float`): The exponent to which the cosine curve is raised.
    Returns:
        :obj:`float`: The value of the cosine exponent curve for the provided yaw misalignment.
    """
    return A * np.cos((np.pi / 180) * (x - Offset)) ** cos_exp


@define(auto_attribs=True)
class StaticYawMisalignment(FromDictMixin, ResetValuesMixin):
    """
    A method for estimating static yaw misalignment for different wind speed bins for each specified
    wind turbine as well as the average static yaw misalignment over all wind speed bins using
    turbine-level SCADA data.

    The method is comprised of the following core steps, which are performed for each specified
    wind turbine. If :py:attr:`UQ` is selected, the following steps are performed multiple times
    using Monte Carlo simulation to produce a distribution of static yaw misalignment estimates
    from which 95% confidence intervals can be derived:

    1. Timestamps containing power curve outliers are removed. Specifically, pitch angles are
       limited to a specified threshold to remove timestamps when turbines are operating in or
       near above-rated conditions where yaw misalignment has little impact on power performance.
       Next to increase the likelihood that power performance deviations are caused by yaw
       misalignment, a power curve outlier detection filter is used to remove timestamps when the
       turbine is operating abnormally. If :py:attr:`UQ` is selected, power curve outlier
       detection parameters will be chosen randomly for each Monte Carlo iteration.
    2. The filtered SCADA data are divided into the specified wind speed bins based on wind speed
       measured by the nacelle anemometer. If :py:attr:`UQ` is selected, the data corresponding
       to each wind speed bin are randomly resampled with replacement each Monte Carlo iteration
       (i.e., bootstrapping).
    3. For each wind speed bin, the power performance is binned by wind vane angle, where power
       performance can be defined as the raw power or a normalized coefficient power formed by
       dividing the raw power by the wind speed cubed.
    4. A cosine exponent curve as a function of wind vane angle is fit to the binned power
       performance values, where the free parameters are the amplitude, the exponent applied to
       the cosine, and the wind vane angle offset where the peak of the cosine curve is located.
    5. For each wind speed bin, the static yaw misalignment is estimated as the difference
       between the wind vane angle where power performance is maximized, based on the wind vane
       angle offset for the best-fit cosine curve, and the mean wind vane angle.
    6. The overall yaw misalignment is estimated as the average yaw misalignment over all wind
       speed bins.

    .. warning:: This is a relatively simple method that has not yet been validated using data from
        wind turbines with known static yaw misalignments. Therefore, the results should be treated
        with caution. One known issue is that the method currently relies on nacelle wind speed
        measurements to determine the power performance as a function of wind vane angle. If the
        measured wind speed is affected by the amount of yaw misalignment, potential biases can
        exist in the estimated static yaw misalignment values.

    Args:
        plant (:obj:`PlantData`): A :py:attr:`openoa.plant.PlantData` object that has been validated
            with at least :py:attr:`openoa.plant.PlantData.analysis_type` = "StaticYawMisalignment".
        turbine_ids (:obj:`list`, optional): List of turbine IDs for which static yaw misalignment
            detection will be performed. If None, all turbines will be analyzed. Defaults to None.
        UQ (:obj:`bool`, optional): Dertermines whether to perform uncertainty quantification using
            Monte Carlo simulation (True) or provide a single yaw misalignment estimate (False).
            Defaults to True.
        num_sim (int, optional): Number of Monte Carlo iterations to perform. Only used if
            :py:attr:`UQ` = True. Defaults to 100.
        ws_bins (float, optional): Wind speed bin centers for which yaw misalignment detection
            will be performed (m/s). Defaults to [5.0, 6.0, 7.0, 8.0].
        ws_bin_width (float, optional): Wind speed bin size to use when detecting yaw
            misalignment for individual wind seed bins (m/s). Defaults to 1 m/s.
        vane_bin_width (float, optional): Wind vane bin size to use when detecting yaw
            misalignment (degrees). Defaults to 1 degree.
        min_vane_bin_count (int, optional): Minimum number of data points needed in a wind vane
            bin for it to be included when detecting yaw misalignment. Defaults to 100.
        max_abs_vane_angle (float, optional): Maximum absolute wind vane angle considered when
            detecting yaw misalignment. Defaults to 25 degrees.
        pitch_thresh (float, optional): Maximum blade pitch angle considered when detecting yaw
            misalignment. Defaults to 0.5 degrees.
        num_power_bins (int, optional): Number of power bins to use for power curve bin
            filtering to remove outlier data points. Defaults to 25.
        min_power_filter (float, optional): Minimum power threshold, defined as a fraction
            of rated power, to which the power curve bin filter should be applied. Defaults to
            0.01.
        max_power_filter (tuple | float, optional): Maximum power threshold, defined as a fraction
            of rated power, to which the power curve bin filter should be applied. This should be
            a tuple when :py:attr:`UQ` = True (values are Monte-Carlo sampled within the specified
            range) or a single value when :py:attr:`UQ` = False. If undefined (None), a value of
            0.95 will be used if :py:attr:`UQ` = False and values of (0.92, 0.98) will be used if
            :py:attr:`UQ` = True. Defaults to None.
        power_bin_mad_thresh (tuple | float, optional): The filter threshold for each power bin
            used to identify abnormal operation, expressed as the number of median absolute
            deviations from the median wind speed. This should be a tuple when :py:attr:`UQ`
            = True (values are Monte-Carlo sampled within the specified range) or a single value
            when :py:attr:`UQ` = False. If undefined (None), a value of 7.0 will be used if
            :py:attr:`UQ` = False and values of (4.0, 13.0) will be used if :py:attr:`UQ` = True.
            Defaults to None.
        use_power_coeff (bool, optional): If True, power performance as a function of wind vane
            angle will be quantified by normalizing power by the cube of the wind speed,
            approximating the power coefficient. If False, only power will be used. Defaults to False.
    """

    plant: PlantData = field(converter=deepcopy, validator=attrs.validators.instance_of(PlantData))
    turbine_ids: list[str] = field(default=None)
    UQ: bool = field(default=True, converter=bool)
    num_sim: int = field(default=100, converter=int)
    ws_bins: list[float] = field(
        default=[5.0, 6.0, 7.0, 8.0],
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(list),
            member_validator=attrs.validators.instance_of((int, float)),
        ),
    )
    ws_bin_width: float = field(default=1.0, converter=float)
    vane_bin_width: float = field(default=1.0, converter=float)
    min_vane_bin_count: int = field(default=100, validator=attrs.validators.instance_of(int))
    max_abs_vane_angle: float = field(default=25.0, converter=float)
    pitch_thresh: float = field(default=0.5, converter=float)
    num_power_bins: int = field(default=25, validator=attrs.validators.instance_of(int))
    min_power_filter: float = field(
        default=0.01, converter=float, validator=validate_half_closed_0_1_right
    )
    max_power_filter: float | tuple[float, float] = field(
        default=(0.92, 0.98), validator=(validate_UQ_input, validate_half_closed_0_1_right)
    )
    power_bin_mad_thresh: float | tuple[float, float] = field(
        default=(4.0, 10.0), validator=validate_UQ_input
    )
    use_power_coeff: bool = field(default=False, validator=attrs.validators.instance_of(bool))

    # Internally created attributes need to be given a type before usage
    inputs: pd.DataFrame = field(init=False)
    power_values_vane_ws: NDArrayFloat = field(init=False)
    yaw_misalignment_ws: NDArrayFloat = field(init=False)
    mean_vane_angle_ws: NDArrayFloat = field(init=False)
    yaw_misalignment: NDArrayFloat = field(init=False)
    mean_vane_angle: NDArrayFloat = field(init=False)
    yaw_misalignment_avg: float = field(init=False)
    yaw_misalignment_std: float = field(init=False)
    yaw_misalignment_95ci: NDArrayFloat = field(init=False)
    yaw_misalignment_avg_ws: NDArrayFloat = field(init=False)
    yaw_misalignment_std_ws: NDArrayFloat = field(init=False)
    yaw_misalignment_95ci_ws: NDArrayFloat = field(init=False)
    _run: pd.DataFrame = field(init=False)
    _vane_bins: list[float] = field(init=False)
    _df_turb: pd.DataFrame = field(init=False)
    _df_turb_ws: pd.DataFrame = field(init=False)
    _curve_fit_params_ws: NDArrayFloat = field(init=False)
    run_parameters: list[str] = field(
        init=False,
        default=[
            "num_sim",
            "ws_bins",
            "ws_bin_width",
            "vane_bin_width",
            "min_vane_bin_count",
            "max_abs_vane_angle",
            "pitch_thresh",
            "num_power_bins",
            "min_power_filter",
            "max_power_filter",
            "power_bin_mad_thresh",
            "use_power_coeff",
        ],
    )

    @logged_method_call
    def __attrs_post_init__(self):
        """
        Initialize logging and post-initialization setup steps.
        """
        if {"StaticYawMisalignment", "all"}.intersection(self.plant.analysis_type) == set():
            self.plant.analysis_type.append("StaticYawMisalignment")

        # Ensure the data are up to spec before continuing with initialization
        self.plant.validate()

        logger.info("Initializing StaticYawMisalignment analysis object")

        # Check that selected UQ is allowed and reset num_sim if no UQ
        if self.UQ:
            logger.info("Note: uncertainty quantification will be performed in the calculation")
        else:
            logger.info("Note: uncertainty quantification will NOT be performed in the calculation")

        if self.turbine_ids is None:
            self.turbine_ids = list(self.plant.turbine_ids)

    @logged_method_call
    def run(
        self,
        num_sim: int | None = None,
        ws_bins: list[float] | None = None,
        ws_bin_width: float | None = None,
        vane_bin_width: float | None = None,
        min_vane_bin_count: int | None = None,
        max_abs_vane_angle: float | None = None,
        pitch_thresh: float | None = None,
        num_power_bins: int | None = None,
        min_power_filter: float | None = None,
        max_power_filter: float | None = None,
        power_bin_mad_thresh: float | None = None,
        use_power_coeff: bool | None = None,
    ):
        """
        Estimates static yaw misalignment for each wind speed bin for each specified wind turbine.
        After performing power curve filtering to remove timestamps when pitch angle is above a
        threshold or the turbine is operating abnormally, best-fit cosine curves are found for
        binned power performance vs. wind vane angle for each wind speed bin and turbine. The
        difference between the wind vane angle where power is maximized, based on the best-fit
        cosine curve, and the mean wind vane angle is treated as the static yaw misalignment. If UQ
        is True, Monte Carlo simulations will be performed to produce a distribution of yaw
        misalignment values from which 95% confidence intervals can be determined.

        Args:
            num_sim (int, optional): Number of Monte Carlo iterations to perform. Only used if
                :py:attr:`UQ` = True. Defaults to 100.
            ws_bins (float, optional): Wind speed bin centers for which yaw misalignment detection
                will be performed (m/s). Defaults to [5.0, 6.0, 7.0, 8.0].
            ws_bin_width (float, optional): Wind speed bin size to use when detecting yaw
                misalignment for individual wind seed bins (m/s). Defaults to 1 m/s.
            vane_bin_width (float, optional): Wind vane bin size to use when detecting yaw
                misalignment (degrees). Defaults to 1 degree.
            min_vane_bin_count (int, optional): Minimum number of data points needed in a wind vane
                bin for it to be included when detecting yaw misalignment. Defaults to 100.
            max_abs_vane_angle (float, optional): Maximum absolute wind vane angle considered when
                detecting yaw misalignment. Defaults to 25 degrees.
            pitch_thresh (float, optional): Maximum blade pitch angle considered when detecting yaw
                misalignment. Defaults to 0.5 degrees.
            num_power_bins (int, optional): Number of power bins to use for power curve bin
                filtering to remove outlier data points. Defaults to 25.
            min_power_filter (float, optional): Minimum power threshold, defined as a fraction
                of rated power, to which the power curve bin filter should be applied. Defaults to
                0.01.
            max_power_filter (tuple | float, optional): Maximum power threshold, defined as a fraction
                of rated power, to which the power curve bin filter should be applied. This should be
                a tuple when :py:attr:`UQ` = True (values are Monte-Carlo sampled within the specified
                range) or a single value when :py:attr:`UQ` = False. If undefined (None), a value of
                0.95 will be used if :py:attr:`UQ` = False and values of (0.92, 0.98) will be used if
                :py:attr:`UQ` = True. Defaults to None.
            power_bin_mad_thresh (tuple | float, optional): The filter threshold for each power bin
                used to identify abnormal operation, expressed as the number of median absolute
                deviations from the median wind speed. This should be a tuple when :py:attr:`UQ`
                = True (values are Monte-Carlo sampled within the specified range) or a single value
                when :py:attr:`UQ` = False. If undefined (None), a value of 7.0 will be used if
                :py:attr:`UQ` = False and values of (4.0, 13.0) will be used if :py:attr:`UQ` = True.
                Defaults to None.
            use_power_coeff (bool, optional): If True, power performance as a function of wind vane
                angle will be quantified by normalizing power by the cube of the wind speed,
                approximating the power coefficient. If False, only power will be used. Defaults to False.
        """
        initial_parameters = {}
        if num_sim is not None:
            initial_parameters["num_sim"] = self.num_sim
            self.num_sim = num_sim
        if ws_bins is not None:
            initial_parameters["ws_bins"] = self.ws_bins
            self.ws_bins = ws_bins
        if ws_bin_width is not None:
            initial_parameters["ws_bin_width"] = self.ws_bin_width
            self.ws_bin_width = ws_bin_width
        if vane_bin_width is not None:
            initial_parameters["vane_bin_width"] = self.vane_bin_width
            self.vane_bin_width = vane_bin_width
        if min_vane_bin_count is not None:
            initial_parameters["min_vane_bin_count"] = self.min_vane_bin_count
            self.min_vane_bin_count = min_vane_bin_count
        if max_abs_vane_angle is not None:
            initial_parameters["max_abs_vane_angle"] = self.max_abs_vane_angle
            self.max_abs_vane_angle = max_abs_vane_angle
        if pitch_thresh is not None:
            initial_parameters["pitch_thresh"] = self.pitch_thresh
            self.pitch_thresh = pitch_thresh
        if num_power_bins is not None:
            initial_parameters["num_power_bins"] = self.num_power_bins
            self.num_power_bins = num_power_bins
        if min_power_filter is not None:
            initial_parameters["min_power_filter"] = self.min_power_filter
            self.min_power_filter = min_power_filter
        if use_power_coeff is not None:
            initial_parameters["use_power_coeff"] = self.use_power_coeff
            self.use_power_coeff = use_power_coeff
        if max_power_filter is not None:
            initial_parameters["max_power_filter"] = self.max_power_filter
            self.max_power_filter = max_power_filter
        if power_bin_mad_thresh is not None:
            initial_parameters["power_bin_mad_thresh"] = self.power_bin_mad_thresh
            self.power_bin_mad_thresh = power_bin_mad_thresh
        # determine wind vane angle bins
        max_abs_vane_angle_trunc = self.vane_bin_width * np.floor(
            self.max_abs_vane_angle / self.vane_bin_width
        )
        self._vane_bins = np.arange(
            -1 * max_abs_vane_angle_trunc, max_abs_vane_angle_trunc, self.vane_bin_width
        ).tolist()

        # Set up Monte Carlo simulation inputs if UQ = True or single simulation inputs if UQ = False.
        self._setup_monte_carlo_inputs()

        for n in tqdm(range(self.num_sim)):
            self._run = self.inputs.loc[n].copy()

            # Estimate static yaw misalginment for each turbine
            for i, t in enumerate(self.turbine_ids):
                # Get turbine-sepcific scada dataframe
                self._df_turb = self.plant.scada.loc[
                    (slice(None), t),
                    ["WMET_HorWdSpd", "WTUR_W", "WMET_HorWdDirRel", "WROT_BlPthAngVal"],
                ]

                # remove power curve outliers
                self._remove_power_curve_outliers(t)

                # Estimate static yaw misalginment for each wind speed bin
                for k, ws in enumerate(self.ws_bins):
                    self._df_turb_ws = self._df_turb.loc[
                        (self._df_turb["WMET_HorWdSpd"] >= (ws - self.ws_bin_width / 2))
                        & (self._df_turb["WMET_HorWdSpd"] < (ws + self.ws_bin_width / 2))
                    ].copy()

                    # Randomly resample 10-minute periods for bootstrapping
                    if self.UQ:
                        self._df_turb_ws = self._df_turb_ws.sample(frac=1.0, replace=True)

                    (
                        yaw_misalignment,
                        mean_vane_angle,
                        curve_fit_params,
                        power_values_vane,
                    ) = self._estimate_static_yaw_misalignment()

                    if self.UQ:
                        self.yaw_misalignment_ws[n, i, k] = yaw_misalignment
                        self.mean_vane_angle_ws[n, i, k] = mean_vane_angle
                        self.power_values_vane_ws[n, i, k, :] = power_values_vane
                        self._curve_fit_params_ws[n, i, k, :] = curve_fit_params
                    else:
                        self.yaw_misalignment_ws[i, k] = yaw_misalignment
                        self.mean_vane_angle_ws[i, k] = mean_vane_angle
                        self.power_values_vane_ws[i, k, :] = power_values_vane
                        self._curve_fit_params_ws[i, k, :] = curve_fit_params

                if self.UQ:
                    self.yaw_misalignment[n, i] = np.mean(self.yaw_misalignment_ws[n, i, :])
                    self.mean_vane_angle[n, i] = np.mean(self.mean_vane_angle_ws[n, i, :])
                else:
                    self.yaw_misalignment[i] = np.mean(self.yaw_misalignment_ws[i, :])
                    self.mean_vane_angle[i] = np.mean(self.mean_vane_angle_ws[i, :])

        # Compute mean, std. dev., and 95% confidence intervals of yaw misalginments
        if self.UQ:
            self.yaw_misalignment_avg = np.mean(self.yaw_misalignment, 0)
            self.yaw_misalignment_std = np.std(self.yaw_misalignment, 0)
            self.yaw_misalignment_95ci = np.percentile(
                self.yaw_misalignment, [2.5, 97.5], 0
            ).transpose()

            self.yaw_misalignment_avg_ws = np.mean(self.yaw_misalignment_ws, 0)
            self.yaw_misalignment_std_ws = np.std(self.yaw_misalignment_ws, 0)
            self.yaw_misalignment_95ci_ws = np.percentile(
                self.yaw_misalignment_ws, [2.5, 97.5], 0
            ).transpose((1, 2, 0))

        self.set_values(initial_parameters)

    @logged_method_call
    def _setup_monte_carlo_inputs(self):
        """
        Create and populate the data frame defining the Monte Carlo simulation parameters. This
        data frame is stored as self.inputs. Variables used to save intermediate variables and
        final results are also initiated.
        """

        if self.UQ:
            inputs = {
                "power_bin_mad_thresh": np.random.randint(
                    self.power_bin_mad_thresh[0], self.power_bin_mad_thresh[1] + 1, self.num_sim
                ),
                "max_power_filter": np.random.randint(
                    self.max_power_filter[0] * 100,
                    self.max_power_filter[1] * 100 + 1,
                    self.num_sim,
                )
                / 100.0,
            }
            self.inputs = pd.DataFrame(inputs)

            # For saving power or power coefficient as a function of wind vane for each wind speed bin
            self.power_values_vane_ws = np.empty(
                [self.num_sim, len(self.turbine_ids), len(self.ws_bins), len(self._vane_bins)]
            )

            # For saving cosine curve fit parameters, yaw misalignment, and mean wind vane angle for each wind speed bin
            self._curve_fit_params_ws = np.empty(
                [self.num_sim, len(self.turbine_ids), len(self.ws_bins), 3]
            )
            self.yaw_misalignment_ws = np.empty(
                [self.num_sim, len(self.turbine_ids), len(self.ws_bins)]
            )
            self.mean_vane_angle_ws = np.empty(
                [self.num_sim, len(self.turbine_ids), len(self.ws_bins)]
            )

            # For saving yaw misalignment and mean wind vane angle averaged over all wind speed bins
            self.yaw_misalignment = np.empty([self.num_sim, len(self.turbine_ids)])
            self.mean_vane_angle = np.empty([self.num_sim, len(self.turbine_ids)])

            self.yaw_misalignment_avg = np.empty([len(self.turbine_ids)])
            self.yaw_misalignment_std = np.empty([len(self.turbine_ids)])
            self.yaw_misalignment_95ci = np.empty([len(self.turbine_ids), 2])
            self.yaw_misalignment_avg_ws = np.empty([len(self.turbine_ids), len(self.ws_bins)])
            self.yaw_misalignment_std_ws = np.empty([len(self.turbine_ids), len(self.ws_bins)])
            self.yaw_misalignment_95ci_ws = np.empty([len(self.turbine_ids), len(self.ws_bins), 2])

        elif not self.UQ:
            inputs = {
                "power_bin_mad_thresh": [self.power_bin_mad_thresh],
                "max_power_filter": [self.max_power_filter],
            }
            self.inputs = pd.DataFrame(inputs)

            # For saving power or power coefficient as a function of wind vane for each wind speed bin
            self.power_values_vane_ws = np.empty(
                [len(self.turbine_ids), len(self.ws_bins), len(self._vane_bins)]
            )

            # For saving cosine curve fit parameters, yaw misalignment, and mean wind vane angle for each wind speed bin
            self._curve_fit_params_ws = np.empty([len(self.turbine_ids), len(self.ws_bins), 3])
            self.yaw_misalignment_ws = np.empty([len(self.turbine_ids), len(self.ws_bins)])
            self.mean_vane_angle_ws = np.empty([len(self.turbine_ids), len(self.ws_bins)])

            # For saving yaw misalignment and mean wind vane angle averaged over all wind speed bins
            self.yaw_misalignment = np.empty([len(self.turbine_ids)])
            self.mean_vane_angle = np.empty([len(self.turbine_ids)])

            self.num_sim = 1

    @logged_method_call
    def _remove_power_curve_outliers(self, turbine_id):
        """
        Removes power curve outliers for a specific turbine by removing timestamps where the pitch
        angle is above a threshold and timestamps where the wind speed is more than a specific
        threshold from the median wind speed in each power bin. The filtered turbine data frame is
        meant to include timestamps when the turbine is operating normally in below-rated
        conditions.

        Args:
            turbine_id (str): The name of the turbine for which power curve outlier removal will be performed.
        """

        # Limit to pitch angles below the specified threshold
        self._df_turb = self._df_turb.loc[self._df_turb["WROT_BlPthAngVal"] <= self.pitch_thresh]

        # Apply bin-based filter to flag samples for which wind speed is greater than a threshold from the median
        # wind speed in each power bin
        turb_capac = self.plant.asset.loc[turbine_id, "rated_power"]
        bin_width_frac = (self._run.max_power_filter - self.min_power_filter) / self.num_power_bins
        flag_bin = filters.bin_filter(
            bin_col=self._df_turb["WTUR_W"],
            value_col=self._df_turb["WMET_HorWdSpd"],
            bin_width=bin_width_frac * turb_capac,
            threshold=self._run.power_bin_mad_thresh,
            center_type="median",
            bin_min=self.min_power_filter * turb_capac,
            bin_max=self._run.max_power_filter * turb_capac,
            threshold_type="mad",
            direction="all",
        )

        self._df_turb = self._df_turb.loc[~flag_bin]

    @logged_method_call
    def _estimate_static_yaw_misalignment(self):
        """
        Estimates static yaw misalignment for a single turbine and wind speed bin by fitting a
        cosine curve to the binned power performance vs. wind vane angle. Yaw misalignment is
        estimated as the difference between the wind vane angle where power is maximized based on
        the best-fit cosine curve and the mean wind vane angle.

        Returns:
            tuple[float, float, np.ndarray, np.ndarray]: The estimated static yaw misaligment, the
                mean wind vane angle, and arrays containing the best-fit cosine curve parameters
                (magnitude, offset (degrees), and cosine exponent) and power performance values
                binned by wind vane angle.
        """

        self._df_turb_ws["vane_bin"] = self.vane_bin_width * np.round(
            self._df_turb_ws["WMET_HorWdDirRel"].values / self.vane_bin_width
        )

        # Normalize by wind speed cubed if using power coefficient to determine power performance
        if self.use_power_coeff:
            self._df_turb_ws["pow_ref"] = self._df_turb_ws["WMET_HorWdSpd"].values ** 3
        else:
            self._df_turb_ws["pow_ref"] = 1.0

        self._df_turb_ws["pow_ratio"] = (
            self._df_turb_ws["WTUR_W"].values / self._df_turb_ws["pow_ref"].values
        )

        mean_vane_angle = self._df_turb_ws["WMET_HorWdDirRel"].values.mean()

        # Bin power performance by wind vane
        df_bin = self._df_turb_ws.groupby("vane_bin").mean()
        df_bin_count = self._df_turb_ws.groupby("vane_bin").count()

        # Remove bins with too few samples or vane angles that are too large
        df_bin = df_bin.loc[
            (df_bin_count["WTUR_W"] > self.min_vane_bin_count)
            & (np.abs(df_bin.index) <= self.max_abs_vane_angle)
        ]

        # Find best fit cosine curve parameters
        curve_fit_params, _ = curve_fit(
            cos_curve, df_bin.index, df_bin["pow_ratio"], [df_bin["pow_ratio"].max(), 0.0, 2.0]
        )

        # yaw_misalignment, mean_vane_angle, curve_fit_params, power_values_vane

        return (
            curve_fit_params[1] - mean_vane_angle,
            mean_vane_angle,
            curve_fit_params,
            df_bin["pow_ratio"].reindex(self._vane_bins).values,
        )

    def plot_yaw_misalignment_by_turbine(
        self,
        turbine_ids: list[str] = None,
        xlim: tuple[float, float] = (None, None),
        ylim: tuple[float, float] = (None, None),
        return_fig: bool = False,
        figure_kwargs: dict = None,
        plot_kwargs_curve: dict = {},
        plot_kwargs_line: dict = {},
        plot_kwargs_fill: dict = {},
        legend_kwargs: dict = {},
    ):
        """Plots power performance vs. wind vane angle along with the best-fit cosine curve for
        each wind speed bin for each turbine specified. The mean wind vane angle and the wind vane
        angle where power performance is maximized are shown for each wind speed bin. Additionally,
        the yaw misalignments for each wind speed bin as well as the mean yaw misalignment
        avergaged over all wind speed bins are listed. If UQ is True, 95% confidence intervals will
        be plotted for the binned power performance values and listed for the yaw misalignment
        estiamtes.

        Args:
            turbine_ids (list[str], optional): Name of turbines for which yaw misalignment data are
                plotted. If None, plots for all turbines for which yaw misalginment detection was
                performed will be generated. Defaults to None.
            xlim (:obj:`tuple[float, float]`, optional): A tuple of floats representing the x-axis
                wind vane angle plotting display limits (degrees). Defaults to (None, None).
            ylim (:obj:`tuple[float, float]`, optional): A tuple of the y-axis plotting display limits
                for the power performance vs. wind vane plots. Defaults to (None, None).
            return_fig (:obj:`bool`, optional): Flag to return the figure and axes objects. Defaults to False.
            figure_kwargs (:obj:`dict`, optional): Additional figure instantiation keyword arguments
                that are passed to ``plt.figure()``. Defaults to None.
            plot_kwargs_curve (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                ``ax.plot()`` for plotting lines for the power performance vs. wind vane plots. Defaults to {}.
            plot_kwargs_line (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                ``ax.plot()`` for plotting vertical lines indicating mean vane angle and vane angle where
                power is maximized. Defaults to {}.
            plot_kwargs_fill (:obj:`dict`, optional): If :py:attr:`UQ` is True, additional plotting keyword arguments
                that are passed to ``ax.fill_between()`` for plotting shading regions for 95% confidence
                intervals for power performance vs. wind vane. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                ``ax.legend()`` for the power performance vs. wind vane plots. Defaults to {}.
        Returns:
            None | dict of tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]:
                If :py:attr:`return_fig` is True, then a dictionary containing the figure and axes object(s)
                corresponding to the yaw misalignment plots for each turbine are returned for further
                tinkering/saving. The turbine names in the `turbine_ids` aregument are the dicitonary
                keys.
        """

        if self.use_power_coeff:
            power_performance_label = "Normalized Cp (-)"
        else:
            power_performance_label = "Normalized Power (-)"

        if turbine_ids is None:
            turbine_ids = self.turbine_ids
        else:
            if set(turbine_ids).difference(self.turbine_ids):
                raise ValueError(
                    "All turbine names in the argument `turbine_ids` must be present in the list of"
                    "turbines for which yaw misalginment detection was performed."
                )

        if return_fig:
            axes_dict = {}

        for turbine_id in turbine_ids:
            i = self.turbine_ids.index(turbine_id)

            if self.UQ:
                power_values_vane_ws = self.power_values_vane_ws[:, i, :, :]
                curve_fit_params_ws = self._curve_fit_params_ws[:, i, :, :]
                mean_vane_angle_ws = np.mean(self.mean_vane_angle_ws[:, i, :], 0)
                yaw_misalignment_ws = self.yaw_misalignment_ws[:, i, :]
            else:
                power_values_vane_ws = self.power_values_vane_ws[i, :, :]
                curve_fit_params_ws = self._curve_fit_params_ws[i, :, :]
                mean_vane_angle_ws = self.mean_vane_angle_ws[i, :]
                yaw_misalignment_ws = self.yaw_misalignment_ws[i, :]

            return_vals = plot.plot_yaw_misalignment(
                self.ws_bins,
                self._vane_bins,
                power_values_vane_ws,
                curve_fit_params_ws,
                mean_vane_angle_ws,
                yaw_misalignment_ws,
                turbine_id,
                power_performance_label,
                xlim,
                ylim,
                return_fig,
                figure_kwargs,
                plot_kwargs_curve,
                plot_kwargs_line,
                plot_kwargs_fill,
                legend_kwargs,
            )

            if return_fig:
                axes_dict[turbine_id] = return_vals

        if return_fig:
            return axes_dict


__defaults_UQ = StaticYawMisalignment.__attrs_attrs__.UQ.default
__defaults_turbine_ids = StaticYawMisalignment.__attrs_attrs__.turbine_ids.default
__defaults_num_sim = StaticYawMisalignment.__attrs_attrs__.num_sim.default
__defaults_ws_bins = StaticYawMisalignment.__attrs_attrs__.ws_bins.default
__defaults_ws_bin_width = StaticYawMisalignment.__attrs_attrs__.ws_bin_width.default
__defaults_vane_bin_width = StaticYawMisalignment.__attrs_attrs__.vane_bin_width.default
__defaults_min_vane_bin_count = StaticYawMisalignment.__attrs_attrs__.min_vane_bin_count.default
__defaults_max_abs_vane_angle = StaticYawMisalignment.__attrs_attrs__.max_abs_vane_angle.default
__defaults_pitch_thresh = StaticYawMisalignment.__attrs_attrs__.pitch_thresh.default
__defaults_num_power_bins = StaticYawMisalignment.__attrs_attrs__.num_power_bins.default
__defaults_min_power_filter = StaticYawMisalignment.__attrs_attrs__.min_power_filter.default
__defaults_max_power_filter = StaticYawMisalignment.__attrs_attrs__.max_power_filter.default
__defaults_power_bin_mad_thresh = StaticYawMisalignment.__attrs_attrs__.power_bin_mad_thresh.default
__defaults_use_power_coeff = StaticYawMisalignment.__attrs_attrs__.use_power_coeff.default


def create_StaticYawMisalignment(
    project: PlantData,
    turbine_ids: list[str] = __defaults_turbine_ids,
    UQ: bool = __defaults_UQ,
    num_sim: int = __defaults_num_sim,
    ws_bins: list[float] = __defaults_ws_bins,
    ws_bin_width: float = __defaults_ws_bin_width,
    vane_bin_width: float = __defaults_vane_bin_width,
    min_vane_bin_count: int = __defaults_min_vane_bin_count,
    max_abs_vane_angle: float = __defaults_max_abs_vane_angle,
    pitch_thresh: float = __defaults_pitch_thresh,
    num_power_bins: int = __defaults_num_power_bins,
    min_power_filter: float = __defaults_min_power_filter,
    max_power_filter: float | tuple[float, float] = __defaults_max_power_filter,
    power_bin_mad_thresh: float | tuple[float, float] = __defaults_power_bin_mad_thresh,
    use_power_coeff: bool = __defaults_use_power_coeff,
) -> StaticYawMisalignment:
    return StaticYawMisalignment(
        project=project,
        turbine_ids=turbine_ids,
        UQ=UQ,
        num_sim=num_sim,
        ws_bins=ws_bins,
        ws_bin_width=ws_bin_width,
        vane_bin_width=vane_bin_width,
        min_vane_bin_count=min_vane_bin_count,
        max_abs_vane_angle=max_abs_vane_angle,
        pitch_thresh=pitch_thresh,
        num_power_bins=num_power_bins,
        min_power_filter=min_power_filter,
        max_power_filter=max_power_filter,
        power_bin_mad_thresh=power_bin_mad_thresh,
        use_power_coeff=use_power_coeff,
    )


create_StaticYawMisalignment.__doc__ = StaticYawMisalignment.__doc__
