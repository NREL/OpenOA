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
# 1. Y. Bao, Q. Yang, L. Fu, Q. Chen, C. Cheng, and Y. Sun, "Identification of Yaw Error Inherent
#    Misalignment for Wind Turbine Based on SCADA Data: A Data Mining Approach," Proc. 12th Asian
#    Control Conference (ASCC), Kitakyushu, Japan, 1095-1100, 2019.
# 2. J. Xue and L. Wang, “Online data-driven approach of yaw error estimation and correction of
#    horizontal axis wind turbine,” IET J. Eng., 2019(18), 4937–4940, 2019.
# 3. D. Astolfi, F. Castellani, and L. Terzi, “An Operation Data-Based Method for the Diagnosis of
#    Zero-Point Shift of Wind Turbines Yaw Angle,” J. Solar Energy Engineering, 142, 024501, 2020.
# 4. B. Jing, Z. Qian, Y. Pei, L. Zhang, and T. Yang, "Improving wind turbine efficiency through
#    detection and calibration of yaw misalignment," Renewable Energy, 160, 1217-1227, 2020.
# 5. L. Gao and J. Hong, “Data-driven yaw misalignment correction for utility-scale wind turbines,”
#    J. Renewable Sustainable Energy, 13, 063302, 2021.

from __future__ import annotations

import random

import attrs
import numpy as np
import pandas as pd
import numpy.typing as npt
from tqdm import tqdm
from attrs import field, define
from scipy.optimize import curve_fit

from openoa.plant import PlantData
from openoa.utils import plot, filters
from openoa.schema import FromDictMixin
from openoa.logging import logging, logged_method_call


logger = logging.getLogger(__name__)
NDArrayFloat = npt.NDArray[np.float64]
plot.set_styling()


def cos_curve(x, A, Offset, cos_exp):
    # Cosine exponent curve as a function of yaw misalignment for curve fitting
    return A * np.cos((np.pi / 180) * (x - Offset)) ** cos_exp


@define(auto_attribs=True)
class StaticYawMisalignment(FromDictMixin):
    """
    A method for estimating static yaw misalignment for different wind speed bins for each specified wind turbine as well as the average static yaw msialignment over all wind speed bins using turbine-level SCADA data.

    The method is comprised of the following core steps, which are performed for each specified
    wind turbine. If :py:attr:`UQ` is selected, the following steps are performed multiple times
    using Monte Carlo simulation to produce a distribution of static yaw misalignment estimates
    from which 95% confidence intervals can be derived:
      1. Timestamps containing power curve outliers are removed. Specifically, pitch angles are
         limited to a specified threshold to remove timestamps when turbines are operating in or
         near above-rated conditions where yaw misalignment has little impact on power performance.
         Next to increase the likelihood that power performance deviations are caused by yaw
         misalignment, a power curve outlier detection filter is used to remove timestamps when the
         turbine is oeprating abnormally. If :py:attr:`UQ` is selected, power curve outlier
         detection parameters will be chosen randomly each Monte Carlo iteration.
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

    Args:
        plant (:obj:`PlantData`): A :py:attr:`openoa.plant.PlantData` object that has been validated
            with at least :py:attr:`openoa.plant.PlantData.analysis_type` = "StaticYawMisalignment".
        turbine_ids (:obj:`list`, optional): List of turbine IDs for which static yaw misalignment
            detection will be performed. If None, all turbines will be analyzed. Defaults to None.
        UQ (:obj:`bool`, optional): Dertermines whether to perform uncertainty quantification using
            Monte Carlo simulation (True) or provide a single yaw misalignment estimate (False).
            Defaults to True.
    """

    plant: PlantData = field(validator=attrs.validators.instance_of(PlantData))
    turbine_ids: list[str] = field(default=None)
    UQ: bool = field(default=True, converter=bool)

    # Internally created attributes need to be given a type before usage
    num_sim: int = field(init=False)
    ws_bins: list[float] = field(init=False)
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
    _ws_bin_width: float = field(init=False)
    _vane_bin_width: float = field(init=False)
    _min_vane_bin_count: int = field(init=False)
    _max_abs_vane_angle: float = field(init=False)
    _pitch_thresh: float = field(init=False)
    _max_power_filter: float | tuple[float, float] = field(init=False)
    _power_bin_mad_thresh: float | tuple[float, float] = field(init=False)
    _use_power_coeff: bool = field(init=False)
    _run: pd.DataFrame = field(init=False)
    _vane_bins: list[float] = field(init=False)
    _df_turb: pd.DataFrame = field(init=False)
    _df_turb_ws: pd.DataFrame = field(init=False)
    _curve_fit_params_ws: NDArrayFloat = field(init=False)

    @plant.validator
    def validate_plant_ready_for_anylsis(
        self, attribute: attrs.Attribute, value: PlantData
    ) -> None:
        """Validates that the value has been validated for a static yaw misalignment analysis."""
        if set(("StaticYawMisalignment", "all")).intersection(value.analysis_type) == set():
            raise TypeError(
                "The input to 'plant' must be validated for at least 'StaticYawMisalignment'"
            )

    @logged_method_call
    def __attrs_post_init__(self):
        """
        Initialize logging and post-initialization setup steps.
        """
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
        num_sim: int = 100,
        ws_bins: list[float] = [5.0, 6.0, 7.0, 8.0],
        ws_bin_width: float = 1.0,
        vane_bin_width: float = 1.0,
        min_vane_bin_count: int = 100,
        max_abs_vane_angle=25.0,
        pitch_thresh: float = 0.5,
        max_power_filter: float = None,
        power_bin_mad_thresh: float = None,
        use_power_coeff: bool = False,
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

        self.num_sim = num_sim
        self.ws_bins = ws_bins
        self._ws_bin_width = ws_bin_width
        self._vane_bin_width = vane_bin_width
        self._min_vane_bin_count = min_vane_bin_count
        self._max_abs_vane_angle = max_abs_vane_angle
        self._pitch_thresh = pitch_thresh
        self._use_power_coeff = use_power_coeff

        # Assign default parameter values depending on whether UQ is performed
        if max_power_filter is not None:
            self._max_power_filter = max_power_filter
        elif self.UQ:
            self._max_power_filter = (0.92, 0.98)
        else:
            self._max_power_filter = 0.95

        if power_bin_mad_thresh is not None:
            self._power_bin_mad_thresh = power_bin_mad_thresh
        elif self.UQ:
            self._power_bin_mad_thresh = (4.0, 10.0)
        else:
            self._power_bin_mad_thresh = 7.0

        # determine wind vane angle bins
        max_abs_vane_angle_trunc = self._vane_bin_width * np.floor(
            self._max_abs_vane_angle / self._vane_bin_width
        )
        self._vane_bins = np.arange(
            -1 * max_abs_vane_angle_trunc, max_abs_vane_angle_trunc, self._vane_bin_width
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
                        (self._df_turb["WMET_HorWdSpd"] >= (ws - self._ws_bin_width / 2))
                        & (self._df_turb["WMET_HorWdSpd"] < (ws + self._ws_bin_width / 2))
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
                    self._power_bin_mad_thresh[0], self._power_bin_mad_thresh[1] + 1, self.num_sim
                ),
                "max_power_filter": np.random.randint(
                    self._max_power_filter[0] * 100,
                    self._max_power_filter[1] * 100 + 1,
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
                "power_bin_mad_thresh": [self._power_bin_mad_thresh],
                "max_power_filter": [self._max_power_filter],
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
        self._df_turb = self._df_turb.loc[self._df_turb["WROT_BlPthAngVal"] <= self._pitch_thresh]

        # Apply bin-based filter to flag samples for which wind speed is greater than a threshold from the median
        # wind speed in each power bin
        turb_capac = self.plant.asset.loc[turbine_id, "rated_power"]
        bin_width_frac = 0.04 * (self._run.max_power_filter - 0.01)
        flag_bin = filters.bin_filter(
            bin_col=self._df_turb["WTUR_W"],
            value_col=self._df_turb["WMET_HorWdSpd"],
            bin_width=bin_width_frac * turb_capac,
            threshold=self._run.power_bin_mad_thresh,
            center_type="median",
            bin_min=0.01 * turb_capac,
            bin_max=self._run.max_power_filter * turb_capac,
            threshold_type="mad",
            direction="all",
        )

        self._df_turb = self._df_turb.loc[~flag_bin]

    def _estimate_static_yaw_misalignment(self):
        """
        Estimates static yaw misalignment for a single turbine and wind speed bin by fitting a
        cosine curve to the binned power performance vs. wind vane angle. Yaw misalignment is
        estimated as the difference between the wind vane angle where power is maximized based on
        the best-fit cosine curve and the mean wind vane angle.

        Returns:
            tuple[float, float, np.ndarray, np.ndarray]: The estimated static yaw misaligment, the
            mean wind vane angle, and arrays containing the best-fit cosine curve parameters
            (magnitude, offset (degrees), and cosine exponent) and power performance values binned
            by wind vane angle.
        """

        self._df_turb_ws["vane_bin"] = (
            self._vane_bin_width
            * (self._df_turb_ws["WMET_HorWdDirRel"] / self._vane_bin_width).round()
        )

        # Normalize by wind speed cubed if using power coefficient to determine power performance
        if self._use_power_coeff:
            self._df_turb_ws["pow_ref"] = self._df_turb_ws["WMET_HorWdSpd"] ** 3
        else:
            self._df_turb_ws["pow_ref"] = 1.0

        self._df_turb_ws["pow_ratio"] = self._df_turb_ws["WTUR_W"] / self._df_turb_ws["pow_ref"]

        mean_vane_angle = self._df_turb_ws["WMET_HorWdDirRel"].mean()

        # Bin power performance by wind vane
        df_bin = self._df_turb_ws.groupby("vane_bin").mean()
        df_bin_count = self._df_turb_ws.groupby("vane_bin").count()

        # Remove bins with too few samples or vane angles that are too large
        df_bin = df_bin.loc[
            (df_bin_count["WTUR_W"] > self._min_vane_bin_count)
            & (np.abs(df_bin.index) <= self._max_abs_vane_angle)
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
                that are passed to `plt.figure()`. Defaults to None.
            plot_kwargs_curve (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                `ax.plot()` for plotting lines for the power performance vs. wind vane plots. Defaults to {}.
            plot_kwargs_line (:obj:`dict`, optional): Additional plotting keyword arguments that are passed to
                `ax.plot()` for plotting vertical lines indicating mean vane angle and vane angle where
                power is maximized. Defaults to {}.
            plot_kwargs_fill (:obj:`dict`, optional): If `UQ` is True, additional plotting keyword arguments
                that are passed to `ax.fill_between()` for plotting shading regions for 95% confidence
                intervals for power performance vs. wind vane. Defaults to {}.
            legend_kwargs (:obj:`dict`, optional): Additional legend keyword arguments that are passed to
                `ax.legend()` for the power performance vs. wind vane plots. Defaults to {}.
        Returns:
            None | dict of tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]:
                If `return_fig` is True, then a dictionary containing the figure and axes object(s)
                corresponding to the yaw misalignment plots for each turbine are returned for further
                tinkering/saving. The turbine names in the `turbine_ids` aregument are the dicitonary
                keys.
        """

        if self._use_power_coeff:
            power_performance_label = "Normalized Cp (-)"
        else:
            power_performance_label = "Normalized Power (-)"

        if turbine_ids is None:
            turbine_ids = self.turbine_ids
        else:
            if (set(turbine_ids) - set(self.turbine_ids)) != set():
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
