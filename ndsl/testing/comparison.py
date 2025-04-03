from typing import Any, List, Optional, Union

import numpy as np
import numpy.typing as npt


def _fixed_width_float_16e(value: np.floating[Any]) -> str:
    """Account for extra '-' character"""
    if value > 0:
        return f" {value:.16e}"
    else:
        return f"{value:.16e}"


def _fixed_width_float_2e(value: np.floating[Any]) -> str:
    """Account for extra '-' character"""
    if value > 0:
        return f" {value:.2e}"
    else:
        return f"{value:.2e}"


class BaseMetric:
    def __init__(
        self,
        reference_values: np.ndarray,
        computed_values: np.ndarray,
    ):
        self.references = np.atleast_1d(reference_values)
        self.computed = np.atleast_1d(computed_values)
        self.check = False

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def report(self, file_path: Optional[str] = None) -> List[str]:
        ...

    def one_line_report(self) -> str:
        ...


class LegacyMetric(BaseMetric):
    """Legacy (AI2) metric used for original FV3 port.

    This metric attempts to smooth error comparison around 0.
    It further tries to deal with close-to-0 breakdown of absolute
    error by allowing `near_zero` threshold to be specified by hand.
    """

    def __init__(
        self,
        reference_values: np.ndarray,
        computed_values: np.ndarray,
        eps: float,
        ignore_near_zero_errors: Union[dict, bool],
        near_zero: float,
    ):
        super().__init__(reference_values, computed_values)
        self.eps = eps
        self._calculated_metric = np.empty_like(self.references)
        self.success = self._compute_errors(
            ignore_near_zero_errors,
            near_zero,
        )
        self.check = np.all(self.success)

    def _compute_errors(
        self,
        ignore_near_zero_errors,
        near_zero,
    ) -> npt.NDArray[np.bool_]:
        if self.references.dtype in (np.float64, np.int64, np.float32, np.int32):
            denom = self.references
            denom[self.references == 0] = self.computed[self.references == 0]
            self._calculated_metric = np.asarray(
                np.abs((self.computed - self.references) / denom)
            )
            self._calculated_metric[denom == 0] = 0.0
        elif self.references.dtype in (np.bool_, bool):
            self._calculated_metric = np.logical_xor(self.computed, self.references)
        else:
            raise TypeError(
                f"received data with unexpected dtype {self.references.dtype}"
            )
        success = np.logical_or(
            np.logical_and(np.isnan(self.computed), np.isnan(self.references)),
            self._calculated_metric < self.eps,
        )
        if isinstance(ignore_near_zero_errors, dict):
            if ignore_near_zero_errors.keys():
                near_zero = ignore_near_zero_errors["near_zero"]
                success = np.logical_or(
                    success,
                    np.logical_and(
                        np.abs(self.computed) < near_zero,
                        np.abs(self.references) < near_zero,
                    ),
                )
        elif ignore_near_zero_errors:
            success = np.logical_or(
                success,
                np.logical_and(
                    np.abs(self.computed) < near_zero,
                    np.abs(self.references) < near_zero,
                ),
            )
        return success

    def one_line_report(self) -> str:
        if self.check:
            return "✅ No numerical differences"
        else:
            return "❌ Numerical failures"

    def report(self, file_path: Optional[str] = None) -> List[str]:
        report = []
        report.append(self.one_line_report())
        if not self.check:
            found_indices = np.logical_not(self.success).nonzero()
            computed_failures = self.computed[found_indices]
            reference_failures = self.references[found_indices]

            # List all errors
            bad_indices_count = len(found_indices[0])
            # Determine worst result
            worst_metric_err = 0.0
            abs_errs = []
            details = [
                "All failures:",
                "Index  Computed  Reference  Absolute E  Metric E",
            ]
            for b in range(bad_indices_count):
                full_index = tuple([f[b] for f in found_indices])

                metric_err = self._calculated_metric[full_index]

                absolute_distance = abs(computed_failures[b] - reference_failures[b])
                abs_errs.append(absolute_distance)

                details.append(
                    f"{full_index}  {computed_failures[b]}  "
                    f"{reference_failures[b]}  {abs_errs[-1]:.3e}  {metric_err:.3e}"
                )

                if np.isnan(metric_err) or (abs(metric_err) > abs(worst_metric_err)):
                    worst_metric_err = metric_err
                    worst_full_idx = full_index
                    worst_abs_err = abs_errs[-1]
                    computed_worst = computed_failures[b]
                    reference_worst = reference_failures[b]
            # Try to quantify noisy errors
            unique_errors = len(np.unique(np.array(abs_errs)))
            # Summary and worst result
            fullcount = len(self.references.flatten())
            report.append(
                f"Failed count: {bad_indices_count}/{fullcount} "
                f"({round(100.0 * (bad_indices_count / fullcount), 2)}%),\n"
                f"Worst failed index {worst_full_idx}\n"
                f"    Computed:{computed_worst}\n"
                f"    Reference: {reference_worst}\n"
                f"    Absolute diff: {worst_abs_err:.3e}\n"
                f"    Metric diff: {worst_metric_err:.3e}\n"
                f"    Metric threshold: {self.eps}\n"
                f"  Noise quantification:\n"
                f"    Reference dtype: {type(reference_worst)}\n"
                f"    Unique errors: {unique_errors}/{bad_indices_count}"
            )
            report.extend(details)

        if file_path:
            with open(file_path, "w") as fd:
                fd.write("\n".join(report))

        return report

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        report = self.report()
        if len(report) > 30:
            report = report[:30]  # ~10 first errors
            report.append("...")
        return "\n".join(report)


class _Metric:
    def __init__(self, value):
        self._value: float = value
        self.is_default = True

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, _value: float):
        self._value = _value
        self.is_default = False


class MultiModalFloatMetric(BaseMetric):
    """Combination of absolute, relative & ULP comparison for floats

    This metric attempts to combine well known comparison on floats
    to leverage a robust 32/64 bit float comparison on large accumulating
    floating errors.

    ULP is used to clear noise (ULP<=1.0 passes)
    Absolute errors for large amplitude
    """

    _f32_absolute_eps = _Metric(1e-10)
    _f64_absolute_eps = _Metric(1e-13)
    relative_fraction = _Metric(0.000001)  # 0.0001%
    ulp_threshold = _Metric(1.0)

    def __init__(
        self,
        reference_values: np.ndarray,
        computed_values: np.ndarray,
        absolute_eps_override: float = -1,
        relative_fraction_override: float = -1,
        ulp_override: float = -1,
        sort_report: str = "ulp",
        **kwargs,
    ):
        super().__init__(reference_values, computed_values)
        self.absolute_distance = np.empty_like(self.references)
        self.absolute_distance_metric = np.empty_like(self.references, dtype=np.bool_)
        self.relative_distance = np.empty_like(self.references)
        self.relative_distance_metric = np.empty_like(self.references, dtype=np.bool_)
        self.ulp_distance = np.empty_like(self.references)
        self.ulp_distance_metric = np.empty_like(self.references, dtype=np.bool_)

        if self.references.dtype in [np.float32, np.int32]:
            self.absolute_eps = _Metric(self._f32_absolute_eps.value)
        else:
            self.absolute_eps = _Metric(self._f64_absolute_eps.value)

        # Assign overrides if needed
        if absolute_eps_override > self.absolute_eps.value:
            self.absolute_eps.value = absolute_eps_override
        if relative_fraction_override > self.relative_fraction.value:
            self.relative_fraction.value = relative_fraction_override
        if ulp_override > self.ulp_threshold.value:
            self.ulp_threshold.value = ulp_override

        self.success = self._compute_all_metrics()
        self.check = np.all(self.success)
        self.sort_report = sort_report

    def _compute_all_metrics(
        self,
    ) -> npt.NDArray[np.bool_]:
        if self.references.dtype in (np.float64, np.int64, np.float32, np.int32):
            max_values = np.maximum(
                np.absolute(self.computed), np.absolute(self.references)
            )
            # Absolute distance
            self.absolute_distance = np.absolute(self.computed - self.references)
            self.absolute_distance_metric = (
                self.absolute_distance < self.absolute_eps.value
            )
            # Relative distance (in pct)
            self.relative_distance = np.divide(self.absolute_distance, max_values)
            self.relative_distance_metric = (
                self.absolute_distance < self.relative_fraction.value * max_values
            )
            # ULP distance
            self.ulp_distance = np.divide(
                self.absolute_distance, np.spacing(max_values)
            )
            self.ulp_distance_metric = self.ulp_distance <= self.ulp_threshold.value

            # Combine all distances into success or failure
            # Success =
            # - no unexpected NANs (e.g. NaN in the ref MUST BE in computation) OR
            # - absolute distance pass OR
            # - relative distance pass OR
            # - ulp distance pass
            naninf_success = np.logical_and(
                np.isnan(self.computed), np.isnan(self.references)
            )
            metric_success = np.logical_or(
                self.relative_distance_metric, self.absolute_distance_metric
            )
            metric_success = np.logical_or(metric_success, self.ulp_distance_metric)
            success = np.logical_or(naninf_success, metric_success)
            return success
        elif self.references.dtype in (np.bool_, bool):
            success = np.logical_xor(self.computed, self.references)
            return success
        else:
            raise TypeError(
                f"received data with unexpected dtype {self.references.dtype}"
            )

    def _has_override(self) -> bool:
        return (
            not self.relative_fraction.is_default
            or not self.absolute_eps.is_default
            or not self.ulp_threshold.is_default
        )

    def one_line_report(self) -> str:
        metric_thresholds = f"{'🔶 ' if not self.absolute_eps.is_default else ''}Absolute E(<{self.absolute_eps.value:.2e})  "
        metric_thresholds += f"{'🔶 ' if not self.relative_fraction.is_default else ''}Relative E(<{self.relative_fraction.value * 100:.2e}%)   "
        metric_thresholds += f"{'🔶 ' if not self.ulp_threshold.is_default else ''}ULP E(<{self.ulp_threshold.value})"
        if self.check and self._has_override():
            return f"🔶 No numerical differences with threshold override - metric: {metric_thresholds}"
        elif self.check:
            return f"✅ No numerical differences - metric: {metric_thresholds}"
        else:
            failed_indices = len(np.logical_not(self.success).nonzero()[0])
            all_indices = len(self.references.flatten())
            return f"❌ Numerical failures: {failed_indices}/{all_indices} failed - metric: {metric_thresholds}"

    def report(self, file_path: Optional[str] = None) -> List[str]:
        report = []
        report.append(self.one_line_report())
        failed_indices = np.logical_not(self.success).nonzero()
        # List all errors to terminal and file
        bad_indices_count = len(failed_indices[0])
        full_count = len(self.references.flatten())
        failures_pct = round(100.0 * (bad_indices_count / full_count), 2)
        report = [
            f"All failures ({bad_indices_count}/{full_count}) ({failures_pct}%),\n",
            f"Index   Computed   Reference   "
            f"{'🔶 ' if not self.absolute_eps.is_default else ''}Absolute E(<{self.absolute_eps.value:.2e})  "
            f"{'🔶 ' if not self.relative_fraction.is_default else ''}Relative E(<{self.relative_fraction.value * 100:.2e}%)   "
            f"{'🔶 ' if not self.ulp_threshold.is_default else ''}ULP E(<{self.ulp_threshold.value})",
        ]
        # Summary and worst result
        if self.sort_report == "ulp":
            indices_flatten = np.argsort(self.ulp_distance.flatten())
        elif self.sort_report == "absolute":
            indices_flatten = np.argsort(self.absolute_distance.flatten())
        elif self.sort_report == "relative":
            indices_flatten = np.argsort(self.relative_distance.flatten())
        elif self.sort_report == "index":
            indices_flatten = list(range(self.ulp_distance.size - 1, -1, -1))
        else:
            RuntimeError(
                f"[Translate test] Unknown {self.sort_report} report sorting option."
            )
        for iFlat in indices_flatten[::-1]:
            fi = np.unravel_index(iFlat, shape=self.ulp_distance.shape)
            if np.isnan(self.computed[fi]) and np.isnan(self.references[fi]):
                continue
            ulp_dist = (
                self.ulp_distance[fi]
                if np.isnan(self.ulp_distance[fi])
                else int(self.ulp_distance[fi])
            )
            index_as_string = "("
            for i in fi:
                index_as_string += f"{i:02},"
            index_as_string = index_as_string[:-1]
            index_as_string += ")"
            report.append(
                f"{index_as_string}  "
                f"{_fixed_width_float_16e(self.computed[fi])}  "
                f"{_fixed_width_float_16e(self.references[fi])}  "
                f"{_fixed_width_float_2e(self.absolute_distance[fi])} {'✅' if self.absolute_distance_metric[fi] else '❌'}  "
                f"{_fixed_width_float_2e(self.relative_distance[fi] * 100)} {'✅' if self.relative_distance_metric[fi] else '❌'}  "
                f"{ulp_dist:02} {'✅' if self.ulp_distance_metric[fi] else '❌'}  "
            )

        if file_path:
            with open(file_path, "w") as fd:
                fd.write("\n".join(report))

        return report

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        report = self.report()
        if len(report) > 12:
            report = report[:12]  # ~10 first errors
            report.append("...")
        return "\n".join(report)
