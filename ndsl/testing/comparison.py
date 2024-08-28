from typing import Union

import numpy as np
import numpy.typing as npt


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
        self.success = self._compute_errors(
            ignore_near_zero_errors,
            near_zero,
        )
        self.check = np.all(self.success)
        self._calculated_metric = np.empty_like(self.references)

    def _compute_errors(
        self,
        ignore_near_zero_errors,
        near_zero,
    ) -> npt.NDArray[np.bool_]:
        if self.references.dtype in (np.float64, np.int64, np.float32, np.int32):
            denom = np.abs(self.references) + np.abs(self.computed)
            self._calculated_metric = np.asarray(
                2.0 * np.abs(self.computed - self.references) / denom
            )
            self._calculated_metric[denom == 0] = 0.0
        elif self.references.dtype in (np.bool_, bool):
            self._calculated_metric = np.logical_xor(self.computed, self.references)
        else:
            raise TypeError(
                f"recieved data with unexpected dtype {self.references.dtype}"
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

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        report = []
        report.append("✅ Success" if self.check else "❌ Numerical failures")

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
            "Index  Computed  Reference  Absloute E  Metric E",
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

            if np.isnan(metric_err) or (metric_err > worst_metric_err):
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

        return "\n".join(report)


class MultiModalFloatMetric(BaseMetric):
    """Combination of absolute, relative & ULP comparison for floats

    This metric attempts to combine well known comparison on floats
    to leverage a robust 32/64 bit float comparison on large accumulating
    floating errors.

    ULP is used to clear noise (ULP<=1.0 passes)
    Absolute errors for large amplitute
    """

    _f32_absolute_eps = 1e-10
    _f64_absolute_eps = 1e-13

    def __init__(
        self,
        reference_values: np.ndarray,
        computed_values: np.ndarray,
        eps: float,
        **kwargs,
    ):
        super().__init__(reference_values, computed_values)
        self.absolute_distance = np.empty_like(self.references)
        self.absolute_distance_metric = np.empty_like(self.references, dtype=np.bool_)
        self.relative_distance = np.empty_like(self.references)
        self.relative_distance_metric = np.empty_like(self.references, dtype=np.bool_)
        self.ulp_distance = np.empty_like(self.references)
        self.ulp_distance_metric = np.empty_like(self.references, dtype=np.bool_)

        self.relative_fraction = 0.000001
        if self.references.dtype is (np.float32, np.int32):
            self.absolute_eps = max(eps, self._f32_absolute_eps)
        else:
            self.absolute_eps = max(eps, self._f64_absolute_eps)
        self.ulp_threshold = 1.0

        self.success = self._compute_all_metrics()
        self.check = np.all(self.success)

    def _compute_all_metrics(
        self,
    ) -> npt.NDArray[np.bool_]:
        if self.references.dtype in (np.float64, np.int64, np.float32, np.int32):
            max_values = np.maximum(
                np.absolute(self.computed), np.absolute(self.references)
            )
            # Absolute distance
            self.absolute_distance = np.absolute(self.computed - self.references)
            self.absolute_distance_metric = self.absolute_distance < self.absolute_eps
            # Relative distance (in pct)
            self.relative_distance = np.divide(self.absolute_distance, max_values)
            self.relative_distance_metric = (
                self.absolute_distance < self.relative_fraction * max_values
            )
            # ULP distance
            self.ulp_distance = np.divide(
                self.absolute_distance, np.spacing(max_values)
            )
            self.ulp_distance_metric = self.ulp_distance <= self.ulp_threshold

            # Combine all distances into sucess or failure
            # Success = no NANs & ( abs or rel or ulp )
            naninf_success = not np.logical_and(
                np.isnan(self.computed), np.isnan(self.references)
            ).all()
            metric_success = np.logical_or(
                self.relative_distance_metric, self.absolute_distance_metric
            )
            metric_success = np.logical_or(metric_success, self.ulp_distance_metric)
            success = np.logical_and(naninf_success, metric_success)
            return success
        elif self.references.dtype in (np.bool_, bool):
            success = np.logical_xor(self.computed, self.references)
            return success
        else:
            raise TypeError(
                f"recieved data with unexpected dtype {self.references.dtype}"
            )

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        report = []
        report.append("✅ Success" if self.check else "❌ Numerical failures")

        found_indices = np.logical_not(self.success).nonzero()
        # List all errors
        bad_indices_count = len(found_indices[0])
        full_count = len(self.references.flatten())
        failures_pct = round(100.0 * (bad_indices_count / full_count), 2)
        report = [
            f"All failures ({bad_indices_count}/{full_count}) ({failures_pct}%),\n",
            f"Index   Computed   Reference   "
            f"Absolute E(<{self.absolute_eps:.2e})  "
            f"Relative E(<{self.relative_fraction*100:.2e}%)   "
            f"ULP E(<{self.ulp_threshold})",
        ]
        # Summary and worst result
        for iBad in range(bad_indices_count):
            fi = tuple([f[iBad] for f in found_indices])
            report.append(
                f"({fi[0]:02}, {fi[1]:02}, {fi[2]:02})  {self.computed[fi]:.16e}  {self.references[fi]:.16e}  "
                f"{self.absolute_distance[fi]:.2e} {'✅' if self.absolute_distance_metric[fi] else '❌'}  "
                f"{self.relative_distance[fi] * 100:.2e} {'✅' if self.relative_distance_metric[fi] else '❌'}  "
                f"{int(self.ulp_distance[fi]):02} {'✅' if self.ulp_distance_metric[fi] else '❌'}  "
            )

        return "\n".join(report)
