"""
Compute a set of observables from truncated-Wigner trajectories.

The input trajectory convention is
    (number_trajectories, number_modes, number_time_points).

All operator expectation values are evaluated with the appropriate Weyl
corrections. The two pair channels follow the mode ordering used in
``twa/eoms.py``, i.e.,
    plus channel:  modes 1 and 2
    minus channel: modes 3 and 4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

PAIR_CHANNELS: Mapping[str, tuple[int, int]] = {
    "plus": (1, 2),
    "minus": (3, 4),
}


@dataclass(frozen=True)
class RealMean:
    """Ensemble mean and standard error of a real-valued observable."""

    mean: RealArray
    standard_error: RealArray


@dataclass(frozen=True)
class ComplexMean:
    """
    Ensemble mean and component-wise standard error of a complex observable.

    The real and imaginary parts are treated as two directly sampled
    real-valued quantities.
    """

    mean: ComplexArray
    standard_error_real: RealArray
    standard_error_imag: RealArray


@dataclass(frozen=True)
class PairChannelObservables:
    """Observables for one pair-production channel."""

    mode_indices: tuple[int, int]
    mode_a_population: RealMean
    mode_b_population: RealMean
    pair_number: RealMean
    population_difference: RealMean
    imbalance_jz: RealMean
    imbalance_variance: RealArray
    relative_number_squeezing: RealArray
    metrological_gain: RealArray
    pair_coherence: ComplexMean
    cross_correlation: RealArray
    autocorrelation_a: RealArray
    autocorrelation_b: RealArray
    cauchy_schwarz_ratio: RealArray


@dataclass(frozen=True)
class ObservableResults:
    """Complete observable results for one trajectory ensemble."""

    mode_populations: RealMean
    pair_channels: dict[str, PairChannelObservables]


def _validate_trajectories(trajectories: ComplexArray) -> ComplexArray:
    """Validate and return the trajectory array as complex128."""

    array = np.asarray(trajectories, dtype=np.complex128)

    if array.ndim != 3:
        raise ValueError(
            "Trajectories must have shape "
            "(number_trajectories, number_modes, number_time_points)."
        )

    if array.shape[0] < 1:
        raise ValueError("At least one trajectory is required.")

    if array.shape[1] != 6:
        raise ValueError(
            "The six-mode model requires exactly six stored modes. "
            f"Received {array.shape[1]}."
        )

    if array.shape[2] < 1:
        raise ValueError("At least one stored time point is required.")

    if not np.all(np.isfinite(array)):
        raise ValueError("The trajectory array contains NaN or infinite values.")

    return array


def _standard_error_real(samples: RealArray) -> RealArray:
    """
    Return the standard error of a real ensemble mean.

    For a single trajectory, the sampling error cannot be estimated and NaN is
    returned.
    """

    number_trajectories = samples.shape[0]

    if number_trajectories < 2:
        return np.full(samples.shape[1:], np.nan, dtype=np.float64)

    return np.std(samples, axis=0, ddof=1) / np.sqrt(number_trajectories)


def _real_mean(samples: RealArray) -> RealMean:
    """Return the ensemble mean and its standard error."""

    real_samples = np.asarray(samples, dtype=np.float64)

    return RealMean(
        mean=np.mean(real_samples, axis=0),
        standard_error=_standard_error_real(real_samples),
    )


def _complex_mean(samples: ComplexArray) -> ComplexMean:
    """Return a complex ensemble mean and component-wise standard errors."""

    complex_samples = np.asarray(samples, dtype=np.complex128)

    return ComplexMean(
        mean=np.mean(complex_samples, axis=0),
        standard_error_real=_standard_error_real(complex_samples.real),
        standard_error_imag=_standard_error_real(complex_samples.imag),
    )


def mode_population_samples(trajectories: ComplexArray) -> RealArray:
    r"""
    Return trajectory-resolved, Weyl-corrected mode populations.

    For every bosonic mode,

        <n> = <|alpha|^2 - 1/2>_W.
    """

    array = _validate_trajectories(trajectories)
    return np.abs(array) ** 2 - 0.5


def _safe_ratio(
    numerator: RealArray,
    denominator: RealArray,
    denominator_tolerance: float,
) -> RealArray:
    """Divide only where the denominator is safely positive."""

    result = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > denominator_tolerance)
    )
    result[valid] = numerator[valid] / denominator[valid]
    return result


def compute_pair_channel_observables(
    trajectories: ComplexArray,
    mode_a: int,
    mode_b: int,
    denominator_tolerance: float = 1.0e-12,
) -> PairChannelObservables:
    r"""
    Compute the observable set for one pair-production channel.

    Definitions
    -----------
    Pair number:
        N_p = (N_a + N_b) / 2
    Population difference:
        Delta N = N_a - N_b
    Imbalance:
        J_z = (N_a - N_b) / 2
    Corrected imbalance variance:
        Var(J_z) = Var[(J_z)_W] - 1/8
    Relative-number squeezing:
        zeta_s^2 = 2 Var(J_z) / <N_p>
    Metrological gain:
        G = 1 / zeta_s^2
    Pair coherence:
        <a b> = <alpha_a alpha_b>_W
    Cauchy--Schwarz ratio:
        C = G_ab^(2) / sqrt(G_aa^(2) G_bb^(2))

    No Weyl correction is required for <ab> because the annihilation
    operators belong to distinct commuting modes.
    """

    array = _validate_trajectories(trajectories)

    if mode_a == mode_b:
        raise ValueError("A pair channel must contain two distinct modes.")

    for mode_index in (mode_a, mode_b):
        if not 0 <= mode_index < array.shape[1]:
            raise IndexError(
                f"Mode index {mode_index} lies outside the array."
            )

    populations = mode_population_samples(array)

    population_a_samples = populations[:, mode_a, :]
    population_b_samples = populations[:, mode_b, :]

    pair_number_samples = 0.5 * (
        population_a_samples + population_b_samples
    )

    population_difference_samples = (
        population_a_samples - population_b_samples
    )

    imbalance_samples = 0.5 * population_difference_samples

    if array.shape[0] < 2:
        imbalance_wigner_variance = np.full(
            array.shape[2],
            np.nan,
            dtype=np.float64,
        )
    else:
        imbalance_wigner_variance = np.var(
            imbalance_samples,
            axis=0,
            ddof=1,
        )

    # Weyl correction for Var(J_z)
    imbalance_variance = (
        imbalance_wigner_variance - 1.0 / 8.0
    )

    pair_number = _real_mean(pair_number_samples)

    relative_number_squeezing = _safe_ratio(
        numerator=2.0 * imbalance_variance,
        denominator=pair_number.mean,
        denominator_tolerance=denominator_tolerance,
    )

    metrological_gain = np.full_like(
        relative_number_squeezing,
        np.nan,
        dtype=np.float64,
    )

    valid_gain = (
        np.isfinite(relative_number_squeezing)
        & (relative_number_squeezing > 0.0)
    )

    metrological_gain[valid_gain] = (
        1.0 / relative_number_squeezing[valid_gain]
    )

    # Pair coherence

    pair_coherence_samples = (
        array[:, mode_a, :] * array[:, mode_b, :]
    )

    # Cauchy-Schwarz correlations

    absolute_square_a = np.abs(
        array[:, mode_a, :]
    ) ** 2

    absolute_square_b = np.abs(
        array[:, mode_b, :]
    ) ** 2

    # <n_a n_b>

    cross_correlation_samples = (
        absolute_square_a - 0.5
    ) * (
        absolute_square_b - 0.5
    )

    # <n_a(n_a-1)>

    autocorrelation_a_samples = (
        absolute_square_a**2
        - 2.0 * absolute_square_a
        + 0.5
    )

    # <n_b(n_b-1)>

    autocorrelation_b_samples = (
        absolute_square_b**2
        - 2.0 * absolute_square_b
        + 0.5
    )

    cross_correlation = np.mean(
        cross_correlation_samples,
        axis=0,
    )

    autocorrelation_a = np.mean(
        autocorrelation_a_samples,
        axis=0,
    )

    autocorrelation_b = np.mean(
        autocorrelation_b_samples,
        axis=0,
    )

    denominator = np.full_like(
        cross_correlation,
        np.nan,
        dtype=np.float64,
    )

    valid_denominator = (
        np.isfinite(autocorrelation_a)
        & np.isfinite(autocorrelation_b)
        & (autocorrelation_a > denominator_tolerance)
        & (autocorrelation_b > denominator_tolerance)
    )

    denominator[valid_denominator] = np.sqrt(
        autocorrelation_a[valid_denominator]
        * autocorrelation_b[valid_denominator]
    )

    cauchy_schwarz_ratio = _safe_ratio(
        numerator=cross_correlation,
        denominator=denominator,
        denominator_tolerance=denominator_tolerance,
    )

    return PairChannelObservables(
        mode_indices=(mode_a, mode_b),
        mode_a_population=_real_mean(
            population_a_samples,
        ),
        mode_b_population=_real_mean(
            population_b_samples,
        ),
        pair_number=pair_number,
        population_difference=_real_mean(
            population_difference_samples,
        ),
        imbalance_jz=_real_mean(
            imbalance_samples,
        ),
        imbalance_variance=np.asarray(
            imbalance_variance,
            dtype=np.float64,
        ),
        relative_number_squeezing=relative_number_squeezing,
        metrological_gain=metrological_gain,
        pair_coherence=_complex_mean(
            pair_coherence_samples,
        ),
        cross_correlation=np.asarray(
            cross_correlation,
            dtype=np.float64,
        ),
        autocorrelation_a=np.asarray(
            autocorrelation_a,
            dtype=np.float64,
        ),
        autocorrelation_b=np.asarray(
            autocorrelation_b,
            dtype=np.float64,
        ),
        cauchy_schwarz_ratio=cauchy_schwarz_ratio,
    )


def compute_observables(
    trajectories: ComplexArray,
    pair_channels: Mapping[str, tuple[int, int]] = PAIR_CHANNELS,
    denominator_tolerance: float = 1.0e-12,
) -> ObservableResults:
    """
    Compute the Weyl-corrected observable set for all pair channels.

    Standard errors are returned for mode populations, pair number,
    population difference, imbalance, and pair coherence. No uncertainty is
    assigned to derived quantities or normally ordered correlation functions.
    """

    array = _validate_trajectories(trajectories)
    population_samples = mode_population_samples(array)

    channel_results = {
        channel_name: compute_pair_channel_observables(
            trajectories=array,
            mode_a=mode_indices[0],
            mode_b=mode_indices[1],
            denominator_tolerance=denominator_tolerance,
        )
        for channel_name, mode_indices in pair_channels.items()
    }

    return ObservableResults(
        mode_populations=_real_mean(population_samples),
        pair_channels=channel_results,
    )

# ======================================================================
# TODO (future work)
# ----------------------------------------------------------------------
# - Spin-nematic squeezing
# - Wineland parameter
# - General SU(1,1) quadratures
# ======================================================================
