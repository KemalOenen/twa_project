"""
Generate one initial condition for a single truncated-Wigner trajectory.

Each call generates one stochastic realization of the experiment by
sampling the initial atom number, the quantum vacuum fluctuations and,
for the two-pump model, the relative pump phase.
"""

from dataclasses import dataclass, replace
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from .parameters import (
    CommonParameters,
    SimulationParameters,
    SinglePumpParameters,
    TwoPumpParameters,
)


ComplexState = NDArray[np.complex128]
EomArguments = Dict[str, complex | float]


@dataclass(frozen=True)
class TrajectorySample:
    """One stochastic realization used by the numerical integrator."""

    initial_state: ComplexState
    eom_arguments: EomArguments
    atom_number: float
    relative_phase_rad: float | None = None


def create_rng(random_seed: int) -> np.random.Generator:
    """Create the random-number generator used for all trajectories."""

    return np.random.default_rng(random_seed)


def sample_atom_number(
    rng: np.random.Generator,
    mean: float,
    standard_deviation: float,
) -> float:
    """Sample a positive initial atom number."""

    if mean < 0.0:
        raise ValueError("The mean atom number must be non-negative.")
    if standard_deviation < 0.0:
        raise ValueError("The atom-number standard deviation must be non-negative.")

    if standard_deviation == 0.0:
        return float(mean)

    atom_number = rng.normal(mean, standard_deviation)
    while atom_number < 0.0:
        atom_number = rng.normal(mean, standard_deviation)

    return float(atom_number)


def sample_wigner_mode(
    rng: np.random.Generator,
    mean_population: float = 0.0,
    phase_rad: float = 0.0,
) -> complex:
    """
    Sample one coherent mode in the Wigner representation.

    The coherent amplitude is sqrt(mean_population) exp(i phase_rad),
    with one half quantum of Gaussian vacuum noise added in each mode.
    """

    if mean_population < 0.0:
        raise ValueError("A mode population cannot be negative.")

    coherent_amplitude = (
        np.sqrt(mean_population) * np.exp(1j * phase_rad)
    )
    vacuum_noise = 0.5 * (
        rng.normal() + 1j * rng.normal()
    )

    return complex(coherent_amplitude + vacuum_noise)


def sample_initial_state(
    rng: np.random.Generator,
    atom_number: float,
    initial_pair_seed: float = 0.0,
) -> ComplexState:
    """
    Sample the six-mode initial state shared by both models.

    The mode ordering follows eoms.py. Mode 0 contains the condensate,
    modes 1-4 are the four pair modes, and mode 5 is the higher-order
    condensate mode. The optional initial_pair_seed specifies the mean
    classical population assigned to each pair mode before adding the
    Wigner vacuum fluctuations.
    """

    if atom_number < 0.0:
        raise ValueError("The initial atom number must be non-negative.")
    if initial_pair_seed < 0.0:
        raise ValueError("The initial pair seed must be non-negative.")

    state = np.empty(6, dtype=np.complex128)
    state[0] = sample_wigner_mode(rng, atom_number)
    state[1:5] = np.array(
        [sample_wigner_mode(rng, initial_pair_seed) for _ in range(4)],
        dtype=np.complex128,
    )
    state[5] = sample_wigner_mode(rng)

    return state

# think of the mean relative phase as the phase difference one locks to experimentally
def sample_relative_phase(
    rng: np.random.Generator,
    mean_relative_phase_rad: float,
    standard_deviation_rad: float,
) -> float:
    """Sample one realization of the relative phase phi_B - phi_A."""

    if standard_deviation_rad < 0.0:
        raise ValueError("The relative-phase standard deviation must be non-negative.")

    if standard_deviation_rad == 0.0:
        return float(mean_relative_phase_rad)

    return float(
        rng.normal(mean_relative_phase_rad, standard_deviation_rad)
    )


def sample_single_pump_trajectory(
    rng: np.random.Generator,
    common: CommonParameters,
    single_pump: SinglePumpParameters,
    simulation: SimulationParameters,
) -> TrajectorySample:
    """Sample one initial realization for the single-pump model."""

    atom_number = sample_atom_number(
        rng,
        common.atom_number,
        common.atom_number_std,
    )
    initial_state = sample_initial_state(
        rng,
        atom_number,
        simulation.initial_pair_seed,
    )

    return TrajectorySample(
        initial_state=initial_state,
        eom_arguments=single_pump.eom_arguments(common),
        atom_number=atom_number,
    )


def sample_two_pump_trajectory(
    rng: np.random.Generator,
    common: CommonParameters,
    two_pump: TwoPumpParameters,
    simulation: SimulationParameters,
) -> TrajectorySample:
    """
    Sample one initial realization for the two-pump model.

    The phase of pump A is kept fixed. A trajectory-specific fluctuation
    is added to the phase of pump B, so that the sampled difference is
    the relative phase entering the cross-coupling coefficients.
    """

    atom_number = sample_atom_number(
        rng,
        common.atom_number,
        common.atom_number_std,
    )
    initial_state = sample_initial_state(
        rng,
        atom_number,
        simulation.initial_pair_seed,
    )

    relative_phase = sample_relative_phase(
        rng,
        two_pump.relative_phase,
        two_pump.relative_phase_std_rad,
    )

    phase_fluctuation = relative_phase - two_pump.relative_phase
    sampled_parameters = replace(
        two_pump,
        phase_b_rad=two_pump.phase_b_rad + phase_fluctuation,
    )

    return TrajectorySample(
        initial_state=initial_state,
        eom_arguments=sampled_parameters.eom_arguments(common),
        atom_number=atom_number,
        relative_phase_rad=relative_phase,
    )