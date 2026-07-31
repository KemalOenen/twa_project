"""
Run and store truncated-Wigner simulations for the single- or two-pump model.

The script samples independent initial Wigner states, integrates one
deterministic trajectory for each sample, and stores the complete ensemble for
later analysis.

Output convention
-----------------
The trajectory array has shape

    (number_trajectories, number_modes, number_time_points).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from twa.eoms import eoms_single_pump, eoms_two_pump
from twa.parameters import COMMON, SIMULATION, SINGLE_PUMP, TWO_PUMP
from twa.sampling import (
    TrajectorySample,
    create_rng,
    sample_single_pump_trajectory,
    sample_two_pump_trajectory,
)


# ----------------------------------------------------------------------
# User choice
# ----------------------------------------------------------------------

# Select the model to be propagated when this file is executed.
MODEL: Literal["single_pump", "two_pump"] = "two_pump"


# ----------------------------------------------------------------------
# Paths and type aliases
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"

ComplexState = NDArray[np.complex128]
TrajectoryArray = NDArray[np.complex128]
EomFunction = Callable[..., ComplexState]
SamplerFunction = Callable[..., TrajectorySample]


# ----------------------------------------------------------------------
# Model-specific information
# ----------------------------------------------------------------------

SINGLE_PUMP_MODE_NAMES = np.array(
    [
        "(0,0)",
        "(+k,+1)",
        "(-k,-1)",
        "(+k,-1)",
        "(-k,+1)",
        "(+-2kx,0)",
    ],
    dtype=str,
)

TWO_PUMP_MODE_NAMES = np.array(
    [
        "(0,F=1,mF=0)",
        "(+k,F=2,mF=+1)",
        "(-k,F=2,mF=-1)",
        "(+k,F=2,mF=-1)",
        "(-k,F=2,mF=+1)",
        "(+-2kx,F=1,mF=0)",
    ],
    dtype=str,
)


def select_model_components(
    model: str,
) -> tuple[EomFunction, SamplerFunction, np.ndarray]:
    """
    Return the EOM, sampler and mode labels belonging to one model.

    Parameters
    ----------
    model
        Either ``"single_pump"`` or ``"two_pump"``.

    Returns
    -------
    eom
        Right-hand side passed to ``scipy.integrate.solve_ivp``.
    sampler
        Function creating one stochastic trajectory sample.
    mode_names
        Labels following the mode ordering in ``twa/eoms.py``.
    """

    if model == "single_pump":
        return (
            eoms_single_pump,
            sample_single_pump_trajectory,
            SINGLE_PUMP_MODE_NAMES,
        )

    if model == "two_pump":
        return (
            eoms_two_pump,
            sample_two_pump_trajectory,
            TWO_PUMP_MODE_NAMES,
        )

    raise ValueError(
        "Unknown model. Choose either 'single_pump' or 'two_pump'."
    )


# ----------------------------------------------------------------------
# Numerical propagation
# ----------------------------------------------------------------------

def integrate_trajectory(
    sample: TrajectorySample,
    eom: EomFunction,
    time_ms: NDArray[np.float64],
) -> TrajectoryArray:
    """
    Integrate one sampled truncated-Wigner trajectory.

    Parameters
    ----------
    sample
        Initial state and trajectory-specific EOM coefficients.
    eom
        Equation-of-motion function for the selected model.
    time_ms
        Time points at which the trajectory is stored, in ms.

    Returns
    -------
    trajectory
        Complex array with shape ``(6, number_time_points)``.

    Raises
    ------
    RuntimeError
        If the numerical solver does not complete successfully.
    ValueError
        If an input or output shape is inconsistent, or if a non-finite value
        occurs during propagation.
    """

    initial_state = np.asarray(sample.initial_state, dtype=np.complex128)

    if initial_state.shape != (6,):
        raise ValueError(
            "Every initial state must contain six complex amplitudes. "
            f"Received shape {initial_state.shape}."
        )

    if time_ms.ndim != 1 or time_ms.size < 2:
        raise ValueError("The simulation time array must contain at least two points.")

    if not np.all(np.isfinite(initial_state)):
        raise ValueError("The sampled initial state contains NaN or infinite values.")

    def right_hand_side(
        time_value: float,
        state: ComplexState,
    ) -> ComplexState:
        return eom(
            time_value,
            state,
            **sample.eom_arguments,
        )

    solution = solve_ivp(
        fun=right_hand_side,
        t_span=(float(time_ms[0]), float(time_ms[-1])),
        y0=initial_state,
        t_eval=time_ms,
        method=SIMULATION.solver_method,
        atol=SIMULATION.absolute_tolerance,
        rtol=SIMULATION.relative_tolerance,
    )

    if not solution.success:
        raise RuntimeError(
            "The numerical integrator failed for one trajectory: "
            f"{solution.message}"
        )

    expected_shape = (6, time_ms.size)
    if solution.y.shape != expected_shape:
        raise ValueError(
            "The numerical solution has an unexpected shape. "
            f"Expected {expected_shape}, received {solution.y.shape}."
        )

    trajectory = np.asarray(solution.y, dtype=np.complex128)
    if not np.all(np.isfinite(trajectory)):
        raise ValueError(
            "The numerical solution contains NaN or infinite amplitudes."
        )

    return trajectory

# TODO: parallelisation
def run_trajectory_ensemble(
    model: Literal["single_pump", "two_pump"],
) -> dict[str, np.ndarray]:
    """
    Sample and serially integrate the complete trajectory ensemble.

    Parameters
    ----------
    model
        Model selected for propagation.

    Returns
    -------
    results
        Dictionary containing the time array, complete trajectories, sampled
        atom numbers, mode labels and, for the two-pump model, sampled relative
        phases.
    """

    eom, sampler, mode_names = select_model_components(model)
    time_ms = np.asarray(SIMULATION.time, dtype=np.float64)
    number_trajectories = SIMULATION.number_trajectories

    if number_trajectories <= 0:
        raise ValueError("The number of trajectories must be positive.")

    trajectories = np.empty(
        (number_trajectories, 6, time_ms.size),
        dtype=np.complex128,
    )
    sampled_atom_numbers = np.empty(number_trajectories, dtype=np.float64)
    sampled_relative_phases = np.full(
        number_trajectories,
        np.nan,
        dtype=np.float64,
    )

    rng = create_rng(SIMULATION.random_seed)

    # Print approximately ten progress updates without flooding the terminal
    progress_interval = max(1, number_trajectories // 10)
    start_time = time.perf_counter()

    for trajectory_index in range(number_trajectories):
        if model == "single_pump":
            sample = sampler(
                rng,
                COMMON,
                SINGLE_PUMP,
                SIMULATION,
            )
        else:
            sample = sampler(
                rng,
                COMMON,
                TWO_PUMP,
                SIMULATION,
            )

        trajectories[trajectory_index] = integrate_trajectory(
            sample,
            eom,
            time_ms,
        )
        sampled_atom_numbers[trajectory_index] = sample.atom_number

        if sample.relative_phase_rad is not None:
            sampled_relative_phases[trajectory_index] = sample.relative_phase_rad

        completed = trajectory_index + 1
        if (
            completed == 1
            or completed % progress_interval == 0
            or completed == number_trajectories
        ):
            elapsed_seconds = time.perf_counter() - start_time
            print(
                f"Completed {completed:>5d}/{number_trajectories} trajectories "
                f"({100.0 * completed / number_trajectories:5.1f} %) "
                f"after {elapsed_seconds:7.1f} s."
            )

    results: dict[str, np.ndarray] = {
        "time_ms": time_ms,
        "trajectories": trajectories,
        "mode_names": mode_names,
        "sampled_atom_numbers": sampled_atom_numbers,
    }

    if model == "two_pump":
        results["sampled_relative_phases_rad"] = sampled_relative_phases

    return results


# ----------------------------------------------------------------------
# Metadata and storage
# ----------------------------------------------------------------------

def create_metadata(
    model: Literal["single_pump", "two_pump"],
    runtime_seconds: float,
) -> dict[str, object]:
    """Create JSON-compatible metadata describing one simulation run."""

    model_parameters = (
        asdict(SINGLE_PUMP)
        if model == "single_pump"
        else asdict(TWO_PUMP)
    )

    return {
        "model": model,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime_seconds": float(runtime_seconds),
        "state_shape_convention": (
            "(number_trajectories, number_modes, number_time_points)"
        ),
        "frequency_units_internal": "rad/ms",
        "time_units": "ms",
        "common_parameters": asdict(COMMON),
        "model_parameters": model_parameters,
        "simulation_parameters": asdict(SIMULATION),
    }


def format_filename_value(value: float, digits: int = 3) -> str:
    """Convert a signed floating-point value into a filename-safe string."""

    formatted = f"{value:.{digits}f}"
    return formatted.replace("-", "m").replace(".", "p")


def create_output_path(
    model: Literal["single_pump", "two_pump"],
) -> Path:
    """Construct a readable, unique output path inside ``data/``."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if model == "single_pump":
        filename = f"single_pump_{timestamp}.npz"
    else:
        mean_phase = format_filename_value(TWO_PUMP.relative_phase)
        phase_std = format_filename_value(TWO_PUMP.relative_phase_std_rad)
        filename = (
            f"two_pump_phi_{mean_phase}_sigma_{phase_std}_{timestamp}.npz"
        )

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return DATA_DIRECTORY / filename


def save_simulation_results(
    output_path: Path,
    results: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    """Store trajectories, sampled variables and metadata in one NPZ file."""

    metadata_json = json.dumps(
        metadata,
        indent=2,
        sort_keys=True,
    )

    np.savez_compressed(
        output_path,
        **results,
        metadata_json=np.array(metadata_json),
    )


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------

def main() -> Path:
    """Run the selected model and save the complete trajectory ensemble."""

    model = MODEL
    select_model_components(model)

    print("=" * 70)
    print("TRUNCATED-WIGNER SIMULATION")
    print("=" * 70)
    print(f"Model:               {model}")
    print(f"Trajectories:        {SIMULATION.number_trajectories}")
    print(f"Stored time points:  {SIMULATION.number_time_points}")
    print(
        "Time interval:       "
        f"{SIMULATION.time_start_ms:.6g} to "
        f"{SIMULATION.time_stop_ms:.6g} ms"
    )
    print(f"Solver:              {SIMULATION.solver_method}")
    print(f"Random seed:         {SIMULATION.random_seed}")

    if model == "two_pump":
        print(f"Mean relative phase: {TWO_PUMP.relative_phase:.6g} rad")
        print(
            "Phase-noise std:     "
            f"{TWO_PUMP.relative_phase_std_rad:.6g} rad"
        )

    print("=" * 70)

    start_time = time.perf_counter()
    results = run_trajectory_ensemble(model)
    runtime_seconds = time.perf_counter() - start_time

    metadata = create_metadata(model, runtime_seconds)
    output_path = create_output_path(model)
    save_simulation_results(output_path, results, metadata)

    maximum_amplitude = float(np.max(np.abs(results["trajectories"])))

    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Runtime:             {runtime_seconds:.1f} s")
    print(f"Saved to:            {output_path}")
    print("=" * 70)

    return output_path


if __name__ == "__main__":
    main()
