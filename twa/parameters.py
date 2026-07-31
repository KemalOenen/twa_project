"""
Central parameter definitions for the truncated-Wigner simulations.

The values to be changed by the user are collected near the bottom
of this file in 4 clearly separated blocks:

    COMMON
    SINGLE_PUMP
    TWO_PUMP
    SIMULATION

Frequency convention
--------------------
Experimental frequencies are entered as f = omega / (2*pi) in Hz.
Internally, all angular frequencies and coupling rates are converted to
rad/ms. The time array is consequently expressed in ms.

For example:
    kappa_hz = 1.25e6
means
    kappa / (2*pi) = 1.25 MHz.

The corresponding internal value is
    kappa = 2*pi*1.25e6 / 1000 rad/ms.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.constants import hbar, pi


# ----------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------

MASS_RB87 = 1.443160648e-25  # Mass of one Rb-87 atom in kg


# ----------------------------------------------------------------------
# Unit-conversion functions
# ----------------------------------------------------------------------

def hz_to_rad_per_ms(frequency_hz: float) -> float:
    """
    Convert f = omega/(2*pi), given in Hz, to omega in rad/ms.
    """

    return 2.0 * pi * frequency_hz / 1000.0


def complex_hz_to_rad_per_ms(
    magnitude_hz: float,
    phase_rad: float = 0.0,
) -> complex:
    """
    Construct a complex angular frequency in rad/ms.

    Parameters
    ----------
    magnitude_hz
        Magnitude expressed as eta/(2*pi) in Hz.
    phase_rad
        Complex phase in radians.
    """

    magnitude = hz_to_rad_per_ms(magnitude_hz)
    return magnitude * np.exp(1j * phase_rad)


# ----------------------------------------------------------------------
# Common physical parameters
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class CommonParameters:
    """
    Parameters shared by the single-pump and two-pump models.

    Parameters
    ----------
    atom_number
        Mean number of atoms initially occupying the condensate.
    atom_number_std
        Shot-to-shot standard deviation of the initial atom number.
    wavelength_m
        Optical wavelength in meter used to calculate the recoil frequency.
        For the two-pump model, the two laser wavelengths differ only
        negligibly, so a single representative wavelength is used.
    magnetic_field_g
        Magnetic field in gauss.
    q_hz_per_g2
        Magnitude of q/(2*pi*B^2) in Hz/G^2.
        The sign appropriate to the hyperfine manifold is introduced
        when the pair energy is calculated.
    kappa_hz
        Cavity-field decay rate specified as kappa/(2*pi) in Hz.
    mode_mixing
        Coefficient a in
            psi_0 = psi_(0) + sqrt(a) psi_(higher).
        The six-mode model uses a = 2/3.
    """

    atom_number: float
    atom_number_std: float
    wavelength_m: float
    magnetic_field_g: float
    q_hz_per_g2: float
    kappa_hz: float
    mode_mixing: float = 2.0 / 3.0

    @property
    def wave_number(self) -> float:
        """Optical wave number k in rad/m."""

        return 2.0 * pi / self.wavelength_m

    @property
    def omega_rec(self) -> float:
        """Recoil angular frequency in rad/ms."""

        omega_rec_rad_per_s = (
            hbar * self.wave_number**2 / (2.0 * MASS_RB87)
        )
        return omega_rec_rad_per_s / 1000.0

    @property
    def q(self) -> float:
        """
        Positive magnitude of the quadratic Zeeman frequency in rad/ms.
        The magnetic-field dependence is q proportional to B^2.
        """

        q_hz = self.q_hz_per_g2 * self.magnetic_field_g**2
        return hz_to_rad_per_ms(q_hz)

    @property
    def kappa(self) -> float:
        """Cavity decay rate in rad/ms."""

        return hz_to_rad_per_ms(self.kappa_hz)

    @property
    def pair_energy_single_pump(self) -> float:
        """
        Bare pair energy omega_0 for the F = 1 single-pump model.
        omega_0 = 4 omega_rec + 2 q
        """

        return 4.0 * self.omega_rec + 2.0 * self.q

    @property
    def pair_energy_two_pump(self) -> float:
        """
        Bare pair energy omega_0 for the F = 2 two-pump model.
        omega_0 = 4 omega_rec - 2 q
        """

        return 4.0 * self.omega_rec - 2.0 * self.q

    @property
    def omega_pair_single_pump(self) -> float:
        """Free frequency omega_0/2 entering each single-pump side mode."""

        return 0.5 * self.pair_energy_single_pump

    @property
    def omega_pair_two_pump(self) -> float:
        """Free frequency omega_0/2 entering each two-pump side mode."""

        return 0.5 * self.pair_energy_two_pump


# ----------------------------------------------------------------------
# Single-pump parameters
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SinglePumpParameters:
    """
    User inputs for the single-pump model.

    Parameters
    ----------
    eta_hz
        Raman coupling magnitude specified as eta/(2*pi) in Hz.
    delta_plus_hz
        Signed detuning delta_plus/(2*pi) in Hz.
    delta_minus_hz
        Signed detuning delta_minus/(2*pi) in Hz.
    """

    eta_hz: float
    delta_plus_hz: float
    delta_minus_hz: float

    @property
    def eta(self) -> float:
        """Single-pump Raman coupling eta in rad/ms."""

        return hz_to_rad_per_ms(self.eta_hz)

    @property
    def delta_plus(self) -> float:
        """Plus-channel detuning in rad/ms."""

        return hz_to_rad_per_ms(self.delta_plus_hz)

    @property
    def delta_minus(self) -> float:
        """Minus-channel detuning in rad/ms."""

        return hz_to_rad_per_ms(self.delta_minus_hz)

    def chi_plus(self, common: CommonParameters) -> float:
        """Coherent coupling of the plus channel in rad/ms."""

        return (
            self.eta**2
            * self.delta_plus
            / (self.delta_plus**2 + common.kappa**2)
        )

    def chi_minus(self, common: CommonParameters) -> float:
        """Coherent coupling of the minus channel in rad/ms."""

        return (
            self.eta**2
            * self.delta_minus
            / (self.delta_minus**2 + common.kappa**2)
        )

    def gamma_plus(self, common: CommonParameters) -> float:
        """Dissipative coupling of the plus channel in rad/ms."""

        return (
            2.0
            * self.eta**2
            * common.kappa
            / (self.delta_plus**2 + common.kappa**2)
        )

    def gamma_minus(self, common: CommonParameters) -> float:
        """Dissipative coupling of the minus channel in rad/ms."""

        return (
            2.0
            * self.eta**2
            * common.kappa
            / (self.delta_minus**2 + common.kappa**2)
        )

    def eom_arguments(self, common: CommonParameters) -> Dict[str, float]:
        """
        Return the parameters required by eoms_single_pump().
        """

        return {
            "gamma_plus": self.gamma_plus(common),
            "gamma_minus": self.gamma_minus(common),
            "chi_plus": self.chi_plus(common),
            "chi_minus": self.chi_minus(common),
            "omega_pair": common.omega_pair_single_pump,
            "mode_mixing": common.mode_mixing,
            "omega_rec": common.omega_rec,
        }


# ----------------------------------------------------------------------
# Two-pump parameters
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class TwoPumpParameters:
    """
    User inputs for the two-colour model (cf. Thesis by Kemal Önen)

    Parameters
    ----------
    eta_a_hz, eta_b_hz
        Raman-coupling magnitudes specified as |eta_A|/(2*pi) and
        |eta_B|/(2*pi) in Hz.
    phase_a_rad, phase_b_rad
        Optical phases of eta_A and eta_B in radians.

        Only their difference is physically relevant. It is often
        convenient to set phase_a_rad = 0 and vary phase_b_rad.
    relative_phase_std_rad
        Shot-to-shot standard deviation of the relative pump phase.
        A new relative phase is sampled for each trajectory.
    delta_plus_hz
        Signed detuning delta_plus/(2*pi) in Hz.
    delta_minus_hz
        Signed detuning delta_minus/(2*pi) in Hz.

    Detuning assignment
    -------------------
    We use the convention (note difference to single pump!)
        chi_ij^+   and gamma_ij^+   with delta_minus,
        chi_ij^-   and gamma_ij^-   with delta_plus.
    """

    eta_a_hz: float
    eta_b_hz: float
    phase_a_rad: float
    phase_b_rad: float
    relative_phase_std_rad: float
    delta_plus_hz: float
    delta_minus_hz: float

    @property
    def eta_a(self) -> complex:
        """Complex Raman coupling eta_A in rad/ms."""

        return complex_hz_to_rad_per_ms(
            self.eta_a_hz,
            self.phase_a_rad,
        )

    @property
    def eta_b(self) -> complex:
        """Complex Raman coupling eta_B in rad/ms."""

        return complex_hz_to_rad_per_ms(
            self.eta_b_hz,
            self.phase_b_rad,
        )

    @property
    def relative_phase(self) -> float:
        """Relative pump phase phase_B - phase_A in radians."""

        return self.phase_b_rad - self.phase_a_rad

    @property
    def delta_plus(self) -> float:
        """Detuning delta_plus in rad/ms."""

        return hz_to_rad_per_ms(self.delta_plus_hz)

    @property
    def delta_minus(self) -> float:
        """Detuning delta_minus in rad/ms."""

        return hz_to_rad_per_ms(self.delta_minus_hz)

    def _chi(
        self,
        eta_i: complex,
        eta_j: complex,
        detuning: float,
        common: CommonParameters,
    ) -> complex:
        """Calculate one coherent two-pump coefficient."""

        return (
            np.conj(eta_i)
            * eta_j
            * detuning
            / (detuning**2 + common.kappa**2)
        )

    def _gamma(
        self,
        eta_i: complex,
        eta_j: complex,
        detuning: float,
        common: CommonParameters,
    ) -> complex:
        """Calculate one dissipative two-pump coefficient."""

        return (
            2.0
            * common.kappa
            * np.conj(eta_i)
            * eta_j
            / (detuning**2 + common.kappa**2)
        )

    def chi_aa_plus(self, common: CommonParameters) -> float:
        return float(
            np.real(
                self._chi(
                    self.eta_a,
                    self.eta_a,
                    self.delta_minus,
                    common,
                )
            )
        )

    def chi_bb_plus(self, common: CommonParameters) -> float:
        return float(
            np.real(
                self._chi(
                    self.eta_b,
                    self.eta_b,
                    self.delta_minus,
                    common,
                )
            )
        )

    def chi_ab_plus(self, common: CommonParameters) -> complex:
        return self._chi(
            self.eta_a,
            self.eta_b,
            self.delta_minus,
            common,
        )

    def chi_aa_minus(self, common: CommonParameters) -> float:
        return float(
            np.real(
                self._chi(
                    self.eta_a,
                    self.eta_a,
                    self.delta_plus,
                    common,
                )
            )
        )

    def chi_bb_minus(self, common: CommonParameters) -> float:
        return float(
            np.real(
                self._chi(
                    self.eta_b,
                    self.eta_b,
                    self.delta_plus,
                    common,
                )
            )
        )

    def chi_ab_minus(self, common: CommonParameters) -> complex:
        return self._chi(
            self.eta_a,
            self.eta_b,
            self.delta_plus,
            common,
        )

    def gamma_aa_plus(self, common: CommonParameters) -> float:
        return float(
            np.real(
                self._gamma(
                    self.eta_a,
                    self.eta_a,
                    self.delta_minus,
                    common,
                )
            )
        )

    def gamma_bb_plus(self, common: CommonParameters) -> float:
        return float(
            np.real(
                self._gamma(
                    self.eta_b,
                    self.eta_b,
                    self.delta_minus,
                    common,
                )
            )
        )

    def gamma_ab_plus(self, common: CommonParameters) -> complex:
        return self._gamma(
            self.eta_a,
            self.eta_b,
            self.delta_minus,
            common,
        )

    def gamma_aa_minus(self, common: CommonParameters) -> float:
        return float(
            np.real(
                self._gamma(
                    self.eta_a,
                    self.eta_a,
                    self.delta_plus,
                    common,
                )
            )
        )

    def gamma_bb_minus(self, common: CommonParameters) -> float:
        return float(
            np.real(
                self._gamma(
                    self.eta_b,
                    self.eta_b,
                    self.delta_plus,
                    common,
                )
            )
        )

    def gamma_ab_minus(self, common: CommonParameters) -> complex:
        return self._gamma(
            self.eta_a,
            self.eta_b,
            self.delta_plus,
            common,
        )

    def eom_arguments(
        self,
        common: CommonParameters,
    ) -> Dict[str, complex | float]:
        """
        Return keyword arguments matching eoms_two_pump().
        """

        return {
            "chi_aa_plus": self.chi_aa_plus(common),
            "chi_bb_plus": self.chi_bb_plus(common),
            "chi_ab_plus": self.chi_ab_plus(common),
            "chi_aa_minus": self.chi_aa_minus(common),
            "chi_bb_minus": self.chi_bb_minus(common),
            "chi_ab_minus": self.chi_ab_minus(common),
            "gamma_aa_plus": self.gamma_aa_plus(common),
            "gamma_bb_plus": self.gamma_bb_plus(common),
            "gamma_ab_plus": self.gamma_ab_plus(common),
            "gamma_aa_minus": self.gamma_aa_minus(common),
            "gamma_bb_minus": self.gamma_bb_minus(common),
            "gamma_ab_minus": self.gamma_ab_minus(common),
            "omega_pair": common.omega_pair_two_pump,
            "mode_mixing": common.mode_mixing,
            "omega_rec": common.omega_rec,
        }


# ----------------------------------------------------------------------
# Numerical simulation parameters
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationParameters:
    """
    Settings used by the sampling and simulation modules.

    Parameters
    ----------
    time_start_ms, time_stop_ms
        Beginning and end of the integration interval in ms.
    number_time_points
        Number of stored time samples.
    number_trajectories
        Number of stochastic truncated-Wigner trajectories.
    initial_pair_seed
        Mean classical seed population placed in each pair mode.
        Set this to zero for vacuum-seeded simulations.
    random_seed
        Seed used to initialise numpy.random.Generator.
    solver_method
        Integration method passed to scipy.integrate.solve_ivp.
    absolute_tolerance, relative_tolerance
        Solver error tolerances.
    """

    time_start_ms: float
    time_stop_ms: float
    number_time_points: int
    number_trajectories: int
    initial_pair_seed: float
    random_seed: int
    solver_method: str = "DOP853" # Explicit Runge-Kutta method of order 8
    absolute_tolerance: float = 1e-10  # chosen tolerance
    relative_tolerance: float = 1e-8  # cf. Rodrigo's thesis

    @property
    def time(self) -> np.ndarray:
        """Simulation time array in ms."""

        return np.linspace(
            self.time_start_ms,
            self.time_stop_ms,
            self.number_time_points,
        )


# ======================================================================
# USER INPUT REGION
# ======================================================================

# ----------------------------------------------------------------------
# Parameters shared by both models
# ----------------------------------------------------------------------

COMMON = CommonParameters(
    atom_number=50_000.0,
    atom_number_std=0.05 * 50_000.0,
    wavelength_m=790.02e-9,
    magnetic_field_g=5.00,
    q_hz_per_g2=72.0,
    kappa_hz=1.25e6,
    mode_mixing=2.0 / 3.0,
)


# ----------------------------------------------------------------------
# Single-pump model
# ----------------------------------------------------------------------

SINGLE_PUMP = SinglePumpParameters(
    eta_hz=2.5e3,
    delta_plus_hz=-20.0e6,
    delta_minus_hz=-50.0e6,
)


# ----------------------------------------------------------------------
# Two-pump model
# ----------------------------------------------------------------------

TWO_PUMP = TwoPumpParameters(
    eta_a_hz=2.5e3,
    eta_b_hz=2.5e3 * np.sqrt(6.0),
    phase_a_rad=0.0,
    phase_b_rad=0.0,
    relative_phase_std_rad=0.0,
    delta_plus_hz=-20.0e6,
    delta_minus_hz=-50.0e6,
)


# ----------------------------------------------------------------------
# Numerical simulation
# ----------------------------------------------------------------------

SIMULATION = SimulationParameters(
    time_start_ms=0.0,
    time_stop_ms=0.2,
    number_time_points=2000,
    number_trajectories=1000,
    initial_pair_seed=0.0,
    random_seed=12345,
    solver_method="DOP853",
    absolute_tolerance=1e-10,
    relative_tolerance=1e-8,
)


# ----------------------------------------------------------------------
# Optional quick inspection
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Internal units: angular frequencies in rad/ms, time in ms")
    print()
    print("omega_rec =", COMMON.omega_rec)
    print("q =", COMMON.q)
    print("omega_pair_single_pump =", COMMON.omega_pair_single_pump)
    print("omega_pair_two_pump =", COMMON.omega_pair_two_pump)
    print()
    print("Single-pump EOM arguments:")
    for name, value in SINGLE_PUMP.eom_arguments(COMMON).items():
        print(f"  {name}: {value}")
    print()
    print("Two-pump EOM arguments:")
    for name, value in TWO_PUMP.eom_arguments(COMMON).items():
        print(f"  {name}: {value}")
