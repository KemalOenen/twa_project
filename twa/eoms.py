"""
Equations of motion (EOMs) for the truncated Wigner approximation (TWA).

This module contains the equations of motion that are integrated
for each stochastic trajectory.
"""

import numpy as np
from numpy.typing import NDArray

def eoms_single_pump(
    t: float,
    psi: NDArray[np.complex128],
    gamma_plus: float,
    gamma_minus: float,
    chi_plus: float,
    chi_minus: float,
    omega_pair: float,
    mode_mixing: float = 2.0 / 3.0,
    omega_rec: float = 0.0,
) -> NDArray[np.complex128]:

    """
    Six-mode equations of motion for Rodrigo's single-pump model.
    The equations directly implement Eq. (6.23) of Rodrigo's thesis.

    Mode ordering
    -------------
    psi[0] = psi_(0,0)
    psi[1] = psi_(+k,+1)
    psi[2] = psi_(-k,-1)
    psi[3] = psi_(+k,-1)
    psi[4] = psi_(-k,+1)
    psi[5] = psi_(±2 k_x,0)

    Parameters
    ----------
    t
        Time variable required by scipy.integrate.solve_ivp.
    psi
        Complex vector containing the six mode amplitudes.
    gamma_plus, gamma_minus
        Dissipative coupling rates of the two pair channels.
    chi_plus, chi_minus
        Coherent coupling rates of the two pair channels.
    omega_pair
        Free evolution frequency of each excited pair mode.
        This corresponds to omega_0 / 2 in Rodrigo's notation.
    mode_mixing
        Coefficient a in the effective condensate amplitude
            psi_0 = psi_(0,0) + sqrt(a) psi_(±2 k_x,0).
        Rodrigo uses a = 2/3.
    omega_rec
        Recoil frequency of the higher-order condensate mode.

    Returns
    -------
    dpsi_dt
        Time derivative of the six complex mode amplitudes.
    """

    if psi.shape != (6,):
        raise ValueError(
            f"Expected a six-component state vector, received shape {psi.shape}."
        )
    # ------------------------------------------------------------------
    # Unpack the six dynamical mode amplitudes.
    # ------------------------------------------------------------------
    psi_00 = psi[0]
    psi_plus_a = psi[1]   # (+k,+1)
    psi_plus_b = psi[2]   # (-k,-1)
    psi_minus_a = psi[3]  # (+k,-1)
    psi_minus_b = psi[4]  # (-k,+1)
    psi_high = psi[5]     # (±2 k_x,0)

    # Effective condensate amplitude entering all interaction terms
    sqrt_a = np.sqrt(mode_mixing)
    psi_0 = psi_00 + sqrt_a * psi_high
    pump_density = np.abs(psi_0) ** 2
    pump_squared = psi_0 ** 2

    # ------------------------------------------------------------------
    # Plus channel:
    # (+k,+1) paired with (-k,-1)
    # ------------------------------------------------------------------
    plus_bracket_a = (
        pump_density * psi_plus_a
        + pump_squared * np.conj(psi_plus_b)
    )
    plus_bracket_b = (
        pump_density * psi_plus_b
        + pump_squared * np.conj(psi_plus_a)
    )
    condensate_plus = (
        -1j
        * chi_plus
        * (
            2.0 * np.conj(psi_0) * psi_plus_a * psi_plus_b
            + psi_0 * np.abs(psi_plus_a) ** 2
            + psi_0 * np.abs(psi_plus_b) ** 2
        )
        + gamma_plus
        * psi_0
        * (
            np.abs(psi_plus_b) ** 2
            - np.abs(psi_plus_a) ** 2
        )
    )
    # ------------------------------------------------------------------
    # Minus channel:
    # (+k,-1) paired with (-k,+1)
    # ------------------------------------------------------------------
    minus_bracket_a = (
        pump_density * psi_minus_a
        + pump_squared * np.conj(psi_minus_b)
    )
    minus_bracket_b = (
        pump_density * psi_minus_b
        + pump_squared * np.conj(psi_minus_a)
    )
    condensate_minus = (
        -1j
        * chi_minus
        * (
            2.0 * np.conj(psi_0) * psi_minus_a * psi_minus_b
            + psi_0 * np.abs(psi_minus_a) ** 2
            + psi_0 * np.abs(psi_minus_b) ** 2
        )
        + gamma_minus
        * psi_0
        * (
            np.abs(psi_minus_b) ** 2
            - np.abs(psi_minus_a) ** 2
        )
    )
    # Allocate the derivative vector in double-precision complex format
    dpsi_dt = np.empty(6, dtype=np.complex128)
    # Central condensate mode
    dpsi_dt[0] = condensate_plus + condensate_minus
    # Plus pair channel
    dpsi_dt[1] = (
        -1j * omega_pair * psi_plus_a
        + (gamma_plus - 1j * chi_plus) * plus_bracket_a
    )
    dpsi_dt[2] = (
        -1j * omega_pair * psi_plus_b
        - (gamma_plus + 1j * chi_plus) * plus_bracket_b
    )
    # Minus pair channel
    dpsi_dt[3] = (
        -1j * omega_pair * psi_minus_a
        + (gamma_minus - 1j * chi_minus) * minus_bracket_a
    )
    dpsi_dt[4] = (
        -1j * omega_pair * psi_minus_b
        - (gamma_minus + 1j * chi_minus) * minus_bracket_b
    )
    # Higher-order condensate mode
    dpsi_dt[5] = (
        -4j * omega_rec * psi_high
        + sqrt_a * (condensate_plus + condensate_minus)
    )
    return dpsi_dt


def eoms_two_pump(
    t: float,
    psi: NDArray[np.complex128],
    *,
    chi_aa_plus: float,
    chi_bb_plus: float,
    chi_ab_plus: complex,
    chi_aa_minus: float,
    chi_bb_minus: float,
    chi_ab_minus: complex,
    gamma_aa_plus: float,
    gamma_bb_plus: float,
    gamma_ab_plus: complex,
    gamma_aa_minus: float,
    gamma_bb_minus: float,
    gamma_ab_minus: complex,
    omega_pair: float,
    mode_mixing: float = 2.0 / 3.0,
    omega_rec: float = 0.0,
) -> NDArray[np.complex128]:

    """
    Six-mode equations of motion for the two-pump model.

    Mode ordering
    -------------
    psi[0] = psi_(0,1,0)
    psi[1] = psi_(+k,2,+1)
    psi[2] = psi_(-k,2,-1)
    psi[3] = psi_(+k,2,-1)
    psi[4] = psi_(-k,2,+1)
    psi[5] = psi_(±2 k_x,1,0)

    Parameters
    ----------
    t
        Time variable required by scipy.integrate.solve_ivp.
    psi
        Complex vector containing the six mode amplitudes.
    chi_aa_plus, chi_bb_plus, chi_ab_plus
        Coherent coupling coefficients of the plus pair channel.
    chi_aa_minus, chi_bb_minus, chi_ab_minus
        Coherent coupling coefficients of the minus pair channel.
    gamma_aa_plus, gamma_bb_plus, gamma_ab_plus
        Dissipative coupling coefficients of the plus pair channel.
    gamma_aa_minus, gamma_bb_minus, gamma_ab_minus
        Dissipative coupling coefficients of the minus pair channel.
    omega_pair
        Free evolution frequency of each excited pair mode.
        This corresponds to omega_0 / 2.
    mode_mixing
        Coefficient a in the effective condensate amplitude
            psi_0 = psi_(0,1,0) + sqrt(a) psi_(±2 k_x,1,0).
        The six-mode model uses a = 2/3.
    omega_rec
        Recoil frequency of the higher-order condensate mode.

    Returns
    -------
    dpsi_dt
        Time derivative of the six complex mode amplitudes.
    """

    if psi.shape != (6,):
        raise ValueError(
            f"Expected a six-component state vector, received shape {psi.shape}."
        )

    # ------------------------------------------------------------------
    # Unpack the six dynamical mode amplitudes.
    # ------------------------------------------------------------------
    psi_00 = psi[0]
    psi_plus_a = psi[1]   # (+k,2,+1)
    psi_plus_b = psi[2]   # (-k,2,-1)
    psi_minus_a = psi[3]  # (+k,2,-1)
    psi_minus_b = psi[4]  # (-k,2,+1)
    psi_high = psi[5]     # (±2 k_x,1,0)

    # Effective condensate amplitude entering all interaction terms
    sqrt_a = np.sqrt(mode_mixing)
    psi_0 = psi_00 + sqrt_a * psi_high

    pump_density = np.abs(psi_0) ** 2
    pump_squared = psi_0 ** 2

    # ------------------------------------------------------------------
    # Plus channel:
    # (+k,2,+1) paired with (-k,2,-1)
    # ------------------------------------------------------------------
    dpsi_plus_a = (
        -1j * omega_pair * psi_plus_a
        + 1j
        * chi_bb_plus
        * pump_density
        * psi_plus_a
        - 1j
        * chi_ab_plus
        * pump_squared
        * np.conj(psi_plus_b)
        + gamma_bb_plus
        * pump_density
        * psi_plus_a
        - gamma_ab_plus
        * pump_squared
        * np.conj(psi_plus_b)
    )

    dpsi_plus_b = (
        -1j * omega_pair * psi_plus_b
        + 1j
        * chi_aa_plus
        * pump_density
        * psi_plus_b
        - 1j
        * chi_ab_plus
        * pump_squared
        * np.conj(psi_plus_a)
        - gamma_aa_plus
        * pump_density
        * psi_plus_b
        + gamma_ab_plus
        * pump_squared
        * np.conj(psi_plus_a)
    )

    condensate_plus = (
        -2j
        * np.conj(psi_0)
        * np.conj(chi_ab_plus)
        * psi_plus_a
        * psi_plus_b
        + 1j
        * psi_0
        * (
            chi_bb_plus * np.abs(psi_plus_a) ** 2
            + chi_aa_plus * np.abs(psi_plus_b) ** 2
        )
        + psi_0
        * (
            gamma_aa_plus * np.abs(psi_plus_b) ** 2
            - gamma_bb_plus * np.abs(psi_plus_a) ** 2
        )
    )

    # ------------------------------------------------------------------
    # Minus channel:
    # (+k,2,-1) paired with (-k,2,+1)
    # ------------------------------------------------------------------
    dpsi_minus_a = (
        -1j * omega_pair * psi_minus_a
        + 1j
        * chi_bb_minus
        * pump_density
        * psi_minus_a
        - 1j
        * chi_ab_minus
        * pump_squared
        * np.conj(psi_minus_b)
        + gamma_bb_minus
        * pump_density
        * psi_minus_a
        - gamma_ab_minus
        * pump_squared
        * np.conj(psi_minus_b)
    )

    dpsi_minus_b = (
        -1j * omega_pair * psi_minus_b
        + 1j
        * chi_aa_minus
        * pump_density
        * psi_minus_b
        - 1j
        * chi_ab_minus
        * pump_squared
        * np.conj(psi_minus_a)
        - gamma_aa_minus
        * pump_density
        * psi_minus_b
        + gamma_ab_minus
        * pump_squared
        * np.conj(psi_minus_a)
    )

    condensate_minus = (
        -2j
        * np.conj(psi_0)
        * np.conj(chi_ab_minus)
        * psi_minus_a
        * psi_minus_b
        + 1j
        * psi_0
        * (
            chi_bb_minus * np.abs(psi_minus_a) ** 2
            + chi_aa_minus * np.abs(psi_minus_b) ** 2
        )
        + psi_0
        * (
            gamma_aa_minus * np.abs(psi_minus_b) ** 2
            - gamma_bb_minus * np.abs(psi_minus_a) ** 2
        )
    )

    # Allocate the derivative vector in double-precision complex format
    dpsi_dt = np.empty(6, dtype=np.complex128)

    # Central condensate mode
    dpsi_dt[0] = condensate_plus + condensate_minus

    # Plus pair channel
    dpsi_dt[1] = dpsi_plus_a
    dpsi_dt[2] = dpsi_plus_b

    # Minus pair channel
    dpsi_dt[3] = dpsi_minus_a
    dpsi_dt[4] = dpsi_minus_b

    # Higher-order condensate mode
    dpsi_dt[5] = (
        -4j * omega_rec * psi_high
        + sqrt_a * (condensate_plus + condensate_minus)
    )

    return dpsi_dt