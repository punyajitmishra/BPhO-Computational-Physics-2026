"""
BPhO Computational Physics Challenge 2026 - Quantum Mechanics
TASK #3: Planck black-body spectrum + Einstein heat capacity model

Two independent plots:

1. Spectral radiance B(lambda, T) for several temperatures.
2. Einstein molar heat capacity C_V(T) for a few solids (gold, copper, iron).

NOTE:
The Einstein model uses a single characteristic temperature theta_E
for each solid. Rather than treating theta_E as an independently
tabulated material constant, an effective Einstein temperature is
derived here from the standard Debye temperature theta_D.

At high temperature, the Einstein and Debye heat capacities have the
expansions

    C_E = 3R [1 - theta_E^2/(12 T^2) + ...]
    C_D = 3R [1 - theta_D^2/(20 T^2) + ...]

Matching the first quantum correction gives

    theta_E = sqrt(3/5) * theta_D.

Standard Debye temperatures used:
    Gold (Au):   theta_D = 170 K
    Copper (Cu): theta_D = 343 K
    Iron (Fe):   theta_D = 470 K

These values are standard tabulated values in solid-state physics
references, e.g. C. Kittel, Introduction to Solid State Physics,
8th ed., Wiley (2004).

The resulting theta_E values are therefore effective single-Einstein
temperatures.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

# physical constants (SI)

H = 6.62607015e-34    # Planck constant, J s
C = 2.99792458e8      # speed of light, m/s
KB = 1.380649e-23     # Boltzmann constant, J/K
R_GAS = 8.314462618   # molar gas constant, J/(mol K)

# Standard Debye temperatures (K).
# Reference: C. Kittel, Introduction to Solid State Physics, 8th ed.
DEBYE_TEMPS = {
    "Gold (Au)": 170.0,
    "Copper (Cu)": 343.0,
    "Iron (Fe)": 470.0,
}

# Convert Debye temperatures to effective Einstein temperatures by
# matching the first high-temperature quantum correction:
#
#     theta_E = sqrt(3/5) * theta_D
#
# These are effective values for the single-Einstein model.
EINSTEIN_TEMPS = {
    label: np.sqrt(3.0 / 5.0) * theta_D
    for label, theta_D in DEBYE_TEMPS.items()
}


def planck_spectrum(wavelengths: np.ndarray, T: float) -> np.ndarray:
    """Spectral radiance B(lambda, T) in W / (m^2 . sr . m), lambda in metres."""
    x = (H * C) / (wavelengths * KB * T)

    # clip x to avoid overflow in exp() for very short wavelengths;
    x = np.minimum(x, 700)

    return (2 * H * C**2) / (wavelengths**5 * np.expm1(x))


def einstein_heat_capacity(T: np.ndarray, theta_E: float) -> np.ndarray:
    """Molar heat capacity C_V(T) [J/(mol K)] from the Einstein solid model."""
    x = theta_E / T

    # Written using exp(-x) for numerical stability at low temperature.
    exp_minus_x = np.exp(-np.minimum(x, 700))

    return (
        3 * R_GAS
        * x**2
        * exp_minus_x
        / (1 - exp_minus_x)**2
    )


if __name__ == "__main__":

    # --- Plot 1: Planck spectrum ---

    wavelengths_nm = np.linspace(1, 3000, 2000)  # nm, avoid lambda=0
    wavelengths_m = wavelengths_nm * 1e-9

    fig1, ax1 = plt.subplots(figsize=(7, 5))

    for T in (4000, 5000, 6000):
        B = planck_spectrum(wavelengths_m, T)

        # Convert W/m^3 -> W/(m^2 sr nm)
        B_per_nm = B * 1e-9

        ax1.plot(
            wavelengths_nm,
            B_per_nm,
            label=f"T = {T} K"
        )

    ax1.set_xlabel("Wavelength / nm")
    ax1.set_ylabel(r"Spectral radiance / W m$^{-2}$ sr$^{-1}$ nm$^{-1}$")
    ax1.set_title("Planck black-body radiation spectrum")
    ax1.legend()

    fig1.tight_layout()
    fig1.savefig("task3_planck_spectrum.png", dpi=150)


    # --- Plot 2: Einstein heat capacity ---

    T_range = np.linspace(5, 800, 2000)  # K, avoid T=0 (singular)

    fig2, ax2 = plt.subplots(figsize=(7, 5))

    for label, theta_E in EINSTEIN_TEMPS.items():
        Cv = einstein_heat_capacity(T_range, theta_E)

        ax2.plot(
            T_range,
            Cv,
            label=rf"{label} ($\theta_E$ = {theta_E:.1f} K)"
        )

    # Dulong-Petit high-temperature limit
    ax2.axhline(
        3 * R_GAS,
        color="grey",
        ls="--",
        lw=1,
        label="Dulong-Petit limit (3R)"
    )

    ax2.set_xlabel("Temperature / K", labelpad=8)
    ax2.set_ylabel(r"Molar heat capacity $C_V$ / J mol$^{-1}$ K$^{-1}$")
    ax2.set_title("Einstein model: molar heat capacity vs temperature")
    ax2.legend()

    fig2.tight_layout()
    fig2.savefig("task3_einstein_heat_capacity.png", dpi=150)

    print("Saved task3_planck_spectrum.png and task3_einstein_heat_capacity.png")
