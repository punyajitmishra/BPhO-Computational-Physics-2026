"""BPhO 2026 Task 9: relativistic Compton scattering of 100 keV photons.

The presentation gives Delta lambda = h/(m_e c) (1-cos(theta)).  Momentum and
energy conservation give the recoil-electron momentum, speed and angle.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = 6.62607015e-34
C = 2.99792458e8
M_E = 9.1093837139e-31
E_CHARGE = 1.602176634e-19
PHOTON_ENERGY_KEV = 100.0


def compton_results(theta: np.ndarray, photon_energy_keV: float = PHOTON_ENERGY_KEV):
    """Return fractional shift, electron speed and recoil angle for theta."""
    initial_energy = photon_energy_keV * 1e3 * E_CHARGE
    wavelength = H * C / initial_energy
    compton_wavelength = H / (M_E * C)
    delta_wavelength = compton_wavelength * (1 - np.cos(theta))
    scattered_wavelength = wavelength + delta_wavelength
    scattered_energy = H * C / scattered_wavelength

    initial_momentum = initial_energy / C
    scattered_momentum = scattered_energy / C
    p_x = initial_momentum - scattered_momentum * np.cos(theta)
    p_y = scattered_momentum * np.sin(theta)
    recoil_angle = np.arctan2(p_y, p_x)

    kinetic_energy = initial_energy - scattered_energy
    gamma = 1 + kinetic_energy / (M_E * C**2)
    speed = C * np.sqrt(1 - gamma**-2)
    return delta_wavelength / wavelength, speed, recoil_angle


def main() -> None:
    theta = np.linspace(0, np.pi, 1000)
    fractional_shift, speed, recoil_angle = compton_results(theta)
    theta_deg = np.rad2deg(theta)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(theta_deg, fractional_shift, color="#2a6fbb")
    axes[0].set(ylabel=r"$\Delta\lambda/\lambda$", title="Fractional wavelength shift")
    axes[1].plot(theta_deg, speed / C, color="#d1495b")
    axes[1].set(ylabel=r"Electron recoil speed, $v/c$", title="Electron recoil speed")
    axes[2].plot(theta_deg, np.rad2deg(recoil_angle), color="#3a9d5d")
    axes[2].set(ylabel=r"Electron recoil angle, $\phi$ / degrees", title="Electron recoil angle")
    for ax in axes:
        ax.set(xlabel=r"Photon scattering angle, $\theta$ / degrees", xlim=(0, 180))
        ax.grid(alpha=0.25)
    fig.suptitle(f"Compton scattering of a {PHOTON_ENERGY_KEV:g} keV photon", fontsize=15)
    fig.tight_layout()
    fig.savefig("task9_compton_scattering.png", dpi=180)
    print("Saved task9_compton_scattering.png")


if __name__ == "__main__":
    main()
