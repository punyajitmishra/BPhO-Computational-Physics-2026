"""BPhO 2026 Task 7: normalised particle-in-a-box eigenstates.

For a box of width a, psi_n(x) = sqrt(2/a) sin(n pi x/a) and
E_n = n^2 pi^2 hbar^2/(2 m a^2).  The plot uses an electron in a 1 nm box.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HBAR = 1.054571817e-34
M_E = 9.1093837139e-31
E_CHARGE = 1.602176634e-19


def energy(n: int, mass: float, width_m: float) -> float:
    """Particle-in-a-box energy in joules."""
    return n**2 * np.pi**2 * HBAR**2 / (2 * mass * width_m**2)


def probability_density(x_m: np.ndarray, n: int, width_m: float) -> np.ndarray:
    """Return |psi_n(x)|^2, zero outside the box."""
    density = np.zeros_like(x_m)
    inside = (x_m > 0) & (x_m < width_m)
    density[inside] = (2 / width_m) * np.sin(n * np.pi * x_m[inside] / width_m)**2
    return density


def main() -> None:
    width_m = 1e-9
    x_m = np.linspace(0, width_m, 1200)
    ns = np.arange(1, 7)

    fig, (ax_energy, ax_density) = plt.subplots(1, 2, figsize=(12, 5))
    energies_eV = np.array([energy(n, M_E, width_m) / E_CHARGE for n in ns])
    ax_energy.scatter(ns, energies_eV, s=60, color="#2a6fbb", zorder=3)
    ax_energy.vlines(ns, 0, energies_eV, color="#2a6fbb", alpha=0.35)
    ax_energy.set(xlabel="Quantum number, n", ylabel="Energy / eV",
                  title=r"Particle in a 1 nm box: $E_n \propto n^2$")
    ax_energy.set_xticks(ns)
    ax_energy.grid(alpha=0.25)

    for n in range(1, 5):
        ax_density.plot(x_m * 1e9, probability_density(x_m, n, width_m) * 1e-9,
                        label=rf"$n={n}$")
    ax_density.set(xlabel="Position in box, x / nm", ylabel=r"Probability density, $|\psi_n|^2$ / nm$^{-1}$",
                   title="Normalised stationary-state probability densities")
    ax_density.legend()
    ax_density.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("task7_particle_in_a_box.png", dpi=180)
    print("Saved task7_particle_in_a_box.png")


if __name__ == "__main__":
    main()
