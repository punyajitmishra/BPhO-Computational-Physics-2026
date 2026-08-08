"""Task 7 extension: verify Heisenberg uncertainty for a particle in a box.

For psi_n(x)=sqrt(2/a)sin(n*pi*x/a), Q2 of the supplied QM3 sheet gives
Delta x Delta p = (n*pi/2) sqrt(1/3 - 1/(2 n^2)).  This program evaluates it
alongside hbar/2, both analytically and from numerical integrals.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HBAR = 1.054571817e-34


def analytic_uncertainty_product(n: np.ndarray) -> np.ndarray:
    """Return Delta-x times Delta-p in J s for the nth stationary state."""
    return (HBAR / 2) * np.sqrt((n**2 * np.pi**2 / 3) - 2)


def numerical_uncertainty_product(n: int, width_m: float) -> float:
    """Integrate |psi|^2 to calculate Delta-x and use <p^2> = (n*pi*hbar/a)^2."""
    x = np.linspace(0, width_m, 100_001)
    density = 2 / width_m * np.sin(n * np.pi * x / width_m)**2
    mean_x = np.trapezoid(x * density, x)
    mean_x2 = np.trapezoid(x**2 * density, x)
    delta_x = np.sqrt(mean_x2 - mean_x**2)
    delta_p = n * np.pi * HBAR / width_m  # <p>=0 in an energy eigenstate
    return delta_x * delta_p


def main() -> None:
    n = np.arange(1, 13)
    analytical = analytic_uncertainty_product(n)
    numerical = np.array([numerical_uncertainty_product(int(level), 1e-9) for level in n])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n, analytical / HBAR, "o-", label="Analytic result", color="#2a6fbb")
    ax.plot(n, numerical / HBAR, "x", ms=8, mew=2, label="Numerical integral", color="#d1495b")
    ax.axhline(0.5, color="black", ls="--", label=r"Heisenberg limit, $\hbar/2$")
    ax.set(xlabel="Quantum number, n", ylabel=r"$\Delta x\,\Delta p / \hbar$",
           title="Particle in a box obeys the uncertainty principle")
    ax.set_xticks(n)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("task7_uncertainty_principle.png", dpi=180)
    print(f"n=1: Delta-x Delta-p = {analytical[0] / HBAR:.6f} hbar")
    print("Saved task7_uncertainty_principle.png")


if __name__ == "__main__":
    main()
