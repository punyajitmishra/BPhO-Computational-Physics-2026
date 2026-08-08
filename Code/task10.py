"""BPhO 2026 Task 10: 2D and 3D hydrogenic-orbital probability densities.

The code evaluates normalised radial functions and spherical harmonics without
external scientific libraries.  Change ORBITALS to select valid (n, l, m)
quantum numbers.  The 3D plots retain points above 15% of each density maximum,
matching the presentation's suggested visualisation threshold.
"""

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A0 = 5.29177210903e-11  # Bohr radius / m
ORBITALS = ((1, 0, 0, "1s"), (2, 1, 0, "2p$_z$"),
            (3, 2, 0, "3d$_{z^2}$"), (4, 3, 2, "4f, m=2"))


def associated_laguerre(order: int, alpha: int, x: np.ndarray) -> np.ndarray:
    """Generalised Laguerre polynomial L_order^alpha(x), by recurrence."""
    if order == 0:
        return np.ones_like(x)
    if order == 1:
        return 1 + alpha - x
    previous = np.ones_like(x)
    current = 1 + alpha - x
    for k in range(2, order + 1):
        next_value = ((2 * k - 1 + alpha - x) * current - (k - 1 + alpha) * previous) / k
        previous, current = current, next_value
    return current


def associated_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Associated Legendre P_l^m(x) for non-negative m."""
    m = abs(m)
    p_mm = (-1)**m * math.prod(range(1, 2 * m, 2)) * (1 - x**2)**(m / 2)
    if l == m:
        return p_mm
    p_m1m = x * (2 * m + 1) * p_mm
    if l == m + 1:
        return p_m1m
    previous, current = p_mm, p_m1m
    for degree in range(m + 2, l + 1):
        next_value = ((2 * degree - 1) * x * current - (degree + m - 1) * previous) / (degree - m)
        previous, current = current, next_value
    return current


def probability_density(x: np.ndarray, y: np.ndarray, z: np.ndarray, n: int, l: int, m: int) -> np.ndarray:
    """Return |psi_nlm|^2 for hydrogen (Z=1), in SI volume-density units."""
    if not (n >= 1 and 0 <= l < n and -l <= m <= l):
        raise ValueError("Require n >= 1, 0 <= l < n and -l <= m <= l.")
    r = np.sqrt(x**2 + y**2 + z**2)
    rho = 2 * r / (n * A0)
    radial_norm = np.sqrt((2 / (n * A0))**3 * math.factorial(n - l - 1) /
                          (2 * n * math.factorial(n + l)))
    radial = radial_norm * np.exp(-rho / 2) * rho**l * associated_laguerre(n - l - 1, 2 * l + 1, rho)
    cos_theta = np.divide(z, r, out=np.ones_like(r), where=r > 0)
    abs_m = abs(m)
    angular = ((2 * l + 1) / (4 * np.pi) * math.factorial(l - abs_m) /
               math.factorial(l + abs_m) * associated_legendre(l, abs_m, cos_theta)**2)
    return radial**2 * angular


def main() -> None:
    extent_a0 = 22
    coordinate = np.linspace(-extent_a0, extent_a0, 260) * A0
    # Use the y=0 x-z plane. Unlike the z=0 plane, it also displays orbitals
    # whose angular factor vanishes in the equatorial plane.
    x, z = np.meshgrid(coordinate, coordinate)
    y = np.zeros_like(x)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, (n, l, m, label) in zip(axes.flat, ORBITALS):
        density = probability_density(x, y, z, n, l, m)
        image = ax.imshow(density / density.max(), extent=(-extent_a0, extent_a0, -extent_a0, extent_a0),
                          origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set(title=f"{label}: $n={n}, l={l}, m={m}$", xlabel=r"x / $a_0$", ylabel=r"z / $a_0$", aspect="equal")
    fig.colorbar(image, ax=axes, label=r"$|\psi|^2 / |\psi|^2_{max}$", shrink=0.8)
    fig.savefig("task10_hydrogenic_orbitals_2d.png", dpi=180)

    # 3D coloured-glass analogue: dense points above 15% of the maximum.
    grid = np.linspace(-14, 14, 45) * A0
    x3, y3, z3 = np.meshgrid(grid, grid, grid, indexing="ij")
    fig3 = plt.figure(figsize=(11, 5))
    for index, (n, l, m, label) in enumerate(ORBITALS[1:3], start=1):
        ax = fig3.add_subplot(1, 2, index, projection="3d")
        density = probability_density(x3, y3, z3, n, l, m)
        mask = density > 0.15 * density.max()
        scaled = density[mask] / density.max()
        ax.scatter(x3[mask] / A0, y3[mask] / A0, z3[mask] / A0, c=scaled,
                   cmap="plasma", s=5, alpha=0.65, linewidths=0)
        ax.set(title=f"3D probability density: {label}", xlabel=r"x / $a_0$", ylabel=r"y / $a_0$", zlabel=r"z / $a_0$")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-55)
    fig3.tight_layout()
    fig3.savefig("task10_hydrogenic_orbitals_3d.png", dpi=180)
    print("Saved task10_hydrogenic_orbitals_2d.png and task10_hydrogenic_orbitals_3d.png")


if __name__ == "__main__":
    main()
