"""
BPhO Computational Physics Challenge 2026 - Quantum Mechanics
TASK #1: Random Walk

Model: N steps of fixed size s, each in a random direction theta ~ U(0, 2*pi).
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt


def random_walk(N: int, s: float, n_walkers: int = 1, seed: int | None = None) -> np.ndarray:
    """
    Simulate `n_walkers` independent 2D random walks of N steps each,
    step length s, step direction theta ~ Uniform(0, 2*pi).

    Returns
    -------
    positions : ndarray, shape (n_walkers, N+1, 2)
        positions[:, 0, :] is the origin; positions[:, k, :] is the
        position after k steps.
    """
    rng = np.random.default_rng(seed)

    theta = rng.uniform(0.0, 2.0 * np.pi, size=(n_walkers, N))
    dx = s * np.cos(theta)
    dy = s * np.sin(theta)

    steps = np.stack([dx, dy], axis=-1)                 # (n_walkers, N, 2)
    positions = np.cumsum(steps, axis=1)                 # (n_walkers, N, 2)
    origin = np.zeros((n_walkers, 1, 2))
    positions = np.concatenate([origin, positions], axis=1)  # (n_walkers, N+1, 2)

    return positions


def net_displacement(positions: np.ndarray) -> np.ndarray:
    """End-to-end displacement |r_N - r_0| for each walker. Shape (n_walkers,)."""
    return np.linalg.norm(positions[:, -1, :] - positions[:, 0, :], axis=-1)


def plot_walks(positions: np.ndarray, s: float, ax=None, show_start=True):
    """Plot every walker's trajectory on one set of axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    n_walkers = positions.shape[0]
    for i in range(n_walkers):
        ax.plot(positions[i, :, 0], positions[i, :, 1], lw=0.7, alpha=0.85)

    if show_start:
        ax.plot(0, 0, "ko", ms=5, label="start")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Random walk. Step size = {s}")
    return ax


if __name__ == "__main__":
    N = 1000        # steps per walk
    s = 1.0         # step size
    seed = 42

    # A modest overlay keeps individual paths legible.
    paths = random_walk(N, s, n_walkers=50, seed=seed)
    # A larger ensemble tests the random-walk prediction <r^2> = N s^2.
    ensemble = random_walk(N, s, n_walkers=3000, seed=seed + 1)
    mean_squared_displacement = np.mean(np.sum(ensemble**2, axis=-1), axis=0)
    step_number = np.arange(N + 1)

    fig, (ax_paths, ax_validation) = plt.subplots(1, 2, figsize=(12, 5.5))
    plot_walks(paths, s, ax=ax_paths)
    ax_paths.set_title("Fifty independent random walks")
    ax_validation.plot(step_number, mean_squared_displacement,
                       label=r"simulation: $\langle r^2\rangle$", lw=2)
    ax_validation.plot(step_number, step_number * s**2, "--", color="black",
                       label=r"theory: $Ns^2$")
    ax_validation.set(xlabel="Number of steps, N",
                      ylabel=r"Mean squared displacement, $\langle r^2\rangle$",
                      title="Statistical check of the random-walk model")
    ax_validation.grid(alpha=0.25)
    ax_validation.legend()
    fig.tight_layout()
    fig.savefig("task1.png", dpi=180)

    # sanity check
    r = net_displacement(ensemble)
    rms_sim = np.sqrt(np.mean(r ** 2))
    rms_theory = s * np.sqrt(N)
    print(f"RMS displacement (simulated) : {rms_sim:.3f}")
    print(f"RMS displacement (theory s*sqrt(N)) : {rms_theory:.3f}")
