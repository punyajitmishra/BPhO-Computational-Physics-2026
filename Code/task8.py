"""BPhO 2026 Task 8: classical and quantum photon-polarisation mismatch.

The formulae are those given on presentation slides 55--58.  Angles theta and
phi specify Alice's and Bob's detector bases.  Edit THETA_DEG and PHI_DEG to
use this file as a visual calculator.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THETA_DEG = -30.0
PHI_DEG = 30.0


def classical_mismatch(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return 1 - np.cos(theta)**2 * np.cos(phi)**2 - np.sin(theta)**2 * np.sin(phi)**2


def quantum_mismatch(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.sin(phi - theta)**2


def main() -> None:
    theta = np.deg2rad(THETA_DEG)
    phi = np.deg2rad(PHI_DEG)
    angle_deg = np.linspace(-90, 90, 500)
    angle = np.deg2rad(angle_deg)
    difference = PHI_DEG - THETA_DEG

    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    ax_curve.plot(angle_deg, classical_mismatch(theta, angle), label="Classical", lw=2)
    ax_curve.plot(angle_deg, quantum_mismatch(theta, angle), label="Quantum", lw=2)
    ax_curve.axvline(PHI_DEG, color="grey", ls="--", lw=1)
    ax_curve.plot(PHI_DEG, classical_mismatch(theta, phi), "o", color="#2a6fbb")
    ax_curve.plot(PHI_DEG, quantum_mismatch(theta, phi), "o", color="#d1495b")
    ax_curve.set(xlabel=r"Bob's angle, $\phi$ / degrees", ylabel="Mismatch probability",
                 title=rf"Alice fixed at $\theta={THETA_DEG:g}^\circ$")
    ax_curve.set_ylim(0, 1)
    ax_curve.grid(alpha=0.25)
    ax_curve.legend()

    values = [classical_mismatch(theta, phi), quantum_mismatch(theta, phi)]
    bars = ax_bar.bar(["Classical", "Quantum"], values, color=["#2a6fbb", "#d1495b"])
    for bar, value in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    ax_bar.set(ylim=(0, 1), ylabel="Mismatch probability",
               title=rf"Calculator: $\theta={THETA_DEG:g}^\circ$, $\phi={PHI_DEG:g}^\circ$ ($\phi-\theta={difference:g}^\circ$)")
    ax_bar.grid(axis="y", alpha=0.25)
    fig.tight_layout(w_pad=3.0)
    fig.savefig("task8_quantum_cryptography.png", dpi=180)
    print("Saved task8_quantum_cryptography.png")


if __name__ == "__main__":
    main()
