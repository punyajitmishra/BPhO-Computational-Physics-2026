"""
BPhO Computational Physics Challenge 2026 - Quantum Mechanics
TASK #4: Photoelectric effect

Plot photoelectron stopping voltage against the frequency of incident
photons for several different metals.

Einstein's photoelectric equation is

    hf = phi + K_max

and the maximum kinetic energy of the emitted electrons is related to
the stopping voltage by

    K_max = e V_s.

Therefore,

    V_s = (h/e) f - phi/e.

For work functions expressed in electron-volts, phi/e has the numerical
value of phi_eV in volts, so the equation becomes

    V_s = (h/e) f - phi_eV.

Below the threshold frequency

    f_0 = phi/h,

the photon energy is insufficient to eject an electron. In this region
there is no photoelectric emission, so no stopping voltage is plotted.

NOTE:
The work-function values used here are standard reference values:
    Cesium (Cs):    2.10 eV
    Sodium (Na):    2.28 eV
    Zinc (Zn):      4.30 eV
    Copper (Cu):    4.70 eV
    Platinum (Pt):  6.35 eV

Reference:
HyperPhysics, "Work Functions for Photoelectric Effect",
Georgia State University, citing Tipler & Llewellyn and the
Handbook of Chemistry and Physics:
https://hyperphysics.phy-astr.gsu.edu/hbase/Tables/photoelec.html

The plotted stopping voltages are calculated directly from Einstein's
equation; the material-dependent quantity is the work function.
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
E_CHARGE = 1.602176634e-19  # elementary charge, C

# Work functions in electron-volts.
# Reference: HyperPhysics / Handbook of Chemistry and Physics.
WORK_FUNCTIONS = {
    "Cesium (Cs)": 2.10,
    "Sodium (Na)": 2.28,
    "Zinc (Zn)": 4.30,
    "Copper (Cu)": 4.70,
    "Platinum (Pt)": 6.35,
}


def threshold_frequency(work_function_eV: float) -> float:
    """Return the photoelectric threshold frequency in Hz."""
    work_function_J = work_function_eV * E_CHARGE
    return work_function_J / H


def stopping_voltage(frequency: np.ndarray,
                     work_function_eV: float) -> np.ndarray:
    """Return photoelectron stopping voltage in volts.

    Frequencies below the threshold are assigned NaN because no
    photoelectric emission occurs there.
    """
    threshold = threshold_frequency(work_function_eV)

    voltage = (H * frequency / E_CHARGE) - work_function_eV
    voltage = np.where(frequency >= threshold, voltage, np.nan)

    return voltage


if __name__ == "__main__":

    # --- Plot: Photoelectric stopping voltage ---

    # Frequency range chosen to cover the thresholds of all five metals.
    frequencies = np.linspace(4.0e14, 2.0e15, 2000)

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, work_function in WORK_FUNCTIONS.items():
        V_stop = stopping_voltage(frequencies, work_function)

        ax.plot(
            frequencies,
            V_stop,
            label=f"{label} ($\\phi$ = {work_function:.2f} eV)"
        )

    ax.axhline(
        0,
        color="grey",
        ls="--",
        lw=1
    )

    ax.set_xlabel("Incident photon frequency / Hz")
    ax.set_ylabel("Stopping voltage / V")
    ax.set_title("Photoelectric effect: stopping voltage vs frequency")
    ax.legend()

    fig.tight_layout()
    fig.savefig("task4_photoelectric_effect.png", dpi=150)

    print("Saved task4_photoelectric_effect.png")

    print("\nThreshold frequencies:")
    for label, work_function in WORK_FUNCTIONS.items():
        f0 = threshold_frequency(work_function)
        print(f"{label}: {f0:.4e} Hz")
