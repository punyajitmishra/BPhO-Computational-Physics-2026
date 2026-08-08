"""
BPhO Computational Physics Challenge 2026 - Quantum Mechanics
TASK #5: Hydrogen emission spectrum and Bohr's model

Create a graph of photon energy against wavelength for photon
emissions from hydrogen atoms due to transitions between electron
energy levels.

The Bohr model is derived here from two classical relations together
with quantisation of orbital angular momentum.

For an electron in a circular orbit of radius r around a proton,

    m_e v^2 / r = e^2 / (4 pi epsilon_0 r^2)

and Bohr's angular-momentum quantisation condition is

    m_e v r = n hbar,

where n = 1, 2, 3, ... and hbar = h/(2 pi).

Combining these gives the allowed orbit radii

    r_n = 4 pi epsilon_0 hbar^2 n^2 / (m_e e^2)

and the corresponding electron energies

    E_n = -m_e e^4 / (8 epsilon_0^2 h^2 n^2).

The ground-state energy is therefore calculated directly from the
physical constants rather than inserted as an unexplained 13.6 eV:

    E_1 = -m_e e^4 / (8 epsilon_0^2 h^2)
        = -13.6057 eV.

Thus,

    E_n = E_1 / n^2.

When an electron falls from an initial level n_i to a lower level n_f,
the emitted photon has energy

    E_gamma = E_ni - E_nf

and

    E_gamma = h c / lambda.

This produces the hydrogen emission spectrum.

NOTE:
The numerical constants used here are standard SI values. The
calculated ground-state energy is 13.6057 eV, while the measured
ionisation energy of neutral hydrogen is approximately 13.5984 eV.
The small difference reflects the idealisations of the basic Bohr
model, including its treatment of the electron and proton masses.

References:
NIST CODATA, Fundamental Physical Constants:
https://physics.nist.gov/cuu/Constants/

NIST Atomic Spectra Database / Handbook of Basic Atomic Spectroscopic
Data, Hydrogen:
https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable1.htm
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt


# Physical constants (SI)

H = 6.62607015e-34             # Planck constant, J s
C = 2.99792458e8               # speed of light, m/s
E_CHARGE = 1.602176634e-19     # elementary charge, C
M_E = 9.1093837139e-31         # electron mass, kg
EPSILON_0 = 8.8541878188e-12   # vacuum permittivity, F/m

HBAR = H / (2 * np.pi)


# -------------------------------------------------------------------
# Bohr model
# -------------------------------------------------------------------

def bohr_radius(n: int) -> float:
    """
    Return the Bohr-model orbital radius for level n in metres.

    Starting from

        m_e v^2 / r = e^2 / (4 pi epsilon_0 r^2)

    and

        m_e v r = n hbar,

    the allowed radius is

        r_n = 4 pi epsilon_0 hbar^2 n^2
              / (m_e e^2).
    """

    return (
        4 * np.pi * EPSILON_0 * HBAR**2 * n**2
        / (M_E * E_CHARGE**2)
    )


# Calculate the Bohr ground-state energy directly from the model.

# From the Coulomb force equation,

#     m_e v^2 / r = e^2 / (4 pi epsilon_0 r^2)

# we have

#     m_e v^2 = e^2 / (4 pi epsilon_0 r).

# Therefore the total electron energy is

#     E = K + U
#       = 1/2 m_e v^2 - e^2 / (4 pi epsilon_0 r)

# which becomes

#     E = -e^2 / (8 pi epsilon_0 r).

# Substituting the quantised radius gives

#     E_n = -m_e e^4 / (8 epsilon_0^2 h^2 n^2).

GROUND_STATE_ENERGY_J = (
    -M_E * E_CHARGE**4
    / (8 * EPSILON_0**2 * H**2)
)

# Convert joules to electron-volts.
GROUND_STATE_ENERGY_EV = GROUND_STATE_ENERGY_J / E_CHARGE


def bohr_energy(n: int) -> float:
    """
    Return the Bohr-model energy of hydrogen level n in eV.

        E_n = E_1 / n^2
    """

    return GROUND_STATE_ENERGY_EV / n**2


# -------------------------------------------------------------------
# Photon emission
# -------------------------------------------------------------------

def photon_energy(n_initial: int, n_final: int) -> float:
    """
    Return emitted photon energy for a downward transition in eV.

        E_gamma = E_initial - E_final
    """

    if n_initial <= n_final:
        raise ValueError("Emission requires n_initial > n_final.")

    return (
        bohr_energy(n_initial)
        - bohr_energy(n_final)
    )


def photon_wavelength(energy_eV: float) -> float:
    """
    Return photon wavelength in nm for a photon energy in eV.

        E_gamma = h c / lambda
    """

    energy_J = energy_eV * E_CHARGE
    wavelength_m = H * C / energy_J

    return wavelength_m * 1e9


def plot_bohr_energy_levels() -> None:
    """Plot Bohr energy levels and two representative emission transitions."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    # The first four levels are sufficiently separated to label clearly;
    # higher levels visibly converge towards the ionisation limit.
    levels = range(1, 5)

    for n in levels:
        energy = bohr_energy(n)
        ax.hlines(energy, 0.35, 1.65, color="#2a6fbb", lw=2)
        ax.text(1.72, energy, rf"$n={n}$  ({energy:.2f} eV)", va="center")

    ax.text(0.38, -0.45, r"higher $n$ levels converge to $E=0$", color="#555555")

    # Show a visible Balmer photon and an ultraviolet Lyman photon.
    transitions = ((3, 2, "#d1495b", "656.5 nm"), (2, 1, "#6a4c93", "121.6 nm"))
    for x, (n_i, n_f, colour, label) in zip((0.70, 1.30), transitions):
        ax.annotate("", xy=(x, bohr_energy(n_f) + 0.10),
                    xytext=(x, bohr_energy(n_i) - 0.10),
                    arrowprops=dict(arrowstyle="->", color=colour, lw=2.5))
        ax.text(x + 0.05, (bohr_energy(n_i) + bohr_energy(n_f)) / 2,
                f"${n_i}\\!\\to\\!{n_f}$\n{label}", color=colour, va="center")

    ax.axhline(0, color="black", lw=1, ls="--")
    ax.text(1.72, 0, "ionisation limit", va="center")
    ax.set(xlim=(0, 2.65), ylim=(-14.5, 1), xticks=[],
           ylabel="Electron energy / eV",
           title="Bohr model of hydrogen: quantised energy levels")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("task5_bohr_model.png", dpi=180)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":

    # --- Display the calculated fundamental Bohr quantities ---

    print(
        f"Calculated Bohr ground-state energy: "
        f"{GROUND_STATE_ENERGY_EV:.7f} eV"
    )

    print(
        f"Bohr radius: "
        f"{bohr_radius(1) * 1e10:.6f} Angstrom"
    )


    # --- Calculate hydrogen emission transitions ---

    # Include all downward transitions between n = 2 and n = 6.
    #
    # This produces the first several lines of the Lyman, Balmer,
    # Paschen, Brackett and higher spectral series.

    transitions = []

    for n_initial in range(2, 7):

        for n_final in range(1, n_initial):

            energy = photon_energy(
                n_initial,
                n_final
            )

            wavelength = photon_wavelength(
                energy
            )

            transitions.append(
                (
                    n_initial,
                    n_final,
                    energy,
                    wavelength
                )
            )


    # Sort by wavelength for a clearer spectral plot.

    transitions.sort(
        key=lambda item: item[3]
    )

    wavelengths = np.array(
        [item[3] for item in transitions]
    )

    energies = np.array(
        [item[2] for item in transitions]
    )


    # --- Plot: Hydrogen emission spectrum ---

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        wavelengths,
        energies,
        s=45
    )


    # Label only the Balmer transitions to keep the near-UV region readable.

    for (
        n_initial,
        n_final,
        energy,
        wavelength
    ) in transitions:

        if n_final == 2 and n_initial in (3, 4, 5):
            ax.annotate(
                rf"${n_initial}\rightarrow{n_final}$",
                (wavelength, energy),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8
            )


    ax.set_xlabel(
        "Photon wavelength / nm"
    )

    # The emission lines span ultraviolet through infrared wavelengths.
    # A logarithmic wavelength axis makes all transition series visible.
    ax.set_xscale("log")

    ax.set_ylabel(
        "Photon energy / eV"
    )

    ax.set_title(
        "Hydrogen emission spectrum from the Bohr model"
    )

    fig.tight_layout()

    fig.savefig(
        "task5_hydrogen_emission_spectrum.png",
        dpi=150
    )
    plot_bohr_energy_levels()


    print(
        "Saved task5_hydrogen_emission_spectrum.png and task5_bohr_model.png"
    )


    # --- Print transition data ---

    print("\nHydrogen emission transitions:")
    print(
        "Transition    Wavelength / nm    "
        "Photon energy / eV"
    )

    for (
        n_initial,
        n_final,
        energy,
        wavelength
    ) in transitions:

        print(
            f"{n_initial} -> {n_final}"
            f"        {wavelength:10.3f}"
            f"          {energy:8.4f}"
        )
