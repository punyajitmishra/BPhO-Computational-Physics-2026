"""
BPhO Computational Physics Challenge 2026 - Quantum Mechanics
TASK #6: Electron diffraction from polycrystalline graphite

This model follows slides 40--42 of the supplied Challenge Presentation.

An electron accelerated through a potential difference V has kinetic energy

    e V = p^2 / (2 m_e),

so its de Broglie wavelength is

    lambda = h / sqrt(2 m_e e V).

Constructive diffraction from graphite planes obeys Bragg's law

    n lambda = 2 d sin(phi),

where n is the diffraction order.  A ring appears on the spherical phosphor
screen at

    x = r sin(2 phi),     r = 65 mm.

Combining the first two equations gives the presentation's Task 6a check:

    1 / sqrt(V) = [2 sqrt(2 m_e e) d / (n h)] sin(phi).

Thus a plot of 1/sqrt(V) against sin(phi) is a straight line; the gradient
can recover the graphite layer spacing d.  Both stated graphite spacings,
0.123 nm and 0.213 nm, are included.
"""

from dataclasses import dataclass
import os
from pathlib import Path

# Make the script reliably runnable on machines without a desktop Python GUI.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# SI constants (exact SI definitions where applicable)
H = 6.62607015e-34          # Planck constant / J s
E_CHARGE = 1.602176634e-19  # elementary charge / C
M_E = 9.1093837139e-31      # electron mass / kg
SCREEN_RADIUS_M = 65e-3


@dataclass(frozen=True)
class GraphitePlane:
    """A graphite plane family used to form one series of diffraction rings."""

    name: str
    spacing_m: float
    colour: str


PLANES = (
    GraphitePlane("d₁ = 0.123 nm", 0.123e-9, "#2a6fbb"),
    GraphitePlane("d₂ = 0.213 nm", 0.213e-9, "#d1495b"),
)


def electron_wavelength(voltage_V: float) -> float:
    """Return the non-relativistic de Broglie wavelength in metres."""
    if voltage_V <= 0:
        raise ValueError("Accelerating voltage must be positive.")
    return H / np.sqrt(2 * M_E * E_CHARGE * voltage_V)


def diffraction_rings(voltage_V: float, plane: GraphitePlane):
    """Return all physically allowed (order, phi, radius) rings for one plane.

    Bragg's law requires n*lambda/(2*d) <= 1.  The order is therefore not
    arbitrarily truncated: every allowed integer order is returned.
    """
    wavelength = electron_wavelength(voltage_V)
    maximum_order = int(np.floor(2 * plane.spacing_m / wavelength))
    rings = []

    for order in range(1, maximum_order + 1):
        sin_phi = order * wavelength / (2 * plane.spacing_m)
        phi = np.arcsin(sin_phi)
        radius_m = SCREEN_RADIUS_M * np.sin(2 * phi)
        rings.append((order, phi, radius_m))

    return rings


def recovered_spacing(gradient: float, order: int = 1) -> float:
    """Recover d from the Task 6a gradient in SI units."""
    return gradient * order * H / (2 * np.sqrt(2 * M_E * E_CHARGE))


def draw_screen(ax, voltage_kV: float) -> None:
    """Draw a face-on spherical screen and every allowed ring at one voltage."""
    screen = plt.Circle((0, 0), SCREEN_RADIUS_M * 1e3,
                        facecolor="#e9fff0", edgecolor="#255c3b", lw=2)
    ax.add_patch(screen)
    ax.plot(0, 0, "+", color="#333333", ms=8, mew=1.5)

    for plane in PLANES:
        for order, _, radius_m in diffraction_rings(voltage_kV * 1e3, plane):
            ring = plt.Circle((0, 0), radius_m * 1e3, fill=False,
                              color=plane.colour, lw=1.15, alpha=0.9)
            ax.add_patch(ring)

    radius_mm = SCREEN_RADIUS_M * 1e3
    ax.set(xlim=(-radius_mm, radius_mm), ylim=(-radius_mm, radius_mm),
           aspect="equal", title=f"Model phosphor screen: {voltage_kV:g} kV")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    """Create screen-ring models and the required straight-line validation."""
    fig, axes = plt.subplot_mosaic(
        [["one", "three"], ["five", "validation"]], figsize=(12, 10)
    )

    for axis_name, voltage_kV in (("one", 1), ("three", 3), ("five", 5)):
        draw_screen(axes[axis_name], voltage_kV)

    # Legend placed beneath the screen series, without obscuring rings.
    handles = [plt.Line2D([], [], color=p.colour, lw=2, label=p.name)
               for p in PLANES]
    axes["five"].legend(handles=handles, loc="upper center",
                         bbox_to_anchor=(0.5, -0.05), frameon=False)

    # Task 6a: use first-order rings at a range of voltages.  The result is
    # linear for each spacing; least-squares gradients independently recover d.
    validation_ax = axes["validation"]
    voltages = np.linspace(1e3, 5e3, 15)
    for plane in PLANES:
        phi = np.array([diffraction_rings(V, plane)[0][1] for V in voltages])
        x = np.sin(phi)
        y = 1 / np.sqrt(voltages)
        gradient, intercept = np.polyfit(x, y, 1)
        fitted_spacing_nm = recovered_spacing(gradient) * 1e9

        validation_ax.scatter(x, y, color=plane.colour, s=26, zorder=3)
        validation_ax.plot(
            x, gradient * x + intercept, color=plane.colour, lw=1.5,
            label=(f"{plane.name}; fit d = {fitted_spacing_nm:.3f} nm")
        )

    validation_ax.set(
        xlabel=r"$\sin\phi$",
        ylabel=r"$1 / \sqrt{V}$  / V$^{-1/2}$",
        title=r"Task 6a check: $1/\sqrt{V}$ vs $sin\phi$",
    )
    validation_ax.grid(alpha=0.25)
    validation_ax.legend(fontsize=9)

    fig.suptitle("Electron diffraction by graphite: 1--5 kV", fontsize=16, y=0.98)
    fig.tight_layout()
    output = Path(__file__).with_name("task6_electron_diffraction.png")
    fig.savefig(output, dpi=180)
    print(f"Saved {output.name}")

    for voltage_kV in (1, 3, 5):
        counts = [len(diffraction_rings(voltage_kV * 1e3, plane)) for plane in PLANES]
        print(f"{voltage_kV} kV: {counts[0]} rings for d1; {counts[1]} rings for d2")


if __name__ == "__main__":
    main()
