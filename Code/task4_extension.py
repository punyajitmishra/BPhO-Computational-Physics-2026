"""
BPhO Computational Physics Challenge 2026 - Quantum Mechanics
TASK #4 EXTENSION: Interactive photoelectric-effect simulation

Extension specified by the challenge:
    "Create an animated app like the PhET demo."

This app provides an interactive visual model of the photoelectric effect.
The user can vary:

    - the metal/work function,
    - incident photon frequency,
    - light intensity,
    - stopping voltage.

The physics is based on Einstein's photoelectric equation:

    hf = phi + K_max

with

    K_max = e V_s

so that

    V_s = hf/e - phi/e.

For a work function expressed in eV,

    V_s = (h/e) f - phi_eV.

The threshold frequency is

    f_0 = phi/h.

Photons with f < f_0 cannot eject electrons. Above threshold,
electrons are emitted with maximum kinetic energy

    K_max = hf - phi.

The animation is deliberately a model rather than a microscopic
simulation: emitted electrons are represented by moving particles,
while their initial kinetic energies are sampled between zero and
the maximum allowed value. The number of emitted electrons is
controlled by the intensity parameter.

Reference for the work functions:
HyperPhysics, "Work Functions for Photoelectric Effect",
Georgia State University:
https://hyperphysics.phy-astr.gsu.edu/hbase/Tables/photoelec.html
"""

import tkinter as tk
from tkinter import ttk
import random
import math

# -------------------------------------------------------------------
# Physical constants (SI)
# -------------------------------------------------------------------

H = 6.62607015e-34          # Planck constant, J s
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
    return work_function_eV * E_CHARGE / H


def photon_energy_eV(frequency: float) -> float:
    """Return photon energy hf in electron-volts."""
    return H * frequency / E_CHARGE


def maximum_kinetic_energy_eV(
    frequency: float,
    work_function_eV: float
) -> float:
    """Return maximum photoelectron kinetic energy in eV."""
    return max(
        0.0,
        photon_energy_eV(frequency) - work_function_eV
    )


def stopping_voltage(
    frequency: float,
    work_function_eV: float
) -> float:
    """Return stopping voltage in volts."""
    return maximum_kinetic_energy_eV(
        frequency,
        work_function_eV
    )


class PhotoelectricApp:
    """Animated Tkinter demonstration of the photoelectric effect."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BPhO 2026 — Photoelectric Effect")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        self.running = True
        self.electrons = []
        self.photons = []

        # User-controllable parameters.
        self.metal = tk.StringVar(value="Cesium (Cs)")
        self.frequency = tk.DoubleVar(value=8.0e14)
        self.intensity = tk.DoubleVar(value=50.0)
        self.stopping_voltage_value = tk.DoubleVar(value=0.0)

        self._build_ui()
        self._update_labels()
        self._animate()

    # ---------------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------------

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        title = ttk.Label(
            main,
            text="Photoelectric Effect",
            font=("Segoe UI", 20, "bold")
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            main,
            text="Einstein's photoelectric equation — interactive model",
            font=("Segoe UI", 10)
        )
        subtitle.pack(anchor="w", pady=(0, 10))

        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=(0, 10))

        # Metal selector.
        ttk.Label(
            controls,
            text="Metal:"
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))

        metal_box = ttk.Combobox(
            controls,
            textvariable=self.metal,
            values=list(WORK_FUNCTIONS),
            state="readonly",
            width=18
        )
        metal_box.grid(row=0, column=1, sticky="w")
        metal_box.bind(
            "<<ComboboxSelected>>",
            lambda event: self._update_labels()
        )

        # Frequency slider.
        ttk.Label(
            controls,
            text="Frequency:"
        ).grid(row=0, column=2, sticky="w", padx=(25, 8))

        self.frequency_scale = tk.Scale(
            controls,
            variable=self.frequency,
            from_=3.0e14,
            to=2.0e15,
            resolution=1.0e13,
            orient="horizontal",
            length=360,
            showvalue=False,
            command=lambda value: self._update_labels()
        )
        self.frequency_scale.grid(row=0, column=3, sticky="ew")

        # Intensity slider.
        ttk.Label(
            controls,
            text="Intensity:"
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self.intensity_scale = tk.Scale(
            controls,
            variable=self.intensity,
            from_=0,
            to=100,
            resolution=1,
            orient="horizontal",
            length=250,
            showvalue=False
        )
        self.intensity_scale.grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="w",
            pady=(8, 0)
        )

        # Stopping-voltage slider.
        ttk.Label(
            controls,
            text="Stopping voltage:"
        ).grid(row=1, column=3, sticky="w", pady=(8, 0))

        self.stop_scale = tk.Scale(
            controls,
            variable=self.stopping_voltage_value,
            from_=0,
            to=10,
            resolution=0.1,
            orient="horizontal",
            length=250,
            showvalue=False,
            command=lambda value: self._update_labels()
        )
        self.stop_scale.grid(
            row=1,
            column=4,
            sticky="w",
            pady=(8, 0)
        )

        controls.columnconfigure(3, weight=1)

        # Main simulation canvas.
        self.canvas = tk.Canvas(
            main,
            background="#081018",
            highlightthickness=0
        )
        self.canvas.pack(
            fill="both",
            expand=True
        )

        self.canvas.bind(
            "<Configure>",
            lambda event: self._draw_static_scene()
        )

        # Information panel.
        self.info = ttk.Label(
            main,
            text="",
            font=("Consolas", 10)
        )
        self.info.pack(
            anchor="w",
            pady=(10, 0)
        )

        self.status = ttk.Label(
            main,
            text="",
            font=("Segoe UI", 11, "bold")
        )
        self.status.pack(
            anchor="w",
            pady=(4, 0)
        )

    # ---------------------------------------------------------------
    # Physics / state
    # ---------------------------------------------------------------

    def _current_parameters(self):
        metal = self.metal.get()
        phi = WORK_FUNCTIONS[metal]
        frequency = self.frequency.get()

        return (
            metal,
            phi,
            frequency,
            threshold_frequency(phi),
            photon_energy_eV(frequency),
            maximum_kinetic_energy_eV(frequency, phi),
        )

    def _update_labels(self):
        (
            metal,
            phi,
            frequency,
            threshold,
            photon_energy,
            kinetic_energy
        ) = self._current_parameters()

        self.info.config(
            text=(
                f"{metal}    "
                f"φ = {phi:.2f} eV    "
                f"f = {frequency / 1e14:.2f} × 10¹⁴ Hz    "
                f"f₀ = {threshold / 1e14:.2f} × 10¹⁴ Hz    "
                f"hf = {photon_energy:.2f} eV    "
                f"Kₘₐₓ = {kinetic_energy:.2f} eV"
            )
        )

        if frequency < threshold:
            self.status.config(
                text="NO PHOTOELECTRONS — photon energy is below the work function"
            )
        else:
            applied = self.stopping_voltage_value.get()

            if applied >= kinetic_energy:
                message = (
                    "STOPPED — applied stopping voltage is "
                    "at least the required stopping potential"
                )
            else:
                message = (
                    f"PHOTOELECTRONS EMITTED — stopping voltage required: "
                    f"{kinetic_energy:.2f} V"
                )

            self.status.config(text=message)

    # ---------------------------------------------------------------
    # Drawing
    # ---------------------------------------------------------------

    def _draw_static_scene(self):
        self.canvas.delete("static")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width <= 1 or height <= 1:
            return

        surface_x = int(width * 0.58)
        surface_top = int(height * 0.16)
        surface_bottom = int(height * 0.88)

        # Light source / photon side.
        self.canvas.create_text(
            90,
            surface_top - 30,
            text="PHOTONS",
            fill="#d9edf7",
            font=("Segoe UI", 10, "bold"),
            tags="static"
        )

        # Metal plate.
        self.canvas.create_rectangle(
            surface_x,
            surface_top,
            surface_x + 35,
            surface_bottom,
            fill="#64707b",
            outline="#b7c1c9",
            width=2,
            tags="static"
        )

        self.canvas.create_text(
            surface_x + 18,
            surface_bottom + 22,
            text="metal",
            fill="#d9edf7",
            font=("Segoe UI", 10),
            tags="static"
        )

        # Vacuum region.
        self.canvas.create_text(
            int(width * 0.78),
            surface_top - 30,
            text="VACUUM",
            fill="#d9edf7",
            font=("Segoe UI", 10, "bold"),
            tags="static"
        )

        # Stopping-potential barrier.
        barrier_x = int(width * 0.88)

        self.canvas.create_line(
            barrier_x,
            surface_top,
            barrier_x,
            surface_bottom,
            fill="#e4b44c",
            width=3,
            tags="static"
        )

        self.canvas.create_text(
            barrier_x,
            surface_bottom + 22,
            text="collector",
            fill="#d9edf7",
            font=("Segoe UI", 10),
            tags="static"
        )

        # Labels.
        self.canvas.create_text(
            surface_x - 80,
            surface_bottom - 15,
            text="photoemitting surface",
            fill="#aebbc5",
            font=("Segoe UI", 9),
            tags="static"
        )

    def _draw_photon(self, x, y):
        self.canvas.create_line(
            x,
            y,
            x + 22,
            y,
            fill="#8fe3ff",
            width=2,
            tags="photon"
        )
        self.canvas.create_oval(
            x + 18,
            y - 4,
            x + 26,
            y + 4,
            fill="#8fe3ff",
            outline="",
            tags="photon"
        )

    def _draw_electron(self, x, y):
        self.canvas.create_oval(
            x - 5,
            y - 5,
            x + 5,
            y + 5,
            fill="#f5d76e",
            outline="",
            tags="electron"
        )

    # ---------------------------------------------------------------
    # Animation
    # ---------------------------------------------------------------

    def _spawn_photon(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width <= 1 or height <= 1:
            return

        surface_x = int(width * 0.58)
        surface_top = int(height * 0.16)
        surface_bottom = int(height * 0.88)

        y = random.randint(
            surface_top + 15,
            surface_bottom - 15
        )

        self.photons.append(
            {
                "x": 20.0,
                "y": float(y),
            }
        )

    def _process_photon_hits(self):
        (
            _metal,
            phi,
            frequency,
            threshold,
            _photon_energy,
            kinetic_energy
        ) = self._current_parameters()

        width = self.canvas.winfo_width()
        surface_x = int(width * 0.58)

        remaining = []

        for photon in self.photons:
            photon["x"] += 14

            if photon["x"] >= surface_x:
                # Photon reaches the metal surface.
                if frequency >= threshold:
                    # Sample an emitted electron energy up to K_max.
                    # The animation is illustrative rather than a
                    # microscopic scattering simulation.
                    energy_fraction = random.uniform(0.15, 1.0)

                    self.electrons.append(
                        {
                            "x": float(surface_x + 38),
                            "y": float(photon["y"]),
                            "speed": 2.0 + 5.0 * math.sqrt(
                                energy_fraction
                            ),
                        }
                    )

                continue

            remaining.append(photon)

        self.photons = remaining

    def _update_electrons(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        surface_x = int(width * 0.58)
        collector_x = int(width * 0.88)

        applied_voltage = self.stopping_voltage_value.get()
        kinetic_max = maximum_kinetic_energy_eV(
            self.frequency.get(),
            WORK_FUNCTIONS[self.metal.get()]
        )

        remaining = []

        for electron in self.electrons:
            # A stopping voltage acts as an opposing barrier.
            if applied_voltage >= kinetic_max and kinetic_max > 0:
                electron["speed"] *= 0.94

                if electron["speed"] < 0.12:
                    continue

            electron["x"] += electron["speed"]

            if electron["x"] >= collector_x:
                continue

            if electron["x"] < surface_x:
                continue

            remaining.append(electron)

        self.electrons = remaining

    def _animate(self):
        self.canvas.delete("photon")
        self.canvas.delete("electron")

        # Spawn rate follows intensity.
        if random.random() < self.intensity.get() / 120:
            self._spawn_photon()

        self._process_photon_hits()
        self._update_electrons()

        for photon in self.photons:
            self._draw_photon(
                photon["x"],
                photon["y"]
            )

        for electron in self.electrons:
            self._draw_electron(
                electron["x"],
                electron["y"]
            )

        self._update_labels()

        self.root.after(
            30,
            self._animate
        )


def main():
    root = tk.Tk()

    # Use a native ttk theme where available.
    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass

    PhotoelectricApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
