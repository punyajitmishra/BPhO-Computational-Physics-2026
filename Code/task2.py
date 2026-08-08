"""
BPhO Computational Physics Challenge 2026 - Quantum Mechanics
TASK #2: Brownian Motion

N small particles (mass m, radius r) travel a mean free path Kn*r before a
random collision changes their headings. A single large particle (mass M,
radius R) starts at rest and is kicked around by elastic,
momentum-conserving collisions with the small particles. The whole system is
animated.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def init_system(N: int, L: float, r: float, R: float, s: float, seed=None):
    """Set up N small particles (random positions, random heading) and one
    big particle at the centre, at rest."""
    rng = np.random.default_rng(seed)

    positions = rng.uniform(r, L - r, size=(N, 2))
    big_pos = np.array([L / 2, L / 2])

    # re-roll any small particle that starts overlapping the big one
    overlap = np.linalg.norm(positions - big_pos, axis=1) < (R + r)
    while overlap.any():
        positions[overlap] = rng.uniform(r, L - r, size=(overlap.sum(), 2))
        overlap = np.linalg.norm(positions - big_pos, axis=1) < (R + r)

    theta = rng.uniform(0, 2 * np.pi, N)
    velocities = s * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    big_vel = np.array([0.0, 0.0])

    return positions, velocities, big_pos, big_vel, rng


def elastic_collision_normal(v1, v2, n, m1, m2):
    """2D elastic collision, updating only the velocity component along the
    line of centres n (tangential components are untouched because frictionless)."""
    v1n_s, v2n_s = np.dot(v1, n), np.dot(v2, n)
    v1t, v2t = v1 - v1n_s * n, v2 - v2n_s * n

    new_v1n_s = ((m1 - m2) * v1n_s + 2 * m2 * v2n_s) / (m1 + m2)
    new_v2n_s = ((m2 - m1) * v2n_s + 2 * m1 * v1n_s) / (m1 + m2)

    return new_v1n_s * n + v1t, new_v2n_s * n + v2t


def step(positions, velocities, big_pos, big_vel, L, r, R, m, M, dt, rng,
         time_to_turn, turn_interval):
    """Advance the system by one timestep dt."""
    positions = positions + velocities * dt
    big_pos = big_pos + big_vel * dt

    # The presentation suggests a reorientation interval corresponding to
    # Kn molecular radii, rather than changing direction every frame.
    time_to_turn -= dt
    if time_to_turn <= 0:
        speed = np.linalg.norm(velocities, axis=1)
        theta = rng.uniform(0, 2 * np.pi, positions.shape[0])
        velocities = speed[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        time_to_turn += turn_interval

    # wall reflection - small particles
    for axis in (0, 1):
        lo, hi = positions[:, axis] < r, positions[:, axis] > L - r
        velocities[lo, axis] *= -1
        velocities[hi, axis] *= -1
        positions[lo, axis] = r
        positions[hi, axis] = L - r

    # wall reflection - big particle
    for axis in (0, 1):
        if big_pos[axis] < R:
            big_pos[axis], big_vel[axis] = R, -big_vel[axis]
        elif big_pos[axis] > L - R:
            big_pos[axis], big_vel[axis] = L - R, -big_vel[axis]

    # collisions with the big particle - conserve momentum (elastic)
    d = positions - big_pos
    dist = np.linalg.norm(d, axis=1)
    for i in np.where(dist < (R + r))[0]:
        n = d[i] / dist[i]
        v_rel_n = np.dot(velocities[i] - big_vel, n)
        if v_rel_n < 0:  # only resolve if actually approaching
            velocities[i], big_vel = elastic_collision_normal(
                velocities[i], big_vel, n, m, M
            )
            positions[i] = big_pos + n * (R + r)  # de-overlap

    return positions, velocities, big_pos, big_vel, time_to_turn


def animate_system(N=150, L=100.0, r=0.6, R=4.0, m=1.0, M=200.0,
                    s=2.0, knudsen_number=10.0, dt=1.0, n_frames=200, seed=1,
                    outfile="task2_brownian_motion.gif"):
    positions, velocities, big_pos, big_vel, rng = init_system(N, L, r, R, s, seed)
    # Units may be interpreted as nm and ps: mean free path = Kn*r.
    turn_interval = knudsen_number * r / s
    time_to_turn = turn_interval
    trail = [big_pos.copy()]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    small_scatter = ax.scatter(positions[:, 0], positions[:, 1],
                                s=8, c="tab:blue", zorder=2)
    trail_line, = ax.plot([], [], c="tab:red", lw=1, zorder=3)
    big_circle = plt.Circle(big_pos, R, color="tab:red", zorder=4)
    ax.add_patch(big_circle)
    title = ax.set_title("")

    def update(frame):
        nonlocal positions, velocities, big_pos, big_vel, time_to_turn
        positions, velocities, big_pos, big_vel, time_to_turn = step(
            positions, velocities, big_pos, big_vel, L, r, R, m, M, dt, rng,
            time_to_turn, turn_interval
        )
        trail.append(big_pos.copy())

        small_scatter.set_offsets(positions)
        big_circle.center = big_pos
        trail_arr = np.array(trail[-60:])  # keep trail short & readable
        trail_line.set_data(trail_arr[:, 0], trail_arr[:, 1])
        title.set_text(
            f"Brownian motion: Kn = {knudsen_number:g}, frame {frame+1}/{n_frames}"
        )
        return small_scatter, big_circle, trail_line, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
    anim.save(outfile, writer=PillowWriter(fps=25))
    plt.close(fig)
    return outfile


if __name__ == "__main__":
    animate_system()
