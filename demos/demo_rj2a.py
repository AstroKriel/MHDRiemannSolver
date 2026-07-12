## { SCRIPT

## This file is part of the "MHDRiemannSolver" project (local alias: "aegir").
## Copyright (c) 2026 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES
##

from pathlib import Path

import matplotlib.pyplot as mpl_plot
import numpy

from mhd_riemann_solver import PrimitiveState, sample_profile, solve_riemann_problem
from mhd_riemann_solver import mhd_state

##
## === CONSTANTS
##

## the Ryu & Jones (1995), ApJ 442, 228, "2a" shock tube
MAGNETIC_FIELD_NORMAL = 0.5641895835477562
GAMMA = 5.0 / 3.0
TIME = 0.2
DISCONTINUITY_POSITION = 0.5

##
## === PROGRAM MAIN
##


def main():
    left_state = PrimitiveState(
        density=1.08,
        velocity_normal=1.2,
        velocity_transverse_1=0.01,
        velocity_transverse_2=0.5,
        magnetic_field_transverse_1=1.0155412503859613,
        magnetic_field_transverse_2=0.5641895835477562,
        pressure=0.95,
    )
    right_state = PrimitiveState(
        density=1.0,
        velocity_normal=0.0,
        velocity_transverse_1=0.0,
        velocity_transverse_2=0.0,
        magnetic_field_transverse_1=1.1283791670955125,
        magnetic_field_transverse_2=0.5641895835477562,
        pressure=1.0,
    )
    print("Solving the Riemann problem...")
    solution = solve_riemann_problem(
        left_state=left_state,
        right_state=right_state,
        magnetic_field_normal=MAGNETIC_FIELD_NORMAL,
        gamma=GAMMA,
    )
    print("Sampling the profile...")
    positions = numpy.linspace(0.0, 1.0, 2001)
    profile = sample_profile(
        riemann_solution=solution,
        positions=positions,
        time=TIME,
        discontinuity_position=DISCONTINUITY_POSITION,
    )
    density = numpy.array([state.density for state in profile])
    pressure = numpy.array([state.pressure for state in profile])
    velocity_normal = numpy.array([state.velocity_normal for state in profile])
    velocity_transverse_1 = numpy.array([state.velocity_transverse_1 for state in profile])
    velocity_transverse_2 = numpy.array([state.velocity_transverse_2 for state in profile])
    magnetic_field_transverse_1 = numpy.array([state.magnetic_field_transverse_1 for state in profile])
    magnetic_field_transverse_2 = numpy.array([state.magnetic_field_transverse_2 for state in profile])
    total_energy = numpy.array(
        [
            mhd_state.compute_total_energy(
                state=state,
                magnetic_field_normal=MAGNETIC_FIELD_NORMAL,
                gamma=GAMMA,
            ) for state in profile
        ],
    )

    print("Plotting the profile...")
    fig, axs = mpl_plot.subplots(4, 2, figsize=(8, 10), sharex=True)
    panels = [
        (density, r"$\rho$"),
        (pressure, r"$p$"),
        (velocity_normal, r"$u_1$"),
        (velocity_transverse_1, r"$u_2$"),
        (velocity_transverse_2, r"$u_3$"),
        (magnetic_field_transverse_1, r"$b_2$"),
        (magnetic_field_transverse_2, r"$b_3$"),
        (total_energy, r"$E$"),
    ]
    for (values, label), ax in zip(panels, axs.flatten()):
        ax.plot(positions, values, color="black", linewidth=1.0)
        ax.set_ylabel(label)
    axs[-1, 0].set_xlabel(r"$x_1$")
    axs[-1, 1].set_xlabel(r"$x_1$")
    fig.tight_layout()

    print("Saving figure...")
    gallery_dir = Path(__file__).parents[1] / "gallery"
    gallery_dir.mkdir(exist_ok=True)
    fig_path = gallery_dir / "ryu_jones_2a.png"
    fig.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
    )
    mpl_plot.close(fig)
    print("Saved:", fig_path)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
