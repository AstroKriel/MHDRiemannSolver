## { SCRIPT

## This file is part of the "aegir" (MHDRiemannSolver) project.
## Copyright (c) 2026 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES
##

from pathlib import Path

import matplotlib.pyplot as mpl_plot
import numpy

from aegir import PrimitiveState, mhd_state, sample_profile, solve_riemann_problem

##
## === CONSTANTS
##

## the Ryu & Jones (1995), ApJ 442, 228, "2a" shock tube
MAGNETIC_FIELD_NORMAL = 0.5641895835477562
GAMMA = 5.0 / 3.0
TIME = 0.2
DISCONTINUITY_POSITION = 0.5
NUM_CELLS_REFERENCE = 2001
NUM_CELLS_SAMPLED = 51

##
## === HELPER FUNCTIONS
##


def compute_panels(
    *,
    profile: list[PrimitiveState],
):
    """Return `[(values, latex_label), ...]` for the 8 quantities plotted from `profile`."""
    total_energy = numpy.array(
        [
            mhd_state.compute_total_energy(
                state=state,
                magnetic_field_normal=MAGNETIC_FIELD_NORMAL,
                gamma=GAMMA,
            ) for state in profile
        ],
    )
    return [
        (numpy.array([state.density for state in profile]), r"$\rho$"),
        (numpy.array([state.pressure for state in profile]), r"$p$"),
        (numpy.array([state.velocity_normal for state in profile]), r"$u_1$"),
        (numpy.array([state.velocity_transverse_1 for state in profile]), r"$u_2$"),
        (numpy.array([state.velocity_transverse_2 for state in profile]), r"$u_3$"),
        (numpy.array([state.magnetic_field_transverse_1 for state in profile]), r"$b_2$"),
        (numpy.array([state.magnetic_field_transverse_2 for state in profile]), r"$b_3$"),
        (total_energy, r"$E$"),
    ]


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
    print("Sampling the reference profile...")
    positions_reference = numpy.linspace(0.0, 1.0, NUM_CELLS_REFERENCE)
    profile_reference = sample_profile(
        riemann_solution=solution,
        positions=positions_reference,
        time=TIME,
        discontinuity_position=DISCONTINUITY_POSITION,
    )
    panels_reference = compute_panels(profile=profile_reference)

    print("Sampling the low-resolution profile...")
    positions_sampled = numpy.linspace(0.0, 1.0, NUM_CELLS_SAMPLED)
    profile_sampled = sample_profile(
        riemann_solution=solution,
        positions=positions_sampled,
        time=TIME,
        discontinuity_position=DISCONTINUITY_POSITION,
    )
    panels_sampled = compute_panels(profile=profile_sampled)

    print("Plotting the profile...")
    fig, axs = mpl_plot.subplots(4, 2, figsize=(8, 10), sharex=True)
    for (reference_values, label), (sampled_values, _), ax in zip(panels_reference, panels_sampled, axs.flatten()):
        ax.plot(positions_reference, reference_values, color="black", linewidth=1.0)
        ax.plot(
            positions_sampled,
            sampled_values,
            marker="o",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="black",
            linestyle="none",
        )
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
