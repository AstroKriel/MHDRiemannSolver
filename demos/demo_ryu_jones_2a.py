## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

## third-party
import matplotlib.pyplot as mpl_plot
import numpy
from matplotlib import colormaps
from matplotlib.figure import Figure
from numpy.typing import NDArray

## local
from aegir import exact_solution, mhd_state

##
## === PROFILE EVOLUTION
##


@dataclass
class ProfileEvolution:
    """
    One physical quantity's spatial profile, evolving across every sampled time.

    Fields
    ---
    - `values_by_time`:
        `values_by_time[i]` is this quantity's values across `positions`, at `times[i]`.
    """

    label: str
    values_by_time: list[NDArray[Any]]


##
## === PROBLEM SETUP
##
## Edit this section to point the demo at a different Riemann problem: the
## left/right states either side of the discontinuity, the (constant) normal
## field, and the equation of state.
##

MAGNETIC_FIELD_NORMAL = 0.5641895835477562
GAMMA = 5.0 / 3.0


def build_ryu_jones_2a_states() -> tuple[mhd_state.PrimitiveState, mhd_state.PrimitiveState]:
    """Return `(left_state, right_state)` for the Ryu & Jones (1995), ApJ 442, 228, "2a" shock tube."""
    left_state = mhd_state.PrimitiveState(
        density=1.08,
        velocity_normal=1.2,
        velocity_transverse_1=0.01,
        velocity_transverse_2=0.5,
        magnetic_field_transverse_1=1.0155412503859613,
        magnetic_field_transverse_2=0.5641895835477562,
        pressure=0.95,
    )
    right_state = mhd_state.PrimitiveState(
        density=1.0,
        velocity_normal=0.0,
        velocity_transverse_1=0.0,
        velocity_transverse_2=0.0,
        magnetic_field_transverse_1=1.1283791670955125,
        magnetic_field_transverse_2=0.5641895835477562,
        pressure=1.0,
    )
    return left_state, right_state


##
## === SAMPLING SETUP
##
## Edit this section to change where/when the (self-similar) solution is
## sampled: spatial resolution, the discontinuity's position in that domain,
## and the range of times shown as ridges.
##

## off-center: the fan is asymmetric (the right-going fast wave outruns the left-going one)
DISCONTINUITY_POSITION = 0.35
NUM_POSITIONS = 1001
TIME_MIN = 1e-9
TIME_MAX = 0.3
NUM_TIMES = 8

##
## === PLOT STYLE
##
## Edit this section to change how the ridgeline wave is drawn.
##

## vertical spacing between ridge baselines
LANE_HEIGHT = 0.6
## each ridge's own curve height, as a fraction of `LANE_HEIGHT`
AMPLITUDE_SCALE = 0.6
LINEWIDTH = 2.0
COLORMAP_NAME = "viridis"

##
## === HELPER FUNCTIONS
##


def extract_profile_evolutions(
    *,
    snapshots_by_time: list[list[mhd_state.PrimitiveState]],
) -> list[ProfileEvolution]:
    """Return one `ProfileEvolution` per quantity, each holding its values at every snapshot in `snapshots_by_time`."""
    return [
        ProfileEvolution(
            label="gas density",
            values_by_time=[
                numpy.array([snapshot_state.density for snapshot_state in snapshot])
                for snapshot in snapshots_by_time
            ],
        ),
        ProfileEvolution(
            label="gas pressure",
            values_by_time=[
                numpy.array([snapshot_state.pressure for snapshot_state in snapshot])
                for snapshot in snapshots_by_time
            ],
        ),
        ProfileEvolution(
            label="velocity normal-component",
            values_by_time=[
                numpy.array([snapshot_state.velocity_normal for snapshot_state in snapshot])
                for snapshot in snapshots_by_time
            ],
        ),
        ProfileEvolution(
            label=r"velocity $\perp_1$ component",
            values_by_time=[
                numpy.array([snapshot_state.velocity_transverse_1 for snapshot_state in snapshot])
                for snapshot in snapshots_by_time
            ],
        ),
        ProfileEvolution(
            label=r"velocity $\perp_2$ component",
            values_by_time=[
                numpy.array([snapshot_state.velocity_transverse_2 for snapshot_state in snapshot])
                for snapshot in snapshots_by_time
            ],
        ),
        ProfileEvolution(
            label=r"magnetic $\perp_1$ component",
            values_by_time=[
                numpy.array([snapshot_state.magnetic_field_transverse_1 for snapshot_state in snapshot])
                for snapshot in snapshots_by_time
            ],
        ),
        ProfileEvolution(
            label=r"magnetic $\perp_2$ component",
            values_by_time=[
                numpy.array([snapshot_state.magnetic_field_transverse_2 for snapshot_state in snapshot])
                for snapshot in snapshots_by_time
            ],
        ),
        ProfileEvolution(
            label="total energy",
            values_by_time=[
                numpy.array(
                    [
                        mhd_state.compute_total_energy(
                            state=snapshot_state,
                            magnetic_field_normal=MAGNETIC_FIELD_NORMAL,
                            gamma=GAMMA,
                        ) for snapshot_state in snapshot
                    ],
                ) for snapshot in snapshots_by_time
            ],
        ),
    ]


def sample_profile_evolutions(
    *,
    solution: exact_solution.RiemannSolution,
    positions: NDArray[Any],
    times: NDArray[Any],
) -> list[ProfileEvolution]:
    """Return one `ProfileEvolution` per quantity, each sampled at every time in `times`."""
    snapshots_by_time = [
        exact_solution.sample_snapshot(
            riemann_solution=solution,
            positions=positions,
            time=float(time),
            discontinuity_position=DISCONTINUITY_POSITION,
        ) for time in times
    ]
    return extract_profile_evolutions(snapshots_by_time=snapshots_by_time)


def plot_wave_grid(
    *,
    profile_evolutions: list[ProfileEvolution],
    positions: NDArray[Any],
    times: NDArray[Any],
) -> Figure:
    """Return a 4x2 grid figure, one ridgeline wave panel per quantity in `profile_evolutions`."""
    mpl_plot.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        },
    )
    num_times = len(times)
    amplitude = AMPLITUDE_SCALE * LANE_HEIGHT
    cmap = colormaps[COLORMAP_NAME]
    fig, axs = mpl_plot.subplots(4, 2, figsize=(9, 16))
    for ax, profile_evolution in zip(axs.flatten(), profile_evolutions):
        all_values = numpy.concatenate(profile_evolution.values_by_time)
        value_min, value_max = all_values.min(), all_values.max()
        value_span = value_max - value_min
        for ridge_index, (time, ridge_values) in enumerate(zip(
                times,
                profile_evolution.values_by_time,
        ), ):
            baseline = ridge_index * LANE_HEIGHT
            norm_values = (ridge_values - value_min) / value_span
            ridge_curve = baseline + amplitude * norm_values
            ax.plot(
                positions - DISCONTINUITY_POSITION,
                ridge_curve,
                color=cmap((time - TIME_MIN) / (TIME_MAX - TIME_MIN)),
                linewidth=LINEWIDTH,
                zorder=ridge_index,
            )
        ax.set_yticks([])
        stack_top = (num_times - 1) * LANE_HEIGHT + amplitude
        ax.set_ylim(-0.1 * LANE_HEIGHT, stack_top + 0.6)  # fixed headroom, not scaled by LANE_HEIGHT
        ax.text(
            0.5,
            0.97,
            profile_evolution.label,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
        )
    axs[-1, 0].set_xlabel(r"$x - x_{\mathrm{discontinuity}}$")
    axs[-1, 1].set_xlabel(r"$x - x_{\mathrm{discontinuity}}$")
    fig.tight_layout(h_pad=2.0)
    return fig


##
## === PROGRAM MAIN
##


def main():
    left_state, right_state = build_ryu_jones_2a_states()

    print("Solving the Riemann problem...")
    solution = exact_solution.solve_riemann_problem(
        left_state=left_state,
        right_state=right_state,
        magnetic_field_normal=MAGNETIC_FIELD_NORMAL,
        gamma=GAMMA,
    )

    print("Sampling every quantity at each time...")
    positions = numpy.linspace(0.0, 1.0, NUM_POSITIONS)
    times = numpy.linspace(TIME_MIN, TIME_MAX, NUM_TIMES)
    profile_evolutions = sample_profile_evolutions(
        solution=solution,
        positions=positions,
        times=times,
    )

    print("Plotting the waves...")
    fig = plot_wave_grid(
        profile_evolutions=profile_evolutions,
        positions=positions,
        times=times,
    )

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
