## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray
from scipy.optimize import root as scipy_root

## local
from riemann_solver import mhd_state, rotational_discontinuity, solve_rarefaction, solve_shock
from riemann_solver.mhd_state import PrimitiveState, WaveFamily

##
## === WAVE INFO
##


class WaveKind(str, Enum):
    Shock = "shock"
    Rarefaction = "rarefaction"


@dataclass(frozen=True)
class WaveInfo:
    """
    Propagation speed of one wave in the fan.

    Fields
    ---
    - `kind`:
        `WaveKind.Shock` for a shock, rotational discontinuity, or contact (all
        sharp, `head_speed == tail_speed`); `WaveKind.Rarefaction` for a
        rarefaction fan, spanning `[head_speed, tail_speed]`.

    - `head_speed`, `tail_speed`:
        Propagation speed(s) of the wave; equal for `WaveKind.Shock`.
    """

    kind: WaveKind
    head_speed: float
    tail_speed: float


##
## === WAVE DISPATCH
##


def _solve_wave(
    *,
    upstream_state: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
    pressure_downstream: float,
    wave_family: WaveFamily,
    wave_speed_sign: float,
) -> tuple[PrimitiveState, WaveInfo]:
    """Solve one fast/slow wave, dispatching to a shock or a rarefaction by the sign of `pressure_downstream - upstream_state.pressure`."""
    c_fast_up, c_slow_up = mhd_state.compute_fast_slow_speeds(
        state=upstream_state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    reference_speed_up = c_fast_up if wave_family == WaveFamily.Fast else c_slow_up
    if pressure_downstream > upstream_state.pressure:
        downstream_state, shock_speed = solve_shock.solve_shock(
            upstream_state=upstream_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=pressure_downstream,
            initial_relative_speed_guess=-wave_speed_sign * reference_speed_up,
        )
        return downstream_state, WaveInfo(
            kind=WaveKind.Shock,
            head_speed=shock_speed,
            tail_speed=shock_speed,
        )
    downstream_state = solve_rarefaction.solve_rarefaction(
        upstream_state=upstream_state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
        pressure_downstream=pressure_downstream,
        wave_family=wave_family,
        wave_speed_sign=wave_speed_sign,
    )
    c_fast_down, c_slow_down = mhd_state.compute_fast_slow_speeds(
        state=downstream_state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    reference_speed_down = c_fast_down if wave_family == WaveFamily.Fast else c_slow_down
    head_speed = upstream_state.velocity_normal + wave_speed_sign * reference_speed_up
    tail_speed = downstream_state.velocity_normal + wave_speed_sign * reference_speed_down
    return downstream_state, WaveInfo(
        kind=WaveKind.Rarefaction,
        head_speed=min(head_speed, tail_speed),
        tail_speed=max(head_speed, tail_speed),
    )


def _compute_rotation_wave_info(
    *,
    upstream_state: PrimitiveState,
    magnetic_field_normal: float,
    sign: float,
) -> WaveInfo:
    speed = upstream_state.velocity_normal - sign * magnetic_field_normal / numpy.sqrt(upstream_state.density)
    return WaveInfo(
        kind=WaveKind.Shock,
        head_speed=speed,
        tail_speed=speed,
    )


##
## === REGION SET
##


@dataclass(frozen=True)
class _RegionSet:
    """The 6 solved intermediate states and their bounding waves, for one candidate root-find unknown vector."""

    region2: PrimitiveState
    region3: PrimitiveState
    region4: PrimitiveState
    region5: PrimitiveState
    region6: PrimitiveState
    region7: PrimitiveState
    fast_left: WaveInfo
    rotation_left: WaveInfo
    slow_left: WaveInfo
    slow_right: WaveInfo
    rotation_right: WaveInfo
    fast_right: WaveInfo


##
## === RIEMANN SOLUTION
##


@dataclass(frozen=True)
class RiemannSolution:
    """
    The 8 constant states of a fully-resolved 1D ideal-MHD Riemann fan, and each
    wave's propagation speed.

    Fields
    ---
    - `region1`, `region8`:
        The given left/right states.

    - `region2`, `region3`:
        Left of the contact, either side of the left rotational discontinuity.

    - `region4`:
        Left of the contact, after the left slow wave.

    - `region5`:
        Right of the contact, before the right slow wave.

    - `region6`, `region7`:
        Right of the contact, either side of the right rotational discontinuity.

    - `fast_left`, `rotation_left`, `slow_left`, `contact`, `slow_right`,
      `rotation_right`, `fast_right`:
        The 7 waves, left to right.
    """

    region1: PrimitiveState
    region2: PrimitiveState
    region3: PrimitiveState
    region4: PrimitiveState
    region5: PrimitiveState
    region6: PrimitiveState
    region7: PrimitiveState
    region8: PrimitiveState
    fast_left: WaveInfo
    rotation_left: WaveInfo
    slow_left: WaveInfo
    contact: WaveInfo
    slow_right: WaveInfo
    rotation_right: WaveInfo
    fast_right: WaveInfo


##
## === COUPLED SOLVE
##


def solve_riemann_problem(
    *,
    left_state: PrimitiveState,
    right_state: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
) -> RiemannSolution:
    """
    Solve the full 7-wave 1D ideal-MHD Riemann problem, root-finding the 5
    unknowns that pin down the fan; whether each fast/slow wave is a shock or
    a rarefaction falls out of the solve, not assumed up front (see `_solve_wave`).
    """
    ## rotation signs fixed by the eigenvalue ordering u-|bx|/sqrt(rho) < ... < u+|bx|/sqrt(rho)
    rotation_sign_left = 1.0 if magnetic_field_normal >= 0.0 else -1.0
    rotation_sign_right = -rotation_sign_left

    def build_regions(
        unknowns: NDArray[Any],
    ) -> _RegionSet:
        pressure_2, psi_left, pressure_star, psi_right, pressure_7 = unknowns
        region2_state, fast_left = _solve_wave(
            upstream_state=left_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=pressure_2,
            wave_family=WaveFamily.Fast,
            wave_speed_sign=-1.0,
        )
        region3_state = rotational_discontinuity.apply_rotation(
            upstream_state=region2_state,
            angle=psi_left,
            sign=rotation_sign_left,
        )
        rotation_left = _compute_rotation_wave_info(
            upstream_state=region2_state,
            magnetic_field_normal=magnetic_field_normal,
            sign=rotation_sign_left,
        )
        region4_state, slow_left = _solve_wave(
            upstream_state=region3_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=pressure_star,
            wave_family=WaveFamily.Slow,
            wave_speed_sign=-1.0,
        )
        region7_state, fast_right = _solve_wave(
            upstream_state=right_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=pressure_7,
            wave_family=WaveFamily.Fast,
            wave_speed_sign=1.0,
        )
        region6_state = rotational_discontinuity.apply_rotation(
            upstream_state=region7_state,
            angle=psi_right,
            sign=rotation_sign_right,
        )
        rotation_right = _compute_rotation_wave_info(
            upstream_state=region7_state,
            magnetic_field_normal=magnetic_field_normal,
            sign=rotation_sign_right,
        )
        region5_state, slow_right = _solve_wave(
            upstream_state=region6_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=pressure_star,
            wave_family=WaveFamily.Slow,
            wave_speed_sign=1.0,
        )
        return _RegionSet(
            region2=region2_state,
            region3=region3_state,
            region4=region4_state,
            region5=region5_state,
            region6=region6_state,
            region7=region7_state,
            fast_left=fast_left,
            rotation_left=rotation_left,
            slow_left=slow_left,
            slow_right=slow_right,
            rotation_right=rotation_right,
            fast_right=fast_right,
        )

    def residuals(
        unknowns: NDArray[Any],
    ) -> NDArray[Any]:
        region_set = build_regions(unknowns)
        return numpy.array(
            [
                region_set.region4.velocity_normal - region_set.region5.velocity_normal,
                region_set.region4.velocity_transverse_1 - region_set.region5.velocity_transverse_1,
                region_set.region4.velocity_transverse_2 - region_set.region5.velocity_transverse_2,
                region_set.region4.magnetic_field_transverse_1 -
                region_set.region5.magnetic_field_transverse_1,
                region_set.region4.magnetic_field_transverse_2 -
                region_set.region5.magnetic_field_transverse_2,
            ],
        )

    initial_guess = numpy.array(
        [
            left_state.pressure * 1.2, 0.0, 0.5 * (left_state.pressure + right_state.pressure), 0.0,
            right_state.pressure * 1.2
        ]
    )
    solution = scipy_root(
        residuals,
        x0=initial_guess,
        method="hybr",
    )
    if not solution.success:
        raise RuntimeError(f"riemann-problem root-find did not converge: {solution.message}.")

    region_set = build_regions(solution.x)
    contact_speed = region_set.region4.velocity_normal
    return RiemannSolution(
        region1=left_state,
        region2=region_set.region2,
        region3=region_set.region3,
        region4=region_set.region4,
        region5=region_set.region5,
        region6=region_set.region6,
        region7=region_set.region7,
        region8=right_state,
        fast_left=region_set.fast_left,
        rotation_left=region_set.rotation_left,
        slow_left=region_set.slow_left,
        contact=WaveInfo(
            kind=WaveKind.Shock,
            head_speed=contact_speed,
            tail_speed=contact_speed,
        ),
        slow_right=region_set.slow_right,
        rotation_right=region_set.rotation_right,
        fast_right=region_set.fast_right,
    )


##
## === PROFILE EVALUATION
##


def sample_profile(
    *,
    solution: RiemannSolution,
    x: NDArray[Any],
    t: float,
    x0: float = 0.0,
) -> list[PrimitiveState]:
    """
    Sample `solution` at each position in `x` at time `t`, with the initial
    discontinuity at `x0`.

    Raises `NotImplementedError` if any sample falls strictly inside a
    rarefaction fan: fan-interior profiles are not yet interpolated.
    """
    waves = [
        (solution.fast_left, solution.region2),
        (solution.rotation_left, solution.region3),
        (solution.slow_left, solution.region4),
        (solution.contact, solution.region5),
        (solution.slow_right, solution.region6),
        (solution.rotation_right, solution.region7),
        (solution.fast_right, solution.region8),
    ]
    profile: list[PrimitiveState] = []
    for position in x:
        self_similar_speed = (position - x0) / t
        state = solution.region1
        for wave, right_state in waves:
            if self_similar_speed < wave.head_speed:
                break
            if wave.kind == WaveKind.Rarefaction and self_similar_speed < wave.tail_speed:
                raise NotImplementedError(
                    f"position `{position}` at t=`{t}` falls inside a rarefaction fan; "
                    "fan-interior sampling is not yet supported.",
                )
            state = right_state
        profile.append(state)
    return profile


## } MODULE
