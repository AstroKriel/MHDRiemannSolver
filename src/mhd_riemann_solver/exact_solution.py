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
## === WAVE PROPAGATION
##


class WaveType(str, Enum):
    Shock = "shock"
    Rarefaction = "rarefaction"


@dataclass(frozen=True)
class WavePropagation:
    """
    Propagation speed of one wave in the fan.

    Fields
    ---
    - `wave_type`:
        `WaveType.Shock` for a shock, rotational discontinuity, or contact (all
        sharp, `head_speed == tail_speed`); `WaveType.Rarefaction` for a
        rarefaction fan, spanning `[head_speed, tail_speed]`.

    - `head_speed`, `tail_speed`:
        Propagation speed(s) of the wave; equal for `WaveType.Shock`.
    """

    wave_type: WaveType
    head_speed: float
    tail_speed: float


@dataclass(frozen=True)
class Wave:
    """
    One wave of the fan, paired with the constant state immediately to its
    right (i.e. reached once the self-similar speed exceeds
    `wave_propagation.head_speed`).
    """

    wave_propagation: WavePropagation
    state: PrimitiveState


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
) -> tuple[PrimitiveState, WavePropagation]:
    """Solve one fast/slow wave, dispatching to a shock or a rarefaction by the sign of `pressure_downstream - upstream_state.pressure`."""
    c_fast_upstream, c_slow_upstream = mhd_state.compute_fast_slow_speeds(
        state=upstream_state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    reference_speed_upstream = c_fast_upstream if wave_family == WaveFamily.Fast else c_slow_upstream
    if pressure_downstream > upstream_state.pressure:
        downstream_state, shock_speed = solve_shock.solve_shock(
            upstream_state=upstream_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=pressure_downstream,
            initial_relative_speed_guess=-wave_speed_sign * reference_speed_upstream,
        )
        return downstream_state, WavePropagation(
            wave_type=WaveType.Shock,
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
    c_fast_downstream, c_slow_downstream = mhd_state.compute_fast_slow_speeds(
        state=downstream_state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    reference_speed_downstream = c_fast_downstream if wave_family == WaveFamily.Fast else c_slow_downstream
    head_speed = upstream_state.velocity_normal + wave_speed_sign * reference_speed_upstream
    tail_speed = downstream_state.velocity_normal + wave_speed_sign * reference_speed_downstream
    return downstream_state, WavePropagation(
        wave_type=WaveType.Rarefaction,
        head_speed=min(head_speed, tail_speed),
        tail_speed=max(head_speed, tail_speed),
    )


def _compute_rotation_wave_info(
    *,
    upstream_state: PrimitiveState,
    magnetic_field_normal: float,
    sign: float,
) -> WavePropagation:
    speed = upstream_state.velocity_normal - sign * magnetic_field_normal / numpy.sqrt(upstream_state.density)
    return WavePropagation(
        wave_type=WaveType.Shock,
        head_speed=speed,
        tail_speed=speed,
    )


##
## === WAVE REGIONS
##


@dataclass(frozen=True)
class _WaveRegions:
    """
    The 6 solved waves of the fan, each paired with the constant state
    immediately to its right; the caller-given `left_state`/`right_state`
    boundaries, and the contact wave (synthesized afterwards from
    `slow_left`), are not included here.
    """

    fast_left: Wave
    rotation_left: Wave
    slow_left: Wave
    contact_state: PrimitiveState
    slow_right: Wave
    rotation_right: Wave
    fast_right: Wave


##
## === RIEMANN SOLUTION
##


@dataclass(frozen=True)
class RiemannSolution:
    """
    The fully-resolved 1D ideal-MHD Riemann fan.

    Fields
    ---
    - `left_state`:
        The given left state.

    - `fast_left`, `rotation_left`, `slow_left`, `contact`, `slow_right`,
      `rotation_right`, `fast_right`:
        The 7 waves, left to right; each pairs a `WavePropagation` with the
        constant state immediately to its right (`fast_right.state` is the
        given right state).
    """

    left_state: PrimitiveState
    fast_left: Wave
    rotation_left: Wave
    slow_left: Wave
    contact: Wave
    slow_right: Wave
    rotation_right: Wave
    fast_right: Wave


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
    ) -> _WaveRegions:
        (
            fast_left_downstream_pressure,
            psi_left,
            pressure_star,
            psi_right,
            fast_right_downstream_pressure,
        ) = unknowns
        fast_left_downstream_state, fast_left_propagation = _solve_wave(
            upstream_state=left_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=fast_left_downstream_pressure,
            wave_family=WaveFamily.Fast,
            wave_speed_sign=-1.0,
        )
        rotation_left_downstream_state = rotational_discontinuity.apply_rotation(
            upstream_state=fast_left_downstream_state,
            angle=psi_left,
            sign=rotation_sign_left,
        )
        rotation_left_propagation = _compute_rotation_wave_info(
            upstream_state=fast_left_downstream_state,
            magnetic_field_normal=magnetic_field_normal,
            sign=rotation_sign_left,
        )
        slow_left_downstream_state, slow_left_propagation = _solve_wave(
            upstream_state=rotation_left_downstream_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=pressure_star,
            wave_family=WaveFamily.Slow,
            wave_speed_sign=-1.0,
        )
        fast_right_downstream_state, fast_right_propagation = _solve_wave(
            upstream_state=right_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=fast_right_downstream_pressure,
            wave_family=WaveFamily.Fast,
            wave_speed_sign=1.0,
        )
        rotation_right_downstream_state = rotational_discontinuity.apply_rotation(
            upstream_state=fast_right_downstream_state,
            angle=psi_right,
            sign=rotation_sign_right,
        )
        rotation_right_propagation = _compute_rotation_wave_info(
            upstream_state=fast_right_downstream_state,
            magnetic_field_normal=magnetic_field_normal,
            sign=rotation_sign_right,
        )
        slow_right_downstream_state, slow_right_propagation = _solve_wave(
            upstream_state=rotation_right_downstream_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=pressure_star,
            wave_family=WaveFamily.Slow,
            wave_speed_sign=1.0,
        )
        return _WaveRegions(
            fast_left=Wave(
                wave_propagation=fast_left_propagation,
                state=fast_left_downstream_state,
            ),
            rotation_left=Wave(
                wave_propagation=rotation_left_propagation,
                state=rotation_left_downstream_state,
            ),
            slow_left=Wave(
                wave_propagation=slow_left_propagation,
                state=slow_left_downstream_state,
            ),
            contact_state=slow_right_downstream_state,
            slow_right=Wave(
                wave_propagation=slow_right_propagation,
                state=rotation_right_downstream_state,
            ),
            rotation_right=Wave(
                wave_propagation=rotation_right_propagation,
                state=fast_right_downstream_state,
            ),
            fast_right=Wave(
                wave_propagation=fast_right_propagation,
                state=right_state,
            ),
        )

    def residuals(
        unknowns: NDArray[Any],
    ) -> NDArray[Any]:
        region_set = build_regions(unknowns)
        slow_left_state = region_set.slow_left.state
        contact_state = region_set.contact_state
        return numpy.array(
            [
                slow_left_state.velocity_normal - contact_state.velocity_normal,
                slow_left_state.velocity_transverse_1 - contact_state.velocity_transverse_1,
                slow_left_state.velocity_transverse_2 - contact_state.velocity_transverse_2,
                slow_left_state.magnetic_field_transverse_1 - contact_state.magnetic_field_transverse_1,
                slow_left_state.magnetic_field_transverse_2 - contact_state.magnetic_field_transverse_2,
            ],
        )

    initial_guess = numpy.array(
        [
            left_state.pressure * 1.2,
            0.0,
            0.5 * (left_state.pressure + right_state.pressure),
            0.0,
            right_state.pressure * 1.2,
        ],
    )
    solution = scipy_root(
        residuals,
        x0=initial_guess,
        method="hybr",
    )
    if not solution.success:
        raise RuntimeError(f"riemann-problem root-find did not converge: {solution.message}.")

    region_set = build_regions(solution.x)
    contact_speed = region_set.slow_left.state.velocity_normal
    return RiemannSolution(
        left_state=left_state,
        fast_left=region_set.fast_left,
        rotation_left=region_set.rotation_left,
        slow_left=region_set.slow_left,
        contact=Wave(
            wave_propagation=WavePropagation(
                wave_type=WaveType.Shock,
                head_speed=contact_speed,
                tail_speed=contact_speed,
            ),
            state=region_set.contact_state,
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
        solution.fast_left,
        solution.rotation_left,
        solution.slow_left,
        solution.contact,
        solution.slow_right,
        solution.rotation_right,
        solution.fast_right,
    ]
    profile: list[PrimitiveState] = []
    for position in x:
        self_similar_speed = (position - x0) / t
        state = solution.left_state
        for wave in waves:
            if self_similar_speed < wave.wave_propagation.head_speed:
                break
            if (wave.wave_propagation.wave_type == WaveType.Rarefaction
                    and self_similar_speed < wave.wave_propagation.tail_speed):
                raise NotImplementedError(
                    f"position `{position}` at t=`{t}` falls inside a rarefaction fan; "
                    "fan-interior sampling is not yet supported.",
                )
            state = wave.state
        profile.append(state)
    return profile


## } MODULE
