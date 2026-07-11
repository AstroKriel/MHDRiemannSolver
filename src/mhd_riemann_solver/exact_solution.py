## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

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
    `left_slow_wave`), are not included here.
    """

    left_fast_wave: Wave
    left_rotation_discontinuity: Wave
    left_slow_wave: Wave
    contact_state: PrimitiveState
    right_slow_wave: Wave
    right_rotation_discontinuity: Wave
    right_fast_wave: Wave


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

    - `left_fast_wave`, `left_rotation_discontinuity`, `left_slow_wave`,
      `contact`, `right_slow_wave`, `right_rotation_discontinuity`,
      `right_fast_wave`:
        The 7 waves, left to right; each pairs a `WavePropagation` with the
        constant state immediately to its right (`right_fast_wave.state` is
        the given right state).
    """

    left_state: PrimitiveState
    left_fast_wave: Wave
    left_rotation_discontinuity: Wave
    left_slow_wave: Wave
    contact: Wave
    right_slow_wave: Wave
    right_rotation_discontinuity: Wave
    right_fast_wave: Wave


##
## === RIEMANN PARAMETERS
##

## flattened, positionally-fixed encoding of a `_RiemannParams`, for `scipy.optimize.root`
_ParamsVector: TypeAlias = NDArray[Any]


@dataclass(frozen=True)
class _RiemannParams:
    """The 5 free parameters root-found to pin down the fan."""

    left_fast_wave_downstream_pressure: float
    left_rotation_angle: float
    pressure_star: float
    right_rotation_angle: float
    right_fast_wave_downstream_pressure: float


def _as_params_vector(
    *,
    params: _RiemannParams,
) -> _ParamsVector:
    """Return `params` flattened to the vector `scipy.optimize.root` operates on."""
    return numpy.array(
        [
            params.left_fast_wave_downstream_pressure,
            params.left_rotation_angle,
            params.pressure_star,
            params.right_rotation_angle,
            params.right_fast_wave_downstream_pressure,
        ],
    )


def _params_from_vector(
    *,
    params_vector: _ParamsVector,
) -> _RiemannParams:
    (
        left_fast_wave_downstream_pressure,
        left_rotation_angle,
        pressure_star,
        right_rotation_angle,
        right_fast_wave_downstream_pressure,
    ) = params_vector
    return _RiemannParams(
        left_fast_wave_downstream_pressure=left_fast_wave_downstream_pressure,
        left_rotation_angle=left_rotation_angle,
        pressure_star=pressure_star,
        right_rotation_angle=right_rotation_angle,
        right_fast_wave_downstream_pressure=right_fast_wave_downstream_pressure,
    )


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
    parameters that pin down the fan; whether each fast/slow wave is a shock or
    a rarefaction falls out of the solve, not assumed up front (see `_solve_wave`).
    """
    ## rotation signs fixed by the eigenvalue ordering u-|bx|/sqrt(rho) < ... < u+|bx|/sqrt(rho)
    left_rotation_sign = 1.0 if magnetic_field_normal >= 0.0 else -1.0
    right_rotation_sign = -left_rotation_sign

    def build_regions(
        params: _RiemannParams,
    ) -> _WaveRegions:
        left_fast_wave_downstream_state, left_fast_wave_propagation = _solve_wave(
            upstream_state=left_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=params.left_fast_wave_downstream_pressure,
            wave_family=WaveFamily.Fast,
            wave_speed_sign=-1.0,
        )
        left_rotation_discontinuity_downstream_state = rotational_discontinuity.apply_rotation(
            upstream_state=left_fast_wave_downstream_state,
            angle=params.left_rotation_angle,
            sign=left_rotation_sign,
        )
        left_rotation_discontinuity_propagation = _compute_rotation_wave_info(
            upstream_state=left_fast_wave_downstream_state,
            magnetic_field_normal=magnetic_field_normal,
            sign=left_rotation_sign,
        )
        left_slow_wave_downstream_state, left_slow_wave_propagation = _solve_wave(
            upstream_state=left_rotation_discontinuity_downstream_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=params.pressure_star,
            wave_family=WaveFamily.Slow,
            wave_speed_sign=-1.0,
        )
        right_fast_wave_downstream_state, right_fast_wave_propagation = _solve_wave(
            upstream_state=right_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=params.right_fast_wave_downstream_pressure,
            wave_family=WaveFamily.Fast,
            wave_speed_sign=1.0,
        )
        right_rotation_discontinuity_downstream_state = rotational_discontinuity.apply_rotation(
            upstream_state=right_fast_wave_downstream_state,
            angle=params.right_rotation_angle,
            sign=right_rotation_sign,
        )
        right_rotation_discontinuity_propagation = _compute_rotation_wave_info(
            upstream_state=right_fast_wave_downstream_state,
            magnetic_field_normal=magnetic_field_normal,
            sign=right_rotation_sign,
        )
        right_slow_wave_downstream_state, right_slow_wave_propagation = _solve_wave(
            upstream_state=right_rotation_discontinuity_downstream_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=params.pressure_star,
            wave_family=WaveFamily.Slow,
            wave_speed_sign=1.0,
        )
        return _WaveRegions(
            left_fast_wave=Wave(
                wave_propagation=left_fast_wave_propagation,
                state=left_fast_wave_downstream_state,
            ),
            left_rotation_discontinuity=Wave(
                wave_propagation=left_rotation_discontinuity_propagation,
                state=left_rotation_discontinuity_downstream_state,
            ),
            left_slow_wave=Wave(
                wave_propagation=left_slow_wave_propagation,
                state=left_slow_wave_downstream_state,
            ),
            contact_state=right_slow_wave_downstream_state,
            right_slow_wave=Wave(
                wave_propagation=right_slow_wave_propagation,
                state=right_rotation_discontinuity_downstream_state,
            ),
            right_rotation_discontinuity=Wave(
                wave_propagation=right_rotation_discontinuity_propagation,
                state=right_fast_wave_downstream_state,
            ),
            right_fast_wave=Wave(
                wave_propagation=right_fast_wave_propagation,
                state=right_state,
            ),
        )

    def compute_contact_residual(
        params_vector: _ParamsVector,
    ) -> NDArray[Any]:
        region_set = build_regions(_params_from_vector(params_vector=params_vector))
        left_slow_wave_state = region_set.left_slow_wave.state
        contact_state = region_set.contact_state
        return numpy.array(
            [
                left_slow_wave_state.velocity_normal - contact_state.velocity_normal,
                left_slow_wave_state.velocity_transverse_1 - contact_state.velocity_transverse_1,
                left_slow_wave_state.velocity_transverse_2 - contact_state.velocity_transverse_2,
                left_slow_wave_state.magnetic_field_transverse_1 - contact_state.magnetic_field_transverse_1,
                left_slow_wave_state.magnetic_field_transverse_2 - contact_state.magnetic_field_transverse_2,
            ],
        )

    initial_guess = _RiemannParams(
        left_fast_wave_downstream_pressure=left_state.pressure * 1.2,
        left_rotation_angle=0.0,
        pressure_star=0.5 * (left_state.pressure + right_state.pressure),
        right_rotation_angle=0.0,
        right_fast_wave_downstream_pressure=right_state.pressure * 1.2,
    )
    solution = scipy_root(
        compute_contact_residual,
        x0=_as_params_vector(params=initial_guess),
        method="hybr",
    )
    if not solution.success:
        raise RuntimeError(f"riemann-problem root-find did not converge: {solution.message}.")

    region_set = build_regions(_params_from_vector(params_vector=solution.x))
    contact_speed = region_set.left_slow_wave.state.velocity_normal
    return RiemannSolution(
        left_state=left_state,
        left_fast_wave=region_set.left_fast_wave,
        left_rotation_discontinuity=region_set.left_rotation_discontinuity,
        left_slow_wave=region_set.left_slow_wave,
        contact=Wave(
            wave_propagation=WavePropagation(
                wave_type=WaveType.Shock,
                head_speed=contact_speed,
                tail_speed=contact_speed,
            ),
            state=region_set.contact_state,
        ),
        right_slow_wave=region_set.right_slow_wave,
        right_rotation_discontinuity=region_set.right_rotation_discontinuity,
        right_fast_wave=region_set.right_fast_wave,
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
        solution.left_fast_wave,
        solution.left_rotation_discontinuity,
        solution.left_slow_wave,
        solution.contact,
        solution.right_slow_wave,
        solution.right_rotation_discontinuity,
        solution.right_fast_wave,
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
