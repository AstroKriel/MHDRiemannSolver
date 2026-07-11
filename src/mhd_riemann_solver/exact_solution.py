## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray
from scipy.optimize import root as scipy_root

## local
from riemann_solver.mhd_state import PrimitiveState, compute_fast_slow_speeds
from riemann_solver.rarefaction import solve_rarefaction
from riemann_solver.rotational_discontinuity import apply_rotation
from riemann_solver.shock import solve_shock

##
## === WAVE INFO
##


@dataclass(frozen=True)
class WaveInfo:
    """
    Propagation speed of one wave in the fan. `head_speed == tail_speed` for a
    shock, rotational discontinuity, or contact (all sharp); a rarefaction spans
    `[head_speed, tail_speed]`.
    """

    kind: str  # "shock" or "rarefaction"
    head_speed: float
    tail_speed: float


##
## === WAVE DISPATCH
##


def _solve_wave(
    *,
    upstream: PrimitiveState,
    bx: float,
    gamma: float,
    p_downstream: float,
    wave_family: str,
    wave_speed_sign: float,
) -> tuple[PrimitiveState, WaveInfo]:
    """Solve one fast/slow wave, dispatching to a shock or a rarefaction by the sign of `p_downstream - upstream.p`."""
    c_fast_up, c_slow_up = compute_fast_slow_speeds(state=upstream, bx=bx, gamma=gamma)
    reference_speed_up = c_fast_up if wave_family == "fast" else c_slow_up
    if p_downstream > upstream.p:
        downstream, shock_speed = solve_shock(
            upstream=upstream,
            bx=bx,
            gamma=gamma,
            p_downstream=p_downstream,
            initial_relative_speed_guess=-wave_speed_sign * reference_speed_up,
        )
        return downstream, WaveInfo(kind="shock", head_speed=shock_speed, tail_speed=shock_speed)
    downstream = solve_rarefaction(
        upstream=upstream,
        bx=bx,
        gamma=gamma,
        p_downstream=p_downstream,
        wave_family=wave_family,
        wave_speed_sign=wave_speed_sign,
    )
    c_fast_down, c_slow_down = compute_fast_slow_speeds(state=downstream, bx=bx, gamma=gamma)
    reference_speed_down = c_fast_down if wave_family == "fast" else c_slow_down
    head_speed = upstream.u + wave_speed_sign * reference_speed_up
    tail_speed = downstream.u + wave_speed_sign * reference_speed_down
    return downstream, WaveInfo(kind="rarefaction", head_speed=min(head_speed, tail_speed), tail_speed=max(head_speed, tail_speed))


def _rotation_wave_info(
    *,
    upstream: PrimitiveState,
    bx: float,
    sign: float,
) -> WaveInfo:
    speed = upstream.u - sign * bx / numpy.sqrt(upstream.rho)
    return WaveInfo(kind="shock", head_speed=speed, tail_speed=speed)


##
## === RIEMANN SOLUTION
##


@dataclass(frozen=True)
class RiemannSolution:
    """The 8 constant states of a fully-resolved 1D ideal-MHD Riemann fan, and each wave's propagation speed."""

    region1: PrimitiveState  # given left state
    region2: PrimitiveState  # after the left fast wave
    region3: PrimitiveState  # after the left rotational discontinuity
    region4: PrimitiveState  # after the left slow wave (left of contact)
    region5: PrimitiveState  # after the right slow wave (right of contact)
    region6: PrimitiveState  # after the right rotational discontinuity
    region7: PrimitiveState  # after the right fast wave
    region8: PrimitiveState  # given right state
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
    left: PrimitiveState,
    right: PrimitiveState,
    bx: float,
    gamma: float,
) -> RiemannSolution:
    """
    Solve the full 7-wave 1D ideal-MHD Riemann problem by root-finding the 5
    unknowns that pin down the fan: `p2`/`p7` (post-fast-wave pressures on the
    left/right), `p_star` (the shared pressure at the contact), and `psi_left`/
    `psi_right` (the two rotational-discontinuity rotation angles). Whether each
    fast/slow wave turns out to be a shock or a rarefaction falls out of the
    solve; it is not assumed up front (see `_solve_wave`).

    The two rotational-discontinuity `sign`s (see `rotational_discontinuity.
    apply_rotation`) are fixed by the eigenvalue ordering `u - |bx|/sqrt(rho) <
    ... < u + |bx|/sqrt(rho)`: the left one always carries `sign(bx)`, the right
    one `-sign(bx)`.
    """
    rotation_sign_left = 1.0 if bx >= 0.0 else -1.0
    rotation_sign_right = -rotation_sign_left

    def build_regions(
        unknowns: NDArray[Any],
    ) -> tuple[
        PrimitiveState,
        PrimitiveState,
        PrimitiveState,
        PrimitiveState,
        PrimitiveState,
        PrimitiveState,
        WaveInfo,
        WaveInfo,
        WaveInfo,
        WaveInfo,
        WaveInfo,
        WaveInfo,
    ]:
        p2, psi_left, p_star, psi_right, p7 = unknowns
        region2, fast_left = _solve_wave(upstream=left, bx=bx, gamma=gamma, p_downstream=p2, wave_family="fast", wave_speed_sign=-1.0)
        region3 = apply_rotation(upstream=region2, angle=psi_left, sign=rotation_sign_left)
        rotation_left = _rotation_wave_info(upstream=region2, bx=bx, sign=rotation_sign_left)
        region4, slow_left = _solve_wave(upstream=region3, bx=bx, gamma=gamma, p_downstream=p_star, wave_family="slow", wave_speed_sign=-1.0)
        region7, fast_right = _solve_wave(upstream=right, bx=bx, gamma=gamma, p_downstream=p7, wave_family="fast", wave_speed_sign=1.0)
        region6 = apply_rotation(upstream=region7, angle=psi_right, sign=rotation_sign_right)
        rotation_right = _rotation_wave_info(upstream=region7, bx=bx, sign=rotation_sign_right)
        region5, slow_right = _solve_wave(upstream=region6, bx=bx, gamma=gamma, p_downstream=p_star, wave_family="slow", wave_speed_sign=1.0)
        return region2, region3, region4, region5, region6, region7, fast_left, rotation_left, slow_left, slow_right, rotation_right, fast_right

    def residuals(unknowns: NDArray[Any]) -> NDArray[Any]:
        _, _, region4, region5, _, _, _, _, _, _, _, _ = build_regions(unknowns)
        return numpy.array(
            [
                region4.u - region5.u,
                region4.v - region5.v,
                region4.w - region5.w,
                region4.by - region5.by,
                region4.bz - region5.bz,
            ],
        )

    initial_guess = numpy.array([left.p * 1.2, 0.0, 0.5 * (left.p + right.p), 0.0, right.p * 1.2])
    solution = scipy_root(residuals, x0=initial_guess, method="hybr")
    if not solution.success:
        raise RuntimeError(f"riemann-problem root-find did not converge: {solution.message}.")

    region2, region3, region4, region5, region6, region7, fast_left, rotation_left, slow_left, slow_right, rotation_right, fast_right = (
        build_regions(solution.x)
    )
    contact_speed = region4.u
    return RiemannSolution(
        region1=left,
        region2=region2,
        region3=region3,
        region4=region4,
        region5=region5,
        region6=region6,
        region7=region7,
        region8=right,
        fast_left=fast_left,
        rotation_left=rotation_left,
        slow_left=slow_left,
        contact=WaveInfo(kind="shock", head_speed=contact_speed, tail_speed=contact_speed),
        slow_right=slow_right,
        rotation_right=rotation_right,
        fast_right=fast_right,
    )


##
## === PROFILE EVALUATION
##


def evaluate(
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
    rarefaction fan: this solver resolves the 8 constant states and each wave's
    head/tail speed exactly, but does not yet interpolate the smooth profile
    inside a fan (not needed for RJ2a, whose 4 fast/slow waves are all shocks
    for the standard left/right states).
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
            if wave.kind == "rarefaction" and self_similar_speed < wave.tail_speed:
                raise NotImplementedError(
                    f"position `{position}` at t=`{t}` falls inside a rarefaction fan; "
                    "fan-interior sampling is not yet supported.",
                )
            state = right_state
        profile.append(state)
    return profile


## } MODULE
