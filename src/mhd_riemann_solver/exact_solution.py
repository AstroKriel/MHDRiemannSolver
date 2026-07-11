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
) -> PrimitiveState:
    """Solve one fast/slow wave, dispatching to a shock or a rarefaction by the sign of `p_downstream - upstream.p`."""
    if p_downstream > upstream.p:
        c_fast, c_slow = compute_fast_slow_speeds(state=upstream, bx=bx, gamma=gamma)
        reference_speed = c_fast if wave_family == "fast" else c_slow
        downstream, _ = solve_shock(
            upstream=upstream,
            bx=bx,
            gamma=gamma,
            p_downstream=p_downstream,
            initial_relative_speed_guess=-wave_speed_sign * reference_speed,
        )
        return downstream
    return solve_rarefaction(
        upstream=upstream,
        bx=bx,
        gamma=gamma,
        p_downstream=p_downstream,
        wave_family=wave_family,
        wave_speed_sign=wave_speed_sign,
    )


##
## === RIEMANN SOLUTION
##


@dataclass(frozen=True)
class RiemannSolution:
    """The 8 constant states of a fully-resolved 1D ideal-MHD Riemann fan."""

    region1: PrimitiveState  # given left state
    region2: PrimitiveState  # after the left fast wave
    region3: PrimitiveState  # after the left rotational discontinuity
    region4: PrimitiveState  # after the left slow wave (left of contact)
    region5: PrimitiveState  # after the right slow wave (right of contact)
    region6: PrimitiveState  # after the right rotational discontinuity
    region7: PrimitiveState  # after the right fast wave
    region8: PrimitiveState  # given right state


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
    ) -> tuple[PrimitiveState, PrimitiveState, PrimitiveState, PrimitiveState, PrimitiveState, PrimitiveState]:
        p2, psi_left, p_star, psi_right, p7 = unknowns
        region2 = _solve_wave(upstream=left, bx=bx, gamma=gamma, p_downstream=p2, wave_family="fast", wave_speed_sign=-1.0)
        region3 = apply_rotation(upstream=region2, angle=psi_left, sign=rotation_sign_left)
        region4 = _solve_wave(upstream=region3, bx=bx, gamma=gamma, p_downstream=p_star, wave_family="slow", wave_speed_sign=-1.0)
        region7 = _solve_wave(upstream=right, bx=bx, gamma=gamma, p_downstream=p7, wave_family="fast", wave_speed_sign=1.0)
        region6 = apply_rotation(upstream=region7, angle=psi_right, sign=rotation_sign_right)
        region5 = _solve_wave(upstream=region6, bx=bx, gamma=gamma, p_downstream=p_star, wave_family="slow", wave_speed_sign=1.0)
        return region2, region3, region4, region5, region6, region7

    def residuals(unknowns: NDArray[Any]) -> NDArray[Any]:
        _, _, region4, region5, _, _ = build_regions(unknowns)
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

    region2, region3, region4, region5, region6, region7 = build_regions(solution.x)
    return RiemannSolution(
        region1=left,
        region2=region2,
        region3=region3,
        region4=region4,
        region5=region5,
        region6=region6,
        region7=region7,
        region8=right,
    )


## } MODULE
