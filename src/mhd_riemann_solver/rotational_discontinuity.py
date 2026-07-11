## { MODULE

##
## === DEPENDENCIES
##

## third-party
import numpy

## local
from riemann_solver.mhd_state import PrimitiveState

##
## === ROTATIONAL (ALFVEN) DISCONTINUITY
##


def rotate_transverse_field(
    *,
    by: float,
    bz: float,
    angle: float,
) -> tuple[float, float]:
    """Rotate `(by, bz)` by `angle` radians about the origin, preserving its magnitude."""
    cos_angle = numpy.cos(angle)
    sin_angle = numpy.sin(angle)
    return (
        float(by * cos_angle - bz * sin_angle),
        float(by * sin_angle + bz * cos_angle),
    )


def apply_rotation(
    *,
    upstream: PrimitiveState,
    angle: float,
    sign: float,
) -> PrimitiveState:
    """
    Apply a rotational discontinuity to `upstream`, rotating its transverse field
    by `angle` radians. `sign` selects which of the two characteristic branches
    this discontinuity propagates along, in the lab frame:
    `shock_speed = upstream.u - sign * bx / sqrt(upstream.rho)`.
    Whether a given `sign` is the "left" or "right" rotational discontinuity in a
    particular Riemann fan depends on the sign of `bx` itself; callers building a
    full wave fan must pick `sign` to match the eigenvalue ordering
    `u - |bx|/sqrt(rho) < ... < u + |bx|/sqrt(rho)`, not assume a fixed sign.

    Density, pressure, and normal velocity are preserved (this discontinuity carries
    no density jump); the transverse velocity shifts in lockstep with the transverse
    field, `delta_v_t = sign * delta_b_t / sqrt(rho)`, which is what makes a rotation
    of any angle an exact solution of the jump conditions.
    """
    new_by, new_bz = rotate_transverse_field(by=upstream.by, bz=upstream.bz, angle=angle)
    delta_by = new_by - upstream.by
    delta_bz = new_bz - upstream.bz
    inverse_sqrt_rho = 1.0 / numpy.sqrt(upstream.rho)
    return PrimitiveState(
        rho=upstream.rho,
        u=upstream.u,
        v=upstream.v + sign * delta_by * inverse_sqrt_rho,
        w=upstream.w + sign * delta_bz * inverse_sqrt_rho,
        by=new_by,
        bz=new_bz,
        p=upstream.p,
    )


## } MODULE
