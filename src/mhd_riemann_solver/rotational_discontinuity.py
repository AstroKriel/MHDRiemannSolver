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
    magnetic_field_transverse_1: float,
    magnetic_field_transverse_2: float,
    angle: float,
) -> tuple[float, float]:
    """Rotate `(magnetic_field_transverse_1, magnetic_field_transverse_2)` by `angle` radians about the origin, preserving its magnitude."""
    cos_angle = numpy.cos(angle)
    sin_angle = numpy.sin(angle)
    return (
        float(magnetic_field_transverse_1 * cos_angle - magnetic_field_transverse_2 * sin_angle),
        float(magnetic_field_transverse_1 * sin_angle + magnetic_field_transverse_2 * cos_angle),
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
    `shock_speed = upstream.velocity_normal - sign * magnetic_field_normal / sqrt(upstream.density)`.
    Whether a given `sign` is the "left" or "right" rotational discontinuity in a
    particular Riemann fan depends on the sign of `magnetic_field_normal` itself;
    callers building a full wave fan must pick `sign` to match the eigenvalue
    ordering `u - |bx|/sqrt(rho) < ... < u + |bx|/sqrt(rho)`, not assume a fixed sign.

    Density, pressure, and normal velocity are preserved (this discontinuity carries
    no density jump); the transverse velocity shifts in lockstep with the transverse
    field, `delta_v_t = sign * delta_b_t / sqrt(rho)`, which is what makes a rotation
    of any angle an exact solution of the jump conditions.
    """
    new_magnetic_field_transverse_1, new_magnetic_field_transverse_2 = rotate_transverse_field(
        magnetic_field_transverse_1=upstream.magnetic_field_transverse_1,
        magnetic_field_transverse_2=upstream.magnetic_field_transverse_2,
        angle=angle,
    )
    delta_magnetic_field_transverse_1 = new_magnetic_field_transverse_1 - upstream.magnetic_field_transverse_1
    delta_magnetic_field_transverse_2 = new_magnetic_field_transverse_2 - upstream.magnetic_field_transverse_2
    inverse_sqrt_density = 1.0 / numpy.sqrt(upstream.density)
    return PrimitiveState(
        density=upstream.density,
        velocity_normal=upstream.velocity_normal,
        velocity_transverse_1=upstream.velocity_transverse_1 + sign * delta_magnetic_field_transverse_1 * inverse_sqrt_density,
        velocity_transverse_2=upstream.velocity_transverse_2 + sign * delta_magnetic_field_transverse_2 * inverse_sqrt_density,
        magnetic_field_transverse_1=new_magnetic_field_transverse_1,
        magnetic_field_transverse_2=new_magnetic_field_transverse_2,
        pressure=upstream.pressure,
    )


## } MODULE
