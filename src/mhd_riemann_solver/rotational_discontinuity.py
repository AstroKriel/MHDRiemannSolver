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
    upstream_state: PrimitiveState,
    angle: float,
    sign: float,
) -> PrimitiveState:
    """
    Apply a rotational discontinuity to `upstream_state`, rotating its transverse field
    by `angle` radians; density, pressure, and normal velocity are preserved.

    `sign` selects which characteristic branch this propagates along; whether a
    given `sign` is the "left" or "right" discontinuity in a wave fan depends on
    the sign of `magnetic_field_normal`, not a fixed convention.
    """
    new_magnetic_field_transverse_1, new_magnetic_field_transverse_2 = rotate_transverse_field(
        magnetic_field_transverse_1=upstream_state.magnetic_field_transverse_1,
        magnetic_field_transverse_2=upstream_state.magnetic_field_transverse_2,
        angle=angle,
    )
    delta_magnetic_field_transverse_1 = new_magnetic_field_transverse_1 - upstream_state.magnetic_field_transverse_1
    delta_magnetic_field_transverse_2 = new_magnetic_field_transverse_2 - upstream_state.magnetic_field_transverse_2
    inverse_sqrt_density = 1.0 / numpy.sqrt(upstream_state.density)
    return PrimitiveState(
        density=upstream_state.density,
        velocity_normal=upstream_state.velocity_normal,
        velocity_transverse_1=upstream_state.velocity_transverse_1 +
        sign * delta_magnetic_field_transverse_1 * inverse_sqrt_density,
        velocity_transverse_2=upstream_state.velocity_transverse_2 +
        sign * delta_magnetic_field_transverse_2 * inverse_sqrt_density,
        magnetic_field_transverse_1=new_magnetic_field_transverse_1,
        magnetic_field_transverse_2=new_magnetic_field_transverse_2,
        pressure=upstream_state.pressure,
    )


## } MODULE
