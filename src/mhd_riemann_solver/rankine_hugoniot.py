## { MODULE

##
## === DEPENDENCIES
##

## local
from riemann_solver import mhd_state
from riemann_solver.mhd_state import ConservedVector, PrimitiveState

##
## === JUMP-CONDITION RESIDUAL
##


def compute_jump_residual(
    *,
    upstream_state: PrimitiveState,
    downstream_state: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
    shock_speed: float,
) -> ConservedVector:
    """
    Residual of the Rankine-Hugoniot jump conditions across a discontinuity moving
    at `shock_speed`: `flux - shock_speed * conserved_vector` must be continuous
    across any valid discontinuity (shock, contact, or rotational); zero means
    `upstream_state` and `downstream_state` are consistent jump states for that `shock_speed`.
    """
    upstream_conserved_vector = mhd_state.as_conserved(
        state=upstream_state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    downstream_conserved_vector = mhd_state.as_conserved(
        state=downstream_state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    upstream_flux = mhd_state.compute_flux(
        state=upstream_state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    downstream_flux = mhd_state.compute_flux(
        state=downstream_state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    upstream_term = upstream_flux - shock_speed * upstream_conserved_vector
    downstream_term = downstream_flux - shock_speed * downstream_conserved_vector
    return upstream_term - downstream_term


## } MODULE
