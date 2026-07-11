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
    upstream: PrimitiveState,
    downstream: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
    shock_speed: float,
) -> ConservedVector:
    """
    Residual of the Rankine-Hugoniot jump conditions across a discontinuity moving
    at `shock_speed`: `flux - shock_speed * conserved` must be continuous across
    any valid discontinuity (shock, contact, or rotational). Zero means `upstream`
    and `downstream` are consistent jump states for that `shock_speed`.
    """
    upstream_conserved = mhd_state.as_conserved(state=upstream, magnetic_field_normal=magnetic_field_normal, gamma=gamma)
    downstream_conserved = mhd_state.as_conserved(state=downstream, magnetic_field_normal=magnetic_field_normal, gamma=gamma)
    upstream_flux = mhd_state.compute_flux(state=upstream, magnetic_field_normal=magnetic_field_normal, gamma=gamma)
    downstream_flux = mhd_state.compute_flux(state=downstream, magnetic_field_normal=magnetic_field_normal, gamma=gamma)
    upstream_term = upstream_flux - shock_speed * upstream_conserved
    downstream_term = downstream_flux - shock_speed * downstream_conserved
    return upstream_term - downstream_term


## } MODULE
