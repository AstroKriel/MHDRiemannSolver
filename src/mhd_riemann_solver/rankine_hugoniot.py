## { MODULE

##
## === DEPENDENCIES
##

## local
from riemann_solver.mhd_state import ConservedVector, PrimitiveState, compute_flux, primitive_to_conserved

##
## === JUMP-CONDITION RESIDUAL
##


def compute_jump_residual(
    *,
    upstream: PrimitiveState,
    downstream: PrimitiveState,
    bx: float,
    gamma: float,
    shock_speed: float,
) -> ConservedVector:
    """
    Residual of the Rankine-Hugoniot jump conditions across a discontinuity moving
    at `shock_speed`: `flux - shock_speed * conserved` must be continuous across
    any valid discontinuity (shock, contact, or rotational). Zero means `upstream`
    and `downstream` are consistent jump states for that `shock_speed`.
    """
    upstream_conserved = primitive_to_conserved(state=upstream, bx=bx, gamma=gamma)
    downstream_conserved = primitive_to_conserved(state=downstream, bx=bx, gamma=gamma)
    upstream_flux = compute_flux(state=upstream, bx=bx, gamma=gamma)
    downstream_flux = compute_flux(state=downstream, bx=bx, gamma=gamma)
    upstream_term = upstream_flux - shock_speed * upstream_conserved
    downstream_term = downstream_flux - shock_speed * downstream_conserved
    return upstream_term - downstream_term


## } MODULE
