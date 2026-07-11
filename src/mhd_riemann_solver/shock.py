## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray
from scipy.optimize import root as scipy_root

## local
from riemann_solver.mhd_state import PrimitiveState
from riemann_solver.rankine_hugoniot import compute_jump_residual

##
## === DOWNSTREAM STATE FROM A MASS FLUX
##


def _build_shock_downstream_candidate(
    *,
    upstream: PrimitiveState,
    bx: float,
    mass_flux: float,
    rho_downstream: float,
    p_downstream: float,
) -> tuple[PrimitiveState, float]:
    """
    Build the candidate downstream state and shock speed implied by a mass flux
    and downstream density, using the algebraic tangential-momentum and
    tangential-induction jump relations, which are exact for any `(mass_flux,
    rho_downstream)` pair. Normal-momentum and energy are not enforced here;
    `solve_shock` root-finds on `(mass_flux, rho_downstream)` until they are.
    """
    field_ratio = (mass_flux**2 / upstream.rho - bx**2) / (mass_flux**2 / rho_downstream - bx**2)
    by_downstream = upstream.by * field_ratio
    bz_downstream = upstream.bz * field_ratio
    v_downstream = upstream.v + bx * (by_downstream - upstream.by) / mass_flux
    w_downstream = upstream.w + bx * (bz_downstream - upstream.bz) / mass_flux
    normal_speed_upstream = mass_flux / upstream.rho
    normal_speed_downstream = mass_flux / rho_downstream
    shock_speed = upstream.u - normal_speed_upstream
    u_downstream = shock_speed + normal_speed_downstream
    downstream = PrimitiveState(
        rho=rho_downstream,
        u=u_downstream,
        v=v_downstream,
        w=w_downstream,
        by=by_downstream,
        bz=bz_downstream,
        p=p_downstream,
    )
    return downstream, shock_speed


##
## === SHOCK SOLVE
##


def solve_shock(
    *,
    upstream: PrimitiveState,
    bx: float,
    gamma: float,
    p_downstream: float,
    initial_relative_speed_guess: float,
) -> tuple[PrimitiveState, float]:
    """
    Solve the MHD shock jump conditions for the downstream state given `upstream`
    and a target downstream pressure `p_downstream`, by root-finding the mass flux
    and downstream density that satisfy normal-momentum and energy conservation.

    `initial_relative_speed_guess` seeds the search: the upstream normal speed
    relative to the shock, `u - shock_speed`. Use `+c_fast`/`+c_slow` when
    `upstream` sits on the left of this wave in the Riemann fan (e.g. the L state
    feeding the leftmost wave), `-c_fast`/`-c_slow` when it sits on the right (e.g.
    the R state feeding the rightmost wave). Its sign fixes which physical shock
    branch the root-find converges to.
    """
    initial_mass_flux = upstream.rho * initial_relative_speed_guess
    initial_rho_downstream = upstream.rho * (p_downstream / upstream.p) ** (1.0 / gamma)

    def residuals(unknowns: NDArray[Any]) -> NDArray[Any]:
        mass_flux, rho_downstream = unknowns
        downstream, shock_speed = _build_shock_downstream_candidate(
            upstream=upstream,
            bx=bx,
            mass_flux=mass_flux,
            rho_downstream=rho_downstream,
            p_downstream=p_downstream,
        )
        full_residual = compute_jump_residual(
            upstream=upstream,
            downstream=downstream,
            bx=bx,
            gamma=gamma,
            shock_speed=shock_speed,
        )
        return numpy.array([full_residual[1], full_residual[6]])

    solution = scipy_root(
        residuals,
        x0=numpy.array([initial_mass_flux, initial_rho_downstream]),
        method="hybr",
    )
    if not solution.success:
        raise RuntimeError(f"shock jump-condition root-find did not converge: {solution.message}.")
    mass_flux, rho_downstream = solution.x
    return _build_shock_downstream_candidate(
        upstream=upstream,
        bx=bx,
        mass_flux=mass_flux,
        rho_downstream=rho_downstream,
        p_downstream=p_downstream,
    )


## } MODULE
