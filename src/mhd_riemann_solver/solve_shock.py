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
from riemann_solver import rankine_hugoniot
from riemann_solver.mhd_state import PrimitiveState

##
## === DOWNSTREAM STATE FROM A MASS FLUX
##


def _build_shock_downstream_candidate(
    *,
    upstream_state: PrimitiveState,
    magnetic_field_normal: float,
    mass_flux: float,
    density_downstream: float,
    pressure_downstream: float,
) -> tuple[PrimitiveState, float]:
    """
    Build the candidate downstream state and shock speed implied by a mass flux
    and downstream density, using the algebraic tangential-momentum and
    tangential-induction jump relations, which are exact for any `(mass_flux,
    density_downstream)` pair. Normal-momentum and energy are not enforced here;
    `solve_shock` root-finds on `(mass_flux, density_downstream)` until they are.
    """
    field_ratio = (mass_flux**2 / upstream_state.density -
                   magnetic_field_normal**2) / (mass_flux**2 / density_downstream - magnetic_field_normal**2)
    magnetic_field_transverse_1_downstream = upstream_state.magnetic_field_transverse_1 * field_ratio
    magnetic_field_transverse_2_downstream = upstream_state.magnetic_field_transverse_2 * field_ratio
    velocity_transverse_1_downstream = (
        upstream_state.velocity_transverse_1 + magnetic_field_normal *
        (magnetic_field_transverse_1_downstream - upstream_state.magnetic_field_transverse_1) / mass_flux
    )
    velocity_transverse_2_downstream = (
        upstream_state.velocity_transverse_2 + magnetic_field_normal *
        (magnetic_field_transverse_2_downstream - upstream_state.magnetic_field_transverse_2) / mass_flux
    )
    normal_speed_upstream = mass_flux / upstream_state.density
    normal_speed_downstream = mass_flux / density_downstream
    shock_speed = upstream_state.velocity_normal - normal_speed_upstream
    velocity_normal_downstream = shock_speed + normal_speed_downstream
    downstream_state = PrimitiveState(
        density=density_downstream,
        velocity_normal=velocity_normal_downstream,
        velocity_transverse_1=velocity_transverse_1_downstream,
        velocity_transverse_2=velocity_transverse_2_downstream,
        magnetic_field_transverse_1=magnetic_field_transverse_1_downstream,
        magnetic_field_transverse_2=magnetic_field_transverse_2_downstream,
        pressure=pressure_downstream,
    )
    return downstream_state, shock_speed


##
## === SHOCK SOLVE
##


def solve_shock(
    *,
    upstream_state: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
    pressure_downstream: float,
    initial_relative_speed_guess: float,
) -> tuple[PrimitiveState, float]:
    """
    Solve the MHD shock jump conditions for the downstream state, root-finding
    the mass flux and downstream density that satisfy normal-momentum and
    energy conservation.

    Parameters
    ---
    - `initial_relative_speed_guess`:
        Seeds the search: the upstream normal speed relative to the shock. Use
        `+c_fast`/`+c_slow` if `upstream_state` is the left state of this wave,
        `-c_fast`/`-c_slow` if it's the right state; the sign fixes which shock
        branch is found.
    """
    initial_mass_flux = upstream_state.density * initial_relative_speed_guess
    initial_density_downstream = upstream_state.density * (pressure_downstream /
                                                           upstream_state.pressure)**(1.0 / gamma)

    def residuals(
        unknowns: NDArray[Any],
    ) -> NDArray[Any]:
        mass_flux, density_downstream = unknowns
        downstream_state, shock_speed = _build_shock_downstream_candidate(
            upstream_state=upstream_state,
            magnetic_field_normal=magnetic_field_normal,
            mass_flux=mass_flux,
            density_downstream=density_downstream,
            pressure_downstream=pressure_downstream,
        )
        full_residual = rankine_hugoniot.compute_jump_residual(
            upstream_state=upstream_state,
            downstream_state=downstream_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            shock_speed=shock_speed,
        )
        return numpy.array([full_residual[1], full_residual[6]])

    solution = scipy_root(
        residuals,
        x0=numpy.array([initial_mass_flux, initial_density_downstream]),
        method="hybr",
    )
    if not solution.success:
        raise RuntimeError(f"shock jump-condition root-find did not converge: {solution.message}.")
    mass_flux, density_downstream = solution.x
    return _build_shock_downstream_candidate(
        upstream_state=upstream_state,
        magnetic_field_normal=magnetic_field_normal,
        mass_flux=mass_flux,
        density_downstream=density_downstream,
        pressure_downstream=pressure_downstream,
    )


## } MODULE
