## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from typing import Any, Self, TypeAlias

## third-party
import numpy
from numpy.typing import NDArray
from scipy.optimize import root as scipy_root

## local
from aegir import rankine_hugoniot
from aegir.mhd_state import PrimitiveState

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
## === SHOCK PARAMETERS
##

## flattened, positionally-fixed encoding of a `_ShockParams`, for `scipy.optimize.root`
_ShockParamsVector: TypeAlias = NDArray[Any]


@dataclass(frozen=True)
class _ShockParams:
    """The 2 free parameters root-found to satisfy the shock jump conditions."""

    mass_flux: float
    density_downstream: float

    @classmethod
    def from_params_vector(
        cls,
        *,
        params_vector: _ShockParamsVector,
    ) -> Self:
        return cls(
            mass_flux=params_vector[0],
            density_downstream=params_vector[1],
        )


def _as_params_vector(
    *,
    shock_params: _ShockParams,
) -> _ShockParamsVector:
    """Return `shock_params` flattened to the vector `scipy.optimize.root` operates on."""
    return numpy.array([
        shock_params.mass_flux,
        shock_params.density_downstream,
    ])


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
    initial_guess = _ShockParams(
        mass_flux=initial_mass_flux,
        density_downstream=initial_density_downstream,
    )

    def compute_shock_jump_residual(
        params_vector: _ShockParamsVector,
    ) -> NDArray[Any]:
        shock_params = _ShockParams.from_params_vector(params_vector=params_vector)
        downstream_state, shock_speed = _build_shock_downstream_candidate(
            upstream_state=upstream_state,
            magnetic_field_normal=magnetic_field_normal,
            mass_flux=shock_params.mass_flux,
            density_downstream=shock_params.density_downstream,
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

    shock_jump_root = scipy_root(
        compute_shock_jump_residual,
        x0=_as_params_vector(shock_params=initial_guess),
        method="hybr",
    )
    if not shock_jump_root.success:
        raise RuntimeError(f"shock jump-condition root-find did not converge: {shock_jump_root.message}.")
    shock_params = _ShockParams.from_params_vector(params_vector=shock_jump_root.x)
    return _build_shock_downstream_candidate(
        upstream_state=upstream_state,
        magnetic_field_normal=magnetic_field_normal,
        mass_flux=shock_params.mass_flux,
        density_downstream=shock_params.density_downstream,
        pressure_downstream=pressure_downstream,
    )


## } MODULE
