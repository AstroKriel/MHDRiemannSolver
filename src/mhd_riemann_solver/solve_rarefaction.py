## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any, Callable, TypeAlias

## third-party
import numpy
from numpy.typing import NDArray
from scipy.integrate import solve_ivp as scipy_solve_ivp

## local
from riemann_solver import mhd_state
from riemann_solver.mhd_state import PrimitiveState, WaveFamily

##
## === TYPE ALIASES
##

## [density, velocity_normal, velocity_transverse_1, velocity_transverse_2, magnetic_field_transverse_1, magnetic_field_transverse_2, pressure]
PrimitiveVector: TypeAlias = NDArray[Any]

##
## === PRIMITIVE-VARIABLE EIGENSYSTEM (BUILT NUMERICALLY, NOT HAND-DERIVED)
##


def _as_primitive_vector(
    *,
    state: PrimitiveState,
) -> PrimitiveVector:
    """Return `state` flattened to a `PrimitiveVector`."""
    return numpy.array(
        [
            state.density,
            state.velocity_normal,
            state.velocity_transverse_1,
            state.velocity_transverse_2,
            state.magnetic_field_transverse_1,
            state.magnetic_field_transverse_2,
            state.pressure,
        ],
    )


def _state_from_primitive_vector(
    *,
    vector: PrimitiveVector,
) -> PrimitiveState:
    density, velocity_normal, velocity_transverse_1, velocity_transverse_2, magnetic_field_transverse_1, magnetic_field_transverse_2, pressure = (
        vector
    )
    return PrimitiveState(
        density=density,
        velocity_normal=velocity_normal,
        velocity_transverse_1=velocity_transverse_1,
        velocity_transverse_2=velocity_transverse_2,
        magnetic_field_transverse_1=magnetic_field_transverse_1,
        magnetic_field_transverse_2=magnetic_field_transverse_2,
        pressure=pressure,
    )


def _compute_numerical_jacobian(
    *,
    func: Callable[[NDArray[Any]], NDArray[Any]],
    x: NDArray[Any],
    step: float = 1e-6,
) -> NDArray[Any]:
    """Central-difference Jacobian of `func: R^n -> R^n` at `x`."""
    num_components = x.shape[0]
    jacobian = numpy.zeros((num_components, num_components))
    for column in range(num_components):
        perturbation = numpy.zeros(num_components)
        perturbation[column] = step * max(abs(x[column]), 1.0)
        forward = func(x + perturbation)
        backward = func(x - perturbation)
        jacobian[:, column] = (forward - backward) / (2.0 * perturbation[column])
    return jacobian


def _compute_primitive_eigensystem(
    *,
    state: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Return `(eigenvalues, eigenvectors)` of the primitive-variable flux Jacobian
    at `state`, with `eigenvectors[:, k]` the perturbation direction for `eigenvalues[k]`.
    """
    ## built numerically, not hand-derived, to avoid a fresh eigenvector-formula error
    primitive_vector = _as_primitive_vector(state=state)

    def flux_of_vector(
        vector: NDArray[Any],
    ) -> NDArray[Any]:
        return mhd_state.compute_flux(
            state=_state_from_primitive_vector(vector=vector),
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
        )

    def conserved_of_vector(
        vector: NDArray[Any],
    ) -> NDArray[Any]:
        return mhd_state.as_conserved(
            state=_state_from_primitive_vector(vector=vector),
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
        )

    flux_jacobian = _compute_numerical_jacobian(
        func=flux_of_vector,
        x=primitive_vector,
    )
    conserved_jacobian = _compute_numerical_jacobian(
        func=conserved_of_vector,
        x=primitive_vector,
    )
    primitive_flux_jacobian = numpy.linalg.solve(conserved_jacobian, flux_jacobian)
    eigenvalues, eigenvectors = numpy.linalg.eig(primitive_flux_jacobian)
    return numpy.real(eigenvalues), numpy.real(eigenvectors)


def _select_rarefaction_direction(
    *,
    state: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
    target_eigenvalue: float,
) -> NDArray[Any]:
    """
    Return the primitive-space derivative with respect to pressure for the
    characteristic family whose eigenvalue is closest to `target_eigenvalue`,
    normalized so its own pressure component is `1`.
    """
    eigenvalues, eigenvectors = _compute_primitive_eigensystem(
        state=state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    closest_index = int(numpy.argmin(numpy.abs(eigenvalues - target_eigenvalue)))
    direction = eigenvectors[:, closest_index]
    pressure_component = direction[6]
    return direction[:6] / pressure_component


##
## === RAREFACTION SOLVE
##


def solve_rarefaction(
    *,
    upstream_state: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
    pressure_downstream: float,
    wave_family: WaveFamily,
    wave_speed_sign: float,
) -> PrimitiveState:
    """
    Integrate the fast/slow simple-wave ODE from `upstream_state` to `pressure_downstream`;
    density follows the upstream entropy algebraically, since the wave is isentropic.

    Parameters
    ---
    - `wave_speed_sign`:
        `+1.0` for the `velocity_normal + c` branch, `-1.0` for `velocity_normal - c`.
    """
    c_fast, c_slow = mhd_state.compute_fast_slow_speeds(
        state=upstream_state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    reference_speed = c_fast if wave_family == WaveFamily.Fast else c_slow
    target_eigenvalue = upstream_state.velocity_normal + wave_speed_sign * reference_speed
    entropy_constant = upstream_state.pressure / upstream_state.density**gamma

    def rhs(
        pressure: float,
        state_vector: NDArray[Any],
    ) -> NDArray[Any]:
        velocity_normal, velocity_transverse_1, velocity_transverse_2, magnetic_field_transverse_1, magnetic_field_transverse_2 = state_vector
        density = (pressure / entropy_constant)**(1.0 / gamma)
        state = PrimitiveState(
            density=density,
            velocity_normal=velocity_normal,
            velocity_transverse_1=velocity_transverse_1,
            velocity_transverse_2=velocity_transverse_2,
            magnetic_field_transverse_1=magnetic_field_transverse_1,
            magnetic_field_transverse_2=magnetic_field_transverse_2,
            pressure=pressure,
        )
        direction = _select_rarefaction_direction(
            state=state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            target_eigenvalue=target_eigenvalue,
        )
        return direction[1:]

    initial_vector = numpy.array(
        [
            upstream_state.velocity_normal,
            upstream_state.velocity_transverse_1,
            upstream_state.velocity_transverse_2,
            upstream_state.magnetic_field_transverse_1,
            upstream_state.magnetic_field_transverse_2,
        ],
    )
    solution = scipy_solve_ivp(
        rhs,
        (upstream_state.pressure, pressure_downstream),
        initial_vector,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"rarefaction ode integration failed: {solution.message}.")
    velocity_normal, velocity_transverse_1, velocity_transverse_2, magnetic_field_transverse_1, magnetic_field_transverse_2 = solution.y[:,
                                                                                                                                         -1
                                                                                                                                         ]
    density_downstream = (pressure_downstream / entropy_constant)**(1.0 / gamma)
    return PrimitiveState(
        density=density_downstream,
        velocity_normal=velocity_normal,
        velocity_transverse_1=velocity_transverse_1,
        velocity_transverse_2=velocity_transverse_2,
        magnetic_field_transverse_1=magnetic_field_transverse_1,
        magnetic_field_transverse_2=magnetic_field_transverse_2,
        pressure=pressure_downstream,
    )


## } MODULE
