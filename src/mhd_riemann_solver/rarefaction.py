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
from riemann_solver.mhd_state import PrimitiveState, compute_fast_slow_speeds, compute_flux, primitive_to_conserved

##
## === TYPE ALIASES
##

PrimitiveVector: TypeAlias = NDArray[Any]  # [rho, u, v, w, by, bz, p]

##
## === PRIMITIVE-VARIABLE EIGENSYSTEM (BUILT NUMERICALLY, NOT HAND-DERIVED)
##


def _primitive_vector(
    *,
    state: PrimitiveState,
) -> PrimitiveVector:
    return numpy.array([state.rho, state.u, state.v, state.w, state.by, state.bz, state.p])


def _state_from_primitive_vector(
    *,
    vector: PrimitiveVector,
) -> PrimitiveState:
    rho, u, v, w, by, bz, p = vector
    return PrimitiveState(rho=rho, u=u, v=v, w=w, by=by, bz=bz, p=p)


def _numerical_jacobian(
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
    bx: float,
    gamma: float,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Return `(eigenvalues, eigenvectors)` of the primitive-variable flux Jacobian
    at `state`, with `eigenvectors[:, k]` the primitive-space perturbation
    direction `(rho, u, v, w, by, bz, p)` for `eigenvalues[k]`.

    Built from central-difference Jacobians of `compute_flux` and
    `primitive_to_conserved` (both already covered by the shock/rotation tests)
    rather than a hand-transcribed analytic eigenvector formula, via the
    similarity transform `A_primitive = (dU/dW)^-1 @ (dF/dW)`, which shares
    `A_conserved`'s eigenvalues by construction.
    """
    primitive_vector = _primitive_vector(state=state)

    def flux_of_vector(vector: NDArray[Any]) -> NDArray[Any]:
        return compute_flux(state=_state_from_primitive_vector(vector=vector), bx=bx, gamma=gamma)

    def conserved_of_vector(vector: NDArray[Any]) -> NDArray[Any]:
        return primitive_to_conserved(state=_state_from_primitive_vector(vector=vector), bx=bx, gamma=gamma)

    flux_jacobian = _numerical_jacobian(func=flux_of_vector, x=primitive_vector)
    conserved_jacobian = _numerical_jacobian(func=conserved_of_vector, x=primitive_vector)
    primitive_flux_jacobian = numpy.linalg.solve(conserved_jacobian, flux_jacobian)
    eigenvalues, eigenvectors = numpy.linalg.eig(primitive_flux_jacobian)
    return numpy.real(eigenvalues), numpy.real(eigenvectors)


def _select_rarefaction_direction(
    *,
    state: PrimitiveState,
    bx: float,
    gamma: float,
    target_eigenvalue: float,
) -> NDArray[Any]:
    """
    Return `d(rho, u, v, w, by, bz)/dp` for the characteristic family whose
    eigenvalue is closest to `target_eigenvalue`, normalized so its own `dp`
    component is `1`.
    """
    eigenvalues, eigenvectors = _compute_primitive_eigensystem(state=state, bx=bx, gamma=gamma)
    closest_index = int(numpy.argmin(numpy.abs(eigenvalues - target_eigenvalue)))
    direction = eigenvectors[:, closest_index]
    pressure_component = direction[6]
    return direction[:6] / pressure_component


##
## === RAREFACTION SOLVE
##


def solve_rarefaction(
    *,
    upstream: PrimitiveState,
    bx: float,
    gamma: float,
    p_downstream: float,
    wave_family: str,
    wave_speed_sign: float,
) -> PrimitiveState:
    """
    Integrate the fast/slow simple-wave ODE from `upstream` to `p_downstream`,
    tracing the characteristic family nearest `upstream`'s fast/slow eigenvalue
    (`wave_family` in `{"fast", "slow"}`, `wave_speed_sign` `+1.0`/`-1.0` for the
    `u + c`/`u - c` branch) throughout the integration.

    The wave is isentropic, so density follows `p_upstream / rho_upstream**gamma`
    algebraically; only `(u, v, w, by, bz)` are integrated in `p`.
    """
    c_fast, c_slow = compute_fast_slow_speeds(state=upstream, bx=bx, gamma=gamma)
    reference_speed = c_fast if wave_family == "fast" else c_slow
    target_eigenvalue = upstream.u + wave_speed_sign * reference_speed
    entropy_constant = upstream.p / upstream.rho**gamma

    def rhs(pressure: float, state_vector: NDArray[Any]) -> NDArray[Any]:
        u, v, w, by, bz = state_vector
        rho = (pressure / entropy_constant) ** (1.0 / gamma)
        state = PrimitiveState(rho=rho, u=u, v=v, w=w, by=by, bz=bz, p=pressure)
        direction = _select_rarefaction_direction(state=state, bx=bx, gamma=gamma, target_eigenvalue=target_eigenvalue)
        return direction[1:]

    initial_vector = numpy.array([upstream.u, upstream.v, upstream.w, upstream.by, upstream.bz])
    solution = scipy_solve_ivp(
        rhs,
        (upstream.p, p_downstream),
        initial_vector,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"rarefaction ode integration failed: {solution.message}.")
    u, v, w, by, bz = solution.y[:, -1]
    rho_downstream = (p_downstream / entropy_constant) ** (1.0 / gamma)
    return PrimitiveState(rho=rho_downstream, u=u, v=v, w=w, by=by, bz=bz, p=p_downstream)


## } MODULE
