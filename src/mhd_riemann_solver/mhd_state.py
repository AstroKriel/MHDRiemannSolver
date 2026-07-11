## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from typing import Any, TypeAlias

## third-party
import numpy
from numpy.typing import NDArray

##
## === TYPE ALIASES
##

## conserved vector: [rho, rho*u, rho*v, rho*w, by, bz, energy]
ConservedVector: TypeAlias = NDArray[Any]

##
## === PRIMITIVE STATE
##


@dataclass(frozen=True)
class PrimitiveState:
    """
    A pointwise ideal-MHD primitive state; `u` is the direction-normal velocity,
    `v`/`w` and `by`/`bz` are the two transverse components.

    `bx` is not stored here: it is one constant shared by the whole Riemann
    problem, not a per-region unknown, so callers thread it through separately.
    """

    rho: float
    u: float
    v: float
    w: float
    by: float
    bz: float
    p: float

    @property
    def bt(self) -> float:
        """Magnitude of the transverse field (by, bz)."""
        return float(numpy.hypot(self.by, self.bz))


##
## === WAVE SPEEDS
##


def compute_sound_speed(
    *,
    state: PrimitiveState,
    gamma: float,
) -> float:
    return float(numpy.sqrt(gamma * state.p / state.rho))


def compute_alfven_speed(
    *,
    state: PrimitiveState,
    bx: float,
) -> float:
    return float(numpy.abs(bx) / numpy.sqrt(state.rho))


def compute_fast_slow_speeds(
    *,
    state: PrimitiveState,
    bx: float,
    gamma: float,
) -> tuple[float, float]:
    """
    Return `(c_fast, c_slow)`, the fast/slow magnetosonic speeds normal to `bx`.
    """
    sound_speed_sq = gamma * state.p / state.rho
    total_alfven_speed_sq = (bx**2 + state.by**2 + state.bz**2) / state.rho
    normal_alfven_speed_sq = bx**2 / state.rho
    discriminant = (sound_speed_sq + total_alfven_speed_sq) ** 2 - 4.0 * sound_speed_sq * normal_alfven_speed_sq
    root = numpy.sqrt(max(discriminant, 0.0))
    c_fast = numpy.sqrt(0.5 * (sound_speed_sq + total_alfven_speed_sq + root))
    c_slow = numpy.sqrt(max(0.5 * (sound_speed_sq + total_alfven_speed_sq - root), 0.0))
    return float(c_fast), float(c_slow)


##
## === CONSERVED, PRIMITIVE, AND FLUX CONVERSIONS
##


def compute_energy(
    *,
    state: PrimitiveState,
    bx: float,
    gamma: float,
) -> float:
    kinetic_energy = 0.5 * state.rho * (state.u**2 + state.v**2 + state.w**2)
    magnetic_energy = 0.5 * (bx**2 + state.by**2 + state.bz**2)
    internal_energy = state.p / (gamma - 1.0)
    return internal_energy + kinetic_energy + magnetic_energy


def primitive_to_conserved(
    *,
    state: PrimitiveState,
    bx: float,
    gamma: float,
) -> ConservedVector:
    return numpy.array(
        [
            state.rho,
            state.rho * state.u,
            state.rho * state.v,
            state.rho * state.w,
            state.by,
            state.bz,
            compute_energy(state=state, bx=bx, gamma=gamma),
        ],
    )


def conserved_to_primitive(
    *,
    conserved: ConservedVector,
    bx: float,
    gamma: float,
) -> PrimitiveState:
    rho, mom_u, mom_v, mom_w, by, bz, energy = conserved
    u = mom_u / rho
    v = mom_v / rho
    w = mom_w / rho
    kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
    magnetic_energy = 0.5 * (bx**2 + by**2 + bz**2)
    p = (energy - kinetic_energy - magnetic_energy) * (gamma - 1.0)
    return PrimitiveState(rho=rho, u=u, v=v, w=w, by=by, bz=bz, p=p)


def compute_flux(
    *,
    state: PrimitiveState,
    bx: float,
    gamma: float,
) -> ConservedVector:
    total_pressure = state.p + 0.5 * (state.by**2 + state.bz**2)
    energy = compute_energy(state=state, bx=bx, gamma=gamma)
    return numpy.array(
        [
            state.rho * state.u,
            state.rho * state.u**2 + total_pressure,
            state.rho * state.u * state.v - bx * state.by,
            state.rho * state.u * state.w - bx * state.bz,
            state.u * state.by - state.v * bx,
            state.u * state.bz - state.w * bx,
            state.u * (energy + total_pressure) - bx * (state.v * state.by + state.w * state.bz),
        ],
    )


## } MODULE
