## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

## third-party
import numpy
from numpy.typing import NDArray

##
## === TYPE ALIASES
##

## conserved vector: [density, momentum_0, momentum_1, momentum_2, magnetic_field_transverse_1, magnetic_field_transverse_2, energy]
ConservedVector: TypeAlias = NDArray[Any]

##
## === WAVE FAMILY
##


class WaveFamily(str, Enum):
    Fast = "fast"
    Slow = "slow"


##
## === PRIMITIVE STATE
##


@dataclass(frozen=True)
class PrimitiveState:
    """
    A pointwise ideal-MHD primitive state.

    `magnetic_field_normal` (`bx`) is not stored here: it is one constant shared
    by the whole Riemann problem, not a per-region unknown, so callers thread it
    through separately.
    """

    density: float
    velocity_normal: float
    velocity_transverse_1: float
    velocity_transverse_2: float
    magnetic_field_transverse_1: float
    magnetic_field_transverse_2: float
    pressure: float

    @property
    def magnetic_field_transverse_magnitude(
        self,
    ) -> float:
        """Magnitude of the transverse field (`magnetic_field_transverse_1`, `magnetic_field_transverse_2`)."""
        return float(
            numpy.hypot(
                self.magnetic_field_transverse_1,
                self.magnetic_field_transverse_2,
            ),
        )


##
## === WAVE SPEEDS
##


def compute_sound_speed(
    *,
    state: PrimitiveState,
    gamma: float,
) -> float:
    """Return the adiabatic sound speed of `state`."""
    return float(numpy.sqrt(gamma * state.pressure / state.density))


def compute_alfven_speed(
    *,
    state: PrimitiveState,
    magnetic_field_normal: float,
) -> float:
    """Return the Alfven speed of `state` along `magnetic_field_normal`."""
    return float(numpy.abs(magnetic_field_normal) / numpy.sqrt(state.density))


def compute_fast_slow_speeds(
    *,
    state: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
) -> tuple[float, float]:
    """
    Return `(c_fast, c_slow)`, the fast/slow magnetosonic speeds normal to
    `magnetic_field_normal`.
    """
    sound_speed_sq = gamma * state.pressure / state.density
    total_alfven_speed_sq = (
        magnetic_field_normal**2 + state.magnetic_field_transverse_1**2 + state.magnetic_field_transverse_2**2
    ) / state.density
    normal_alfven_speed_sq = magnetic_field_normal**2 / state.density
    discriminant = (sound_speed_sq + total_alfven_speed_sq)**2 - 4.0 * sound_speed_sq * normal_alfven_speed_sq
    root = numpy.sqrt(
        max(
            discriminant,
            0.0,
        ),
    )
    c_fast = numpy.sqrt(0.5 * (sound_speed_sq + total_alfven_speed_sq + root))
    c_slow = numpy.sqrt(
        max(
            0.5 * (sound_speed_sq + total_alfven_speed_sq - root),
            0.0,
        ),
    )
    return float(c_fast), float(c_slow)


##
## === CONSERVED, PRIMITIVE, AND FLUX CONVERSIONS
##


def compute_energy(
    *,
    state: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
) -> float:
    """Return the total energy density of `state` (internal + kinetic + magnetic)."""
    kinetic_energy = 0.5 * state.density * (
        state.velocity_normal**2 + state.velocity_transverse_1**2 + state.velocity_transverse_2**2
    )
    magnetic_energy = 0.5 * (
        magnetic_field_normal**2 + state.magnetic_field_transverse_1**2 + state.magnetic_field_transverse_2**2
    )
    internal_energy = state.pressure / (gamma - 1.0)
    return internal_energy + kinetic_energy + magnetic_energy


def as_conserved(
    *,
    state: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
) -> ConservedVector:
    """Return `state` resolved to its conserved-variable representation."""
    momentum_0 = state.density * state.velocity_normal
    momentum_1 = state.density * state.velocity_transverse_1
    momentum_2 = state.density * state.velocity_transverse_2
    energy = compute_energy(
        state=state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    return numpy.array(
        [
            state.density,
            momentum_0,
            momentum_1,
            momentum_2,
            state.magnetic_field_transverse_1,
            state.magnetic_field_transverse_2,
            energy,
        ],
    )


def as_primitive(
    *,
    conserved: ConservedVector,
    magnetic_field_normal: float,
    gamma: float,
) -> PrimitiveState:
    """Return `conserved` resolved to its primitive-variable representation."""
    density, momentum_0, momentum_1, momentum_2, magnetic_field_transverse_1, magnetic_field_transverse_2, energy = conserved
    velocity_normal = momentum_0 / density
    velocity_transverse_1 = momentum_1 / density
    velocity_transverse_2 = momentum_2 / density
    kinetic_energy = 0.5 * density * (
        velocity_normal**2 + velocity_transverse_1**2 + velocity_transverse_2**2
    )
    magnetic_energy = 0.5 * (
        magnetic_field_normal**2 + magnetic_field_transverse_1**2 + magnetic_field_transverse_2**2
    )
    pressure = (energy - kinetic_energy - magnetic_energy) * (gamma - 1.0)
    return PrimitiveState(
        density=density,
        velocity_normal=velocity_normal,
        velocity_transverse_1=velocity_transverse_1,
        velocity_transverse_2=velocity_transverse_2,
        magnetic_field_transverse_1=magnetic_field_transverse_1,
        magnetic_field_transverse_2=magnetic_field_transverse_2,
        pressure=pressure,
    )


def compute_flux(
    *,
    state: PrimitiveState,
    magnetic_field_normal: float,
    gamma: float,
) -> ConservedVector:
    """Return the ideal-MHD flux vector of `state`."""
    total_pressure = state.pressure + 0.5 * (
        magnetic_field_normal**2 + state.magnetic_field_transverse_1**2 + state.magnetic_field_transverse_2**2
    )
    energy = compute_energy(
        state=state,
        magnetic_field_normal=magnetic_field_normal,
        gamma=gamma,
    )
    mass_flux = state.density * state.velocity_normal
    momentum_0_flux = state.density * state.velocity_normal**2 + total_pressure - magnetic_field_normal**2
    momentum_1_flux = (
        state.density * state.velocity_normal * state.velocity_transverse_1 -
        magnetic_field_normal * state.magnetic_field_transverse_1
    )
    momentum_2_flux = (
        state.density * state.velocity_normal * state.velocity_transverse_2 -
        magnetic_field_normal * state.magnetic_field_transverse_2
    )
    magnetic_field_transverse_1_flux = (
        state.velocity_normal * state.magnetic_field_transverse_1 -
        state.velocity_transverse_1 * magnetic_field_normal
    )
    magnetic_field_transverse_2_flux = (
        state.velocity_normal * state.magnetic_field_transverse_2 -
        state.velocity_transverse_2 * magnetic_field_normal
    )
    energy_flux = state.velocity_normal * (energy + total_pressure) - magnetic_field_normal * (
        state.velocity_normal * magnetic_field_normal + state.velocity_transverse_1 *
        state.magnetic_field_transverse_1 + state.velocity_transverse_2 * state.magnetic_field_transverse_2
    )
    return numpy.array(
        [
            mass_flux,
            momentum_0_flux,
            momentum_1_flux,
            momentum_2_flux,
            magnetic_field_transverse_1_flux,
            magnetic_field_transverse_2_flux,
            energy_flux,
        ],
    )


## } MODULE
