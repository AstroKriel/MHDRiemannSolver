## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from riemann_solver import rankine_hugoniot, rotational_discontinuity
from riemann_solver.mhd_state import PrimitiveState

##
## === FIXTURES
##

_LEFT_STATE = PrimitiveState(
    density=1.08,
    velocity_normal=1.2,
    velocity_transverse_1=0.01,
    velocity_transverse_2=0.5,
    magnetic_field_transverse_1=1.0155412503859613,
    magnetic_field_transverse_2=0.5641895835477562,
    pressure=0.95,
)
_RIGHT_STATE = PrimitiveState(
    density=1.0,
    velocity_normal=0.0,
    velocity_transverse_1=0.0,
    velocity_transverse_2=0.0,
    magnetic_field_transverse_1=1.1283791670955125,
    magnetic_field_transverse_2=0.5641895835477562,
    pressure=1.0,
)
_MAGNETIC_FIELD_NORMAL = 0.5641895835477562
_GAMMA = 5.0 / 3.0

##
## === TEST SUITE
##


class TestRotationalDiscontinuity_JumpConditions(unittest.TestCase):

    def test_sign_negative_rotation_satisfies_rankine_hugoniot(
        self,
    ):
        angle = 0.7
        sign = -1.0
        downstream_state = rotational_discontinuity.apply_rotation(
            upstream_state=_LEFT_STATE,
            angle=angle,
            sign=sign,
        )
        shock_speed = _LEFT_STATE.velocity_normal - sign * _MAGNETIC_FIELD_NORMAL / numpy.sqrt(
            _LEFT_STATE.density,
        )
        residual = rankine_hugoniot.compute_jump_residual(
            upstream_state=_LEFT_STATE,
            downstream_state=downstream_state,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            shock_speed=shock_speed,
        )
        for component in residual:
            self.assertAlmostEqual(float(component), 0.0, places=10)

    def test_sign_positive_rotation_satisfies_rankine_hugoniot(
        self,
    ):
        angle = -0.35
        sign = 1.0
        downstream_state = rotational_discontinuity.apply_rotation(
            upstream_state=_RIGHT_STATE,
            angle=angle,
            sign=sign,
        )
        shock_speed = _RIGHT_STATE.velocity_normal - sign * _MAGNETIC_FIELD_NORMAL / numpy.sqrt(
            _RIGHT_STATE.density,
        )
        residual = rankine_hugoniot.compute_jump_residual(
            upstream_state=_RIGHT_STATE,
            downstream_state=downstream_state,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            shock_speed=shock_speed,
        )
        for component in residual:
            self.assertAlmostEqual(float(component), 0.0, places=10)

    def test_rotation_preserves_density_pressure_and_normal_velocity(
        self,
    ):
        downstream_state = rotational_discontinuity.apply_rotation(
            upstream_state=_LEFT_STATE,
            angle=1.1,
            sign=-1.0,
        )
        self.assertAlmostEqual(downstream_state.density, _LEFT_STATE.density, places=12)
        self.assertAlmostEqual(downstream_state.velocity_normal, _LEFT_STATE.velocity_normal, places=12)
        self.assertAlmostEqual(downstream_state.pressure, _LEFT_STATE.pressure, places=12)
        self.assertAlmostEqual(
            downstream_state.magnetic_field_transverse_magnitude,
            _LEFT_STATE.magnetic_field_transverse_magnitude,
            places=12,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
