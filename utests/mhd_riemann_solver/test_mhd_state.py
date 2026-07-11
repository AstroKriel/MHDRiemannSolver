## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from riemann_solver import mhd_state
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


class TestWaveSpeeds_Ordering(unittest.TestCase):

    def test_slow_le_alfven_le_fast_for_generic_state(
        self,
    ):
        c_fast, c_slow = mhd_state.compute_fast_slow_speeds(
            state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        c_alfven = mhd_state.compute_alfven_speed(
            state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
        )
        self.assertLessEqual(c_slow, c_alfven + 1e-12)
        self.assertLessEqual(c_alfven, c_fast + 1e-12)

    def test_slow_le_alfven_le_fast_for_ryu_jones_2a_right_state(
        self,
    ):
        c_fast, c_slow = mhd_state.compute_fast_slow_speeds(
            state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        c_alfven = mhd_state.compute_alfven_speed(
            state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
        )
        self.assertLessEqual(c_slow, c_alfven + 1e-12)
        self.assertLessEqual(c_alfven, c_fast + 1e-12)


class TestWaveSpeeds_HydroLimit(unittest.TestCase):

    def test_zero_field_recovers_sound_speed(
        self,
    ):
        state = PrimitiveState(
            density=1.0,
            velocity_normal=0.0,
            velocity_transverse_1=0.0,
            velocity_transverse_2=0.0,
            magnetic_field_transverse_1=0.0,
            magnetic_field_transverse_2=0.0,
            pressure=1.0,
        )
        c_fast, c_slow = mhd_state.compute_fast_slow_speeds(
            state=state,
            magnetic_field_normal=0.0,
            gamma=_GAMMA,
        )
        c_sound = mhd_state.compute_sound_speed(
            state=state,
            gamma=_GAMMA,
        )
        self.assertAlmostEqual(c_fast, c_sound, places=12)
        self.assertAlmostEqual(c_slow, 0.0, places=12)


class TestConservedPrimitive_RoundTrip(unittest.TestCase):

    def test_as_conserved_and_back_recovers_state(
        self,
    ):
        conserved = mhd_state.as_conserved(
            state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        recovered = mhd_state.as_primitive(
            conserved=conserved,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        self.assertAlmostEqual(recovered.density, _LEFT_STATE.density, places=12)
        self.assertAlmostEqual(recovered.velocity_normal, _LEFT_STATE.velocity_normal, places=12)
        self.assertAlmostEqual(recovered.velocity_transverse_1, _LEFT_STATE.velocity_transverse_1, places=12)
        self.assertAlmostEqual(recovered.velocity_transverse_2, _LEFT_STATE.velocity_transverse_2, places=12)
        self.assertAlmostEqual(
            recovered.magnetic_field_transverse_1, _LEFT_STATE.magnetic_field_transverse_1, places=12
        )
        self.assertAlmostEqual(
            recovered.magnetic_field_transverse_2, _LEFT_STATE.magnetic_field_transverse_2, places=12
        )
        self.assertAlmostEqual(recovered.pressure, _LEFT_STATE.pressure, places=12)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
