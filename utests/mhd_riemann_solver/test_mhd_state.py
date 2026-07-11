## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from riemann_solver.mhd_state import (
    PrimitiveState,
    compute_alfven_speed,
    compute_fast_slow_speeds,
    compute_sound_speed,
    conserved_to_primitive,
    primitive_to_conserved,
)

##
## === TEST SUITE
##


class TestWaveSpeeds_Ordering(unittest.TestCase):

    def test_slow_le_alfven_le_fast_for_generic_state(
        self,
    ):
        state = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        c_fast, c_slow = compute_fast_slow_speeds(state=state, bx=bx, gamma=gamma)
        c_alfven = compute_alfven_speed(state=state, bx=bx)
        self.assertLessEqual(c_slow, c_alfven + 1e-12)
        self.assertLessEqual(c_alfven, c_fast + 1e-12)

    def test_slow_le_alfven_le_fast_for_rj2a_right_state(
        self,
    ):
        state = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=1.1283791670955125, bz=0.5641895835477562, p=1.0)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        c_fast, c_slow = compute_fast_slow_speeds(state=state, bx=bx, gamma=gamma)
        c_alfven = compute_alfven_speed(state=state, bx=bx)
        self.assertLessEqual(c_slow, c_alfven + 1e-12)
        self.assertLessEqual(c_alfven, c_fast + 1e-12)


class TestWaveSpeeds_HydroLimit(unittest.TestCase):

    def test_zero_field_recovers_sound_speed(
        self,
    ):
        state = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=0.0, bz=0.0, p=1.0)
        gamma = 5.0 / 3.0
        c_fast, c_slow = compute_fast_slow_speeds(state=state, bx=0.0, gamma=gamma)
        c_sound = compute_sound_speed(state=state, gamma=gamma)
        self.assertAlmostEqual(c_fast, c_sound, places=12)
        self.assertAlmostEqual(c_slow, 0.0, places=12)


class TestConservedPrimitive_RoundTrip(unittest.TestCase):

    def test_primitive_to_conserved_and_back_recovers_state(
        self,
    ):
        state = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        conserved = primitive_to_conserved(state=state, bx=bx, gamma=gamma)
        recovered = conserved_to_primitive(conserved=conserved, bx=bx, gamma=gamma)
        self.assertAlmostEqual(recovered.rho, state.rho, places=12)
        self.assertAlmostEqual(recovered.u, state.u, places=12)
        self.assertAlmostEqual(recovered.v, state.v, places=12)
        self.assertAlmostEqual(recovered.w, state.w, places=12)
        self.assertAlmostEqual(recovered.by, state.by, places=12)
        self.assertAlmostEqual(recovered.bz, state.bz, places=12)
        self.assertAlmostEqual(recovered.p, state.p, places=12)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
