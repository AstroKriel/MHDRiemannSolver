## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import math
import unittest

## third-party
import numpy

## local
from riemann_solver import exact_solution
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


class TestSolveRiemannProblem_RyuJones2a(unittest.TestCase):
    """
    Cross-check against the Ryu & Jones (1995), ApJ 442, 228, Figure 2a reference,
    using the region densities hardcoded in athenak's own regression test
    (`tst/test_suite/nr/test_nr_rj2a_cpu.py`): an independent, previously
    published solution this solver was not tuned against.
    """

    def test_region_densities_match_ryu_jones_1995(
        self,
    ):
        riemann_solution = exact_solution.solve_riemann_problem(
            left_state=_LEFT_STATE,
            right_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        self.assertAlmostEqual(riemann_solution.left_fast_wave.state.density, 1.4903, places=3)
        self.assertAlmostEqual(riemann_solution.left_rotation_discontinuity.state.density, 1.4903, places=3)
        self.assertAlmostEqual(riemann_solution.left_slow_wave.state.density, 1.6343, places=3)
        self.assertAlmostEqual(riemann_solution.contact.state.density, 1.4735, places=3)
        self.assertAlmostEqual(riemann_solution.right_slow_wave.state.density, 1.3090, places=3)
        self.assertAlmostEqual(riemann_solution.right_rotation_discontinuity.state.density, 1.3090, places=3)

    def test_rotational_discontinuities_preserve_density(
        self,
    ):
        riemann_solution = exact_solution.solve_riemann_problem(
            left_state=_LEFT_STATE,
            right_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        self.assertAlmostEqual(
            riemann_solution.left_fast_wave.state.density,
            riemann_solution.left_rotation_discontinuity.state.density,
            places=8,
        )
        self.assertAlmostEqual(
            riemann_solution.right_slow_wave.state.density,
            riemann_solution.right_rotation_discontinuity.state.density,
            places=8,
        )

    def test_contact_matches_pressure_and_velocity_not_density(
        self,
    ):
        riemann_solution = exact_solution.solve_riemann_problem(
            left_state=_LEFT_STATE,
            right_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        self.assertAlmostEqual(
            riemann_solution.left_slow_wave.state.pressure,
            riemann_solution.contact.state.pressure,
            places=6,
        )
        self.assertAlmostEqual(
            riemann_solution.left_slow_wave.state.velocity_normal,
            riemann_solution.contact.state.velocity_normal,
            places=6,
        )
        self.assertNotAlmostEqual(
            riemann_solution.left_slow_wave.state.density,
            riemann_solution.contact.state.density,
            places=2,
        )

    def test_wave_speeds_match_ryu_jones_1995(
        self,
    ):
        """
        Cross-check against the same wave-speed formulas hardcoded in athenak's
        Ryu-Jones 2a regression test (`compute_error` in `test_nr_rj2a_cpu.py`),
        sourced independently of this solver.
        """
        riemann_solution = exact_solution.solve_riemann_problem(
            left_state=_LEFT_STATE,
            right_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        expected_left_fast_wave = 1.2 - 2.3305 / 1.08
        expected_left_rotation_discontinuity = 0.60588 - 1.0 / math.sqrt(math.pi * 1.4903)
        expected_left_slow_wave = 0.60588 - 0.51594 / 1.4903
        expected_contact = 0.57538
        expected_right_slow_wave = 0.53432 + 0.48144 / 1.309
        expected_right_rotation_discontinuity = 0.53432 + 1.0 / math.sqrt(math.pi * 1.309)
        expected_right_fast_wave = 2.2638
        self.assertAlmostEqual(
            riemann_solution.left_fast_wave.wave_propagation.head_speed,
            expected_left_fast_wave,
            places=3,
        )
        self.assertAlmostEqual(
            riemann_solution.left_rotation_discontinuity.wave_propagation.head_speed,
            expected_left_rotation_discontinuity,
            places=3,
        )
        self.assertAlmostEqual(
            riemann_solution.left_slow_wave.wave_propagation.head_speed,
            expected_left_slow_wave,
            places=3,
        )
        self.assertAlmostEqual(
            riemann_solution.contact.wave_propagation.head_speed,
            expected_contact,
            places=3,
        )
        self.assertAlmostEqual(
            riemann_solution.right_slow_wave.wave_propagation.head_speed,
            expected_right_slow_wave,
            places=3,
        )
        self.assertAlmostEqual(
            riemann_solution.right_rotation_discontinuity.wave_propagation.head_speed,
            expected_right_rotation_discontinuity,
            places=3,
        )
        self.assertAlmostEqual(
            riemann_solution.right_fast_wave.wave_propagation.head_speed,
            expected_right_fast_wave,
            places=3,
        )


class TestSampleProfile_ReturnsCorrectRegion(unittest.TestCase):

    def test_sample_profile_returns_expected_region_at_sample_points(
        self,
    ):
        riemann_solution = exact_solution.solve_riemann_problem(
            left_state=_LEFT_STATE,
            right_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        time = 0.2
        discontinuity_position = 0.5
        sample_positions = numpy.array([0.0, 0.4, 0.58, 0.62, 0.69, 1.0])
        expected_regions = [
            riemann_solution.left_state,
            riemann_solution.left_fast_wave.state,
            riemann_solution.left_slow_wave.state,
            riemann_solution.contact.state,
            riemann_solution.right_slow_wave.state,
            riemann_solution.right_fast_wave.state,
        ]
        profile = exact_solution.sample_profile(
            riemann_solution=riemann_solution,
            positions=sample_positions,
            time=time,
            discontinuity_position=discontinuity_position,
        )
        for expected_region, got_state in zip(expected_regions, profile):
            self.assertAlmostEqual(got_state.density, expected_region.density, places=8)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
