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
        solution = exact_solution.solve_riemann_problem(
            left_state=_LEFT_STATE,
            right_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        self.assertAlmostEqual(solution.region2.density, 1.4903, places=3)
        self.assertAlmostEqual(solution.region3.density, 1.4903, places=3)
        self.assertAlmostEqual(solution.region4.density, 1.6343, places=3)
        self.assertAlmostEqual(solution.region5.density, 1.4735, places=3)
        self.assertAlmostEqual(solution.region6.density, 1.3090, places=3)
        self.assertAlmostEqual(solution.region7.density, 1.3090, places=3)

    def test_rotational_discontinuities_preserve_density(
        self,
    ):
        solution = exact_solution.solve_riemann_problem(
            left_state=_LEFT_STATE,
            right_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        self.assertAlmostEqual(solution.region2.density, solution.region3.density, places=8)
        self.assertAlmostEqual(solution.region6.density, solution.region7.density, places=8)

    def test_contact_matches_pressure_and_velocity_not_density(
        self,
    ):
        solution = exact_solution.solve_riemann_problem(
            left_state=_LEFT_STATE,
            right_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        self.assertAlmostEqual(solution.region4.pressure, solution.region5.pressure, places=6)
        self.assertAlmostEqual(solution.region4.velocity_normal, solution.region5.velocity_normal, places=6)
        self.assertNotAlmostEqual(solution.region4.density, solution.region5.density, places=2)

    def test_wave_speeds_match_ryu_jones_1995(
        self,
    ):
        """
        Cross-check against the same wave-speed formulas hardcoded in athenak's
        Ryu-Jones 2a regression test (`compute_error` in `test_nr_rj2a_cpu.py`),
        sourced independently of this solver.
        """
        solution = exact_solution.solve_riemann_problem(
            left_state=_LEFT_STATE,
            right_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        expected_fast_left = 1.2 - 2.3305 / 1.08
        expected_rotation_left = 0.60588 - 1.0 / math.sqrt(math.pi * 1.4903)
        expected_slow_left = 0.60588 - 0.51594 / 1.4903
        expected_contact = 0.57538
        expected_slow_right = 0.53432 + 0.48144 / 1.309
        expected_rotation_right = 0.53432 + 1.0 / math.sqrt(math.pi * 1.309)
        expected_fast_right = 2.2638
        self.assertAlmostEqual(solution.fast_left.head_speed, expected_fast_left, places=3)
        self.assertAlmostEqual(solution.rotation_left.head_speed, expected_rotation_left, places=3)
        self.assertAlmostEqual(solution.slow_left.head_speed, expected_slow_left, places=3)
        self.assertAlmostEqual(solution.contact.head_speed, expected_contact, places=3)
        self.assertAlmostEqual(solution.slow_right.head_speed, expected_slow_right, places=3)
        self.assertAlmostEqual(solution.rotation_right.head_speed, expected_rotation_right, places=3)
        self.assertAlmostEqual(solution.fast_right.head_speed, expected_fast_right, places=3)


class TestSampleProfile_ReturnsCorrectRegion(unittest.TestCase):

    def test_sample_profile_returns_expected_region_at_sample_points(
        self,
    ):
        solution = exact_solution.solve_riemann_problem(
            left_state=_LEFT_STATE,
            right_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        t = 0.2
        x0 = 0.5
        sample_positions = numpy.array([0.0, 0.4, 0.58, 0.62, 0.69, 1.0])
        expected_regions = [
            solution.region1,
            solution.region2,
            solution.region4,
            solution.region5,
            solution.region6,
            solution.region8,
        ]
        profile = exact_solution.sample_profile(solution=solution, x=sample_positions, t=t, x0=x0)
        for expected_region, got_state in zip(expected_regions, profile):
            self.assertAlmostEqual(got_state.density, expected_region.density, places=8)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
