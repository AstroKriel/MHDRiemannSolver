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
from riemann_solver.exact_solution import evaluate, solve_riemann_problem
from riemann_solver.mhd_state import PrimitiveState

##
## === TEST SUITE
##


class TestSolveRiemannProblem_RJ2a(unittest.TestCase):
    """
    Cross-check against the RJ2a reference from Ryu & Jones (1995), ApJ 442, 228,
    Figure 2a, using the region densities hardcoded in athenak's own regression
    test (`tst/test_suite/nr/test_nr_rj2a_cpu.py`): an independent, previously
    published solution this solver was not tuned against.
    """

    def test_region_densities_match_ryu_jones_1995(
        self,
    ):
        left = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        right = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=1.1283791670955125, bz=0.5641895835477562, p=1.0)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        solution = solve_riemann_problem(left=left, right=right, bx=bx, gamma=gamma)
        self.assertAlmostEqual(solution.region2.rho, 1.4903, places=3)
        self.assertAlmostEqual(solution.region3.rho, 1.4903, places=3)
        self.assertAlmostEqual(solution.region4.rho, 1.6343, places=3)
        self.assertAlmostEqual(solution.region5.rho, 1.4735, places=3)
        self.assertAlmostEqual(solution.region6.rho, 1.3090, places=3)
        self.assertAlmostEqual(solution.region7.rho, 1.3090, places=3)

    def test_rotational_discontinuities_preserve_density(
        self,
    ):
        left = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        right = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=1.1283791670955125, bz=0.5641895835477562, p=1.0)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        solution = solve_riemann_problem(left=left, right=right, bx=bx, gamma=gamma)
        self.assertAlmostEqual(solution.region2.rho, solution.region3.rho, places=8)
        self.assertAlmostEqual(solution.region6.rho, solution.region7.rho, places=8)

    def test_contact_matches_pressure_and_velocity_not_density(
        self,
    ):
        left = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        right = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=1.1283791670955125, bz=0.5641895835477562, p=1.0)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        solution = solve_riemann_problem(left=left, right=right, bx=bx, gamma=gamma)
        self.assertAlmostEqual(solution.region4.p, solution.region5.p, places=6)
        self.assertAlmostEqual(solution.region4.u, solution.region5.u, places=6)
        self.assertNotAlmostEqual(solution.region4.rho, solution.region5.rho, places=2)

    def test_wave_speeds_match_ryu_jones_1995(
        self,
    ):
        """
        Cross-check against the same wave-speed formulas hardcoded in athenak's
        RJ2a regression test (`compute_error` in `test_nr_rj2a_cpu.py`), sourced
        independently of this solver.
        """
        left = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        right = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=1.1283791670955125, bz=0.5641895835477562, p=1.0)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        solution = solve_riemann_problem(left=left, right=right, bx=bx, gamma=gamma)
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


class TestEvaluate_SamplesCorrectRegion(unittest.TestCase):

    def test_evaluate_returns_expected_region_at_sample_points(
        self,
    ):
        left = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        right = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=1.1283791670955125, bz=0.5641895835477562, p=1.0)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        solution = solve_riemann_problem(left=left, right=right, bx=bx, gamma=gamma)
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
        profile = evaluate(solution=solution, x=sample_positions, t=t, x0=x0)
        for expected_region, got_state in zip(expected_regions, profile):
            self.assertAlmostEqual(got_state.rho, expected_region.rho, places=8)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
