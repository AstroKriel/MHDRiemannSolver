## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from riemann_solver.exact_solution import solve_riemann_problem
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


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
