## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from riemann_solver.mhd_state import PrimitiveState, compute_fast_slow_speeds
from riemann_solver.rankine_hugoniot import compute_jump_residual
from riemann_solver.shock import solve_shock

##
## === TEST SUITE
##


class TestSolveShock_JumpConditions(unittest.TestCase):

    def test_fast_shock_from_rj2a_left_state_satisfies_rankine_hugoniot(
        self,
    ):
        upstream = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        c_fast, _ = compute_fast_slow_speeds(state=upstream, bx=bx, gamma=gamma)
        downstream, shock_speed = solve_shock(
            upstream=upstream,
            bx=bx,
            gamma=gamma,
            p_downstream=1.5,
            initial_relative_speed_guess=c_fast,
        )
        residual = compute_jump_residual(
            upstream=upstream,
            downstream=downstream,
            bx=bx,
            gamma=gamma,
            shock_speed=shock_speed,
        )
        for component in residual:
            self.assertAlmostEqual(float(component), 0.0, places=8)

    def test_slow_shock_from_rj2a_right_state_satisfies_rankine_hugoniot(
        self,
    ):
        upstream = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=1.1283791670955125, bz=0.5641895835477562, p=1.0)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        _, c_slow = compute_fast_slow_speeds(state=upstream, bx=bx, gamma=gamma)
        downstream, shock_speed = solve_shock(
            upstream=upstream,
            bx=bx,
            gamma=gamma,
            p_downstream=1.3,
            initial_relative_speed_guess=-c_slow,
        )
        residual = compute_jump_residual(
            upstream=upstream,
            downstream=downstream,
            bx=bx,
            gamma=gamma,
            shock_speed=shock_speed,
        )
        for component in residual:
            self.assertAlmostEqual(float(component), 0.0, places=8)

    def test_hydro_limit_satisfies_rankine_hugoniot(
        self,
    ):
        upstream = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=0.0, bz=0.0, p=1.0)
        bx = 0.0
        gamma = 1.4
        c_fast, _ = compute_fast_slow_speeds(state=upstream, bx=bx, gamma=gamma)
        downstream, shock_speed = solve_shock(
            upstream=upstream,
            bx=bx,
            gamma=gamma,
            p_downstream=2.0,
            initial_relative_speed_guess=c_fast,
        )
        residual = compute_jump_residual(
            upstream=upstream,
            downstream=downstream,
            bx=bx,
            gamma=gamma,
            shock_speed=shock_speed,
        )
        for component in residual:
            self.assertAlmostEqual(float(component), 0.0, places=8)
        self.assertGreater(downstream.rho, upstream.rho)


class TestSolveShock_TransverseFieldDirection(unittest.TestCase):

    def test_shock_preserves_transverse_field_direction(
        self,
    ):
        """A shock (unlike a rotational discontinuity) only scales |Bt|; direction is fixed."""
        upstream = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        c_fast, _ = compute_fast_slow_speeds(state=upstream, bx=bx, gamma=gamma)
        downstream, _ = solve_shock(
            upstream=upstream,
            bx=bx,
            gamma=gamma,
            p_downstream=1.5,
            initial_relative_speed_guess=c_fast,
        )
        self.assertAlmostEqual(downstream.by / downstream.bz, upstream.by / upstream.bz, places=10)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
