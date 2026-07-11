## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from riemann_solver import mhd_state, rankine_hugoniot, solve_shock
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


class TestSolveShock_JumpConditions(unittest.TestCase):

    def test_fast_shock_from_ryu_jones_2a_left_state_satisfies_rankine_hugoniot(
        self,
    ):
        c_fast, _ = mhd_state.compute_fast_slow_speeds(state=_LEFT_STATE, magnetic_field_normal=_MAGNETIC_FIELD_NORMAL, gamma=_GAMMA)
        downstream_state, shock_speed = solve_shock.solve_shock(
            upstream_state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            pressure_downstream=1.5,
            initial_relative_speed_guess=c_fast,
        )
        residual = rankine_hugoniot.compute_jump_residual(
            upstream_state=_LEFT_STATE,
            downstream_state=downstream_state,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            shock_speed=shock_speed,
        )
        for component in residual:
            self.assertAlmostEqual(float(component), 0.0, places=8)

    def test_slow_shock_from_ryu_jones_2a_right_state_satisfies_rankine_hugoniot(
        self,
    ):
        _, c_slow = mhd_state.compute_fast_slow_speeds(state=_RIGHT_STATE, magnetic_field_normal=_MAGNETIC_FIELD_NORMAL, gamma=_GAMMA)
        downstream_state, shock_speed = solve_shock.solve_shock(
            upstream_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            pressure_downstream=1.3,
            initial_relative_speed_guess=-c_slow,
        )
        residual = rankine_hugoniot.compute_jump_residual(
            upstream_state=_RIGHT_STATE,
            downstream_state=downstream_state,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            shock_speed=shock_speed,
        )
        for component in residual:
            self.assertAlmostEqual(float(component), 0.0, places=8)

    def test_hydro_limit_satisfies_rankine_hugoniot(
        self,
    ):
        upstream_state = PrimitiveState(
            density=1.0,
            velocity_normal=0.0,
            velocity_transverse_1=0.0,
            velocity_transverse_2=0.0,
            magnetic_field_transverse_1=0.0,
            magnetic_field_transverse_2=0.0,
            pressure=1.0,
        )
        magnetic_field_normal = 0.0
        gamma = 1.4
        c_fast, _ = mhd_state.compute_fast_slow_speeds(state=upstream_state, magnetic_field_normal=magnetic_field_normal, gamma=gamma)
        downstream_state, shock_speed = solve_shock.solve_shock(
            upstream_state=upstream_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            pressure_downstream=2.0,
            initial_relative_speed_guess=c_fast,
        )
        residual = rankine_hugoniot.compute_jump_residual(
            upstream_state=upstream_state,
            downstream_state=downstream_state,
            magnetic_field_normal=magnetic_field_normal,
            gamma=gamma,
            shock_speed=shock_speed,
        )
        for component in residual:
            self.assertAlmostEqual(float(component), 0.0, places=8)
        self.assertGreater(downstream_state.density, upstream_state.density)


class TestSolveShock_TransverseFieldDirection(unittest.TestCase):

    def test_shock_preserves_transverse_field_direction(
        self,
    ):
        """A shock (unlike a rotational discontinuity) only scales |Bt|; direction is fixed."""
        c_fast, _ = mhd_state.compute_fast_slow_speeds(state=_LEFT_STATE, magnetic_field_normal=_MAGNETIC_FIELD_NORMAL, gamma=_GAMMA)
        downstream_state, _ = solve_shock.solve_shock(
            upstream_state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            pressure_downstream=1.5,
            initial_relative_speed_guess=c_fast,
        )
        self.assertAlmostEqual(
            downstream_state.magnetic_field_transverse_1 / downstream_state.magnetic_field_transverse_2,
            _LEFT_STATE.magnetic_field_transverse_1 / _LEFT_STATE.magnetic_field_transverse_2,
            places=10,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
