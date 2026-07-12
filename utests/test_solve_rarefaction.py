## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from aegir import mhd_state, solve_rarefaction, solve_shock
from aegir.mhd_state import PrimitiveState, WaveFamily

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


class TestEigensystem_MatchesAnalyticWaveSpeeds(unittest.TestCase):

    def test_eigenvalues_match_fast_slow_alfven_entropy_speeds(
        self,
    ):
        """
        The numerically-built primitive Jacobian's eigenvalues must reproduce the
        7 standard MHD characteristic speeds exactly, `u` and `u +/- c_alfven`
        being the easy modes, `u +/- c_fast`/`u +/- c_slow` the ones a bug in
        `compute_flux` can quietly corrupt without breaking any jump-condition test
        (jump conditions only see flux differences; eigenvalues see the flux's
        absolute derivative, which is why this test caught a real bug the RH-residual
        tests did not).
        """
        c_fast, c_slow = mhd_state.compute_fast_slow_speeds(
            state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        c_alfven = mhd_state.compute_alfven_speed(
            state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
        )
        expected = sorted(
            [
                _LEFT_STATE.velocity_normal - c_fast,
                _LEFT_STATE.velocity_normal - c_alfven,
                _LEFT_STATE.velocity_normal - c_slow,
                _LEFT_STATE.velocity_normal,
                _LEFT_STATE.velocity_normal + c_slow,
                _LEFT_STATE.velocity_normal + c_alfven,
                _LEFT_STATE.velocity_normal + c_fast,
            ],
        )
        eigenvalues, _ = solve_rarefaction._compute_primitive_eigensystem(
            state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        for expected_value, got_value in zip(expected, sorted(eigenvalues)):
            self.assertAlmostEqual(got_value, expected_value, places=6)


class TestEigensystem_IsentropicConsistency(unittest.TestCase):

    def test_fast_family_density_component_matches_isentropic_slope(
        self,
    ):
        """
        A genuine MHD fast/slow eigenvector must satisfy `d(density)/dp = density
        / (gamma * pressure)` exactly: entropy is conserved along these
        characteristics by construction, so this is an independent check on the
        eigenvector itself, not just its eigenvalue.
        """
        c_fast, _ = mhd_state.compute_fast_slow_speeds(
            state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        direction = solve_rarefaction._select_rarefaction_direction(
            state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            target_eigenvalue=_LEFT_STATE.velocity_normal - c_fast,
        )
        self.assertAlmostEqual(direction[0], _LEFT_STATE.density / (_GAMMA * _LEFT_STATE.pressure), places=6)


class TestSolveRarefaction_WeakWaveLimit(unittest.TestCase):

    def test_weak_fast_rarefaction_converges_to_weak_fast_shock(
        self,
    ):
        """A vanishingly weak rarefaction and shock of the same family must coincide."""
        c_fast, _ = mhd_state.compute_fast_slow_speeds(
            state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        delta_p = 1e-4
        rarefaction_state = solve_rarefaction.solve_rarefaction(
            upstream_state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            pressure_downstream=_LEFT_STATE.pressure - delta_p,
            wave_family=WaveFamily.Fast,
            wave_speed_sign=-1.0,
        )
        shock_state, _ = solve_shock.solve_shock(
            upstream_state=_LEFT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            pressure_downstream=_LEFT_STATE.pressure - delta_p,
            initial_relative_speed_guess=c_fast,
        )
        self.assertAlmostEqual(rarefaction_state.velocity_normal, shock_state.velocity_normal, places=8)
        self.assertAlmostEqual(rarefaction_state.density, shock_state.density, places=8)
        self.assertAlmostEqual(
            rarefaction_state.magnetic_field_transverse_1,
            shock_state.magnetic_field_transverse_1,
            places=8,
        )

    def test_weak_slow_rarefaction_converges_to_weak_slow_shock(
        self,
    ):
        _, c_slow = mhd_state.compute_fast_slow_speeds(
            state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
        )
        delta_p = 1e-4
        rarefaction_state = solve_rarefaction.solve_rarefaction(
            upstream_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            pressure_downstream=_RIGHT_STATE.pressure + delta_p,
            wave_family=WaveFamily.Slow,
            wave_speed_sign=1.0,
        )
        shock_state, _ = solve_shock.solve_shock(
            upstream_state=_RIGHT_STATE,
            magnetic_field_normal=_MAGNETIC_FIELD_NORMAL,
            gamma=_GAMMA,
            pressure_downstream=_RIGHT_STATE.pressure + delta_p,
            initial_relative_speed_guess=-c_slow,
        )
        self.assertAlmostEqual(rarefaction_state.velocity_normal, shock_state.velocity_normal, places=8)
        self.assertAlmostEqual(rarefaction_state.density, shock_state.density, places=8)
        self.assertAlmostEqual(
            rarefaction_state.magnetic_field_transverse_1,
            shock_state.magnetic_field_transverse_1,
            places=8,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
