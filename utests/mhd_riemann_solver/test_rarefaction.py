## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from riemann_solver.mhd_state import PrimitiveState, compute_alfven_speed, compute_fast_slow_speeds
from riemann_solver.rarefaction import _compute_primitive_eigensystem, _select_rarefaction_direction, solve_rarefaction
from riemann_solver.shock import solve_shock

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
        state = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        c_fast, c_slow = compute_fast_slow_speeds(state=state, bx=bx, gamma=gamma)
        c_alfven = compute_alfven_speed(state=state, bx=bx)
        expected = sorted(
            [
                state.u - c_fast,
                state.u - c_alfven,
                state.u - c_slow,
                state.u,
                state.u + c_slow,
                state.u + c_alfven,
                state.u + c_fast,
            ],
        )
        eigenvalues, _ = _compute_primitive_eigensystem(state=state, bx=bx, gamma=gamma)
        for expected_value, got_value in zip(expected, sorted(eigenvalues)):
            self.assertAlmostEqual(got_value, expected_value, places=6)


class TestEigensystem_IsentropicConsistency(unittest.TestCase):

    def test_fast_family_rho_component_matches_isentropic_slope(
        self,
    ):
        """
        A genuine MHD fast/slow eigenvector must satisfy `d(rho)/dp = rho / (gamma * p)`
        exactly: entropy is conserved along these characteristics by construction, so
        this is an independent check on the eigenvector itself, not just its eigenvalue.
        """
        state = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        c_fast, _ = compute_fast_slow_speeds(state=state, bx=bx, gamma=gamma)
        direction = _select_rarefaction_direction(state=state, bx=bx, gamma=gamma, target_eigenvalue=state.u - c_fast)
        self.assertAlmostEqual(direction[0], state.rho / (gamma * state.p), places=6)


class TestSolveRarefaction_WeakWaveLimit(unittest.TestCase):

    def test_weak_fast_rarefaction_converges_to_weak_fast_shock(
        self,
    ):
        """A vanishingly weak rarefaction and shock of the same family must coincide."""
        upstream = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        c_fast, _ = compute_fast_slow_speeds(state=upstream, bx=bx, gamma=gamma)
        delta_p = 1e-4
        rarefaction_state = solve_rarefaction(
            upstream=upstream,
            bx=bx,
            gamma=gamma,
            p_downstream=upstream.p - delta_p,
            wave_family="fast",
            wave_speed_sign=-1.0,
        )
        shock_state, _ = solve_shock(
            upstream=upstream,
            bx=bx,
            gamma=gamma,
            p_downstream=upstream.p - delta_p,
            initial_relative_speed_guess=c_fast,
        )
        self.assertAlmostEqual(rarefaction_state.u, shock_state.u, places=8)
        self.assertAlmostEqual(rarefaction_state.rho, shock_state.rho, places=8)
        self.assertAlmostEqual(rarefaction_state.by, shock_state.by, places=8)

    def test_weak_slow_rarefaction_converges_to_weak_slow_shock(
        self,
    ):
        upstream = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=1.1283791670955125, bz=0.5641895835477562, p=1.0)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        _, c_slow = compute_fast_slow_speeds(state=upstream, bx=bx, gamma=gamma)
        delta_p = 1e-4
        rarefaction_state = solve_rarefaction(
            upstream=upstream,
            bx=bx,
            gamma=gamma,
            p_downstream=upstream.p + delta_p,
            wave_family="slow",
            wave_speed_sign=1.0,
        )
        shock_state, _ = solve_shock(
            upstream=upstream,
            bx=bx,
            gamma=gamma,
            p_downstream=upstream.p + delta_p,
            initial_relative_speed_guess=-c_slow,
        )
        self.assertAlmostEqual(rarefaction_state.u, shock_state.u, places=8)
        self.assertAlmostEqual(rarefaction_state.rho, shock_state.rho, places=8)
        self.assertAlmostEqual(rarefaction_state.by, shock_state.by, places=8)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
