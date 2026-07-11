## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from riemann_solver.mhd_state import PrimitiveState
from riemann_solver.rankine_hugoniot import compute_jump_residual
from riemann_solver.rotational_discontinuity import apply_rotation

##
## === TEST SUITE
##


class TestRotationalDiscontinuity_JumpConditions(unittest.TestCase):

    def test_sign_negative_rotation_satisfies_rankine_hugoniot(
        self,
    ):
        upstream = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        angle = 0.7
        sign = -1.0
        downstream = apply_rotation(upstream=upstream, angle=angle, sign=sign)
        shock_speed = upstream.u - sign * bx / numpy.sqrt(upstream.rho)
        residual = compute_jump_residual(
            upstream=upstream,
            downstream=downstream,
            bx=bx,
            gamma=gamma,
            shock_speed=shock_speed,
        )
        for component in residual:
            self.assertAlmostEqual(float(component), 0.0, places=10)

    def test_sign_positive_rotation_satisfies_rankine_hugoniot(
        self,
    ):
        upstream = PrimitiveState(rho=1.0, u=0.0, v=0.0, w=0.0, by=1.1283791670955125, bz=0.5641895835477562, p=1.0)
        bx = 0.5641895835477562
        gamma = 5.0 / 3.0
        angle = -0.35
        sign = 1.0
        downstream = apply_rotation(upstream=upstream, angle=angle, sign=sign)
        shock_speed = upstream.u - sign * bx / numpy.sqrt(upstream.rho)
        residual = compute_jump_residual(
            upstream=upstream,
            downstream=downstream,
            bx=bx,
            gamma=gamma,
            shock_speed=shock_speed,
        )
        for component in residual:
            self.assertAlmostEqual(float(component), 0.0, places=10)

    def test_rotation_preserves_density_pressure_and_normal_velocity(
        self,
    ):
        upstream = PrimitiveState(rho=1.08, u=1.2, v=0.01, w=0.5, by=1.0155412503859613, bz=0.5641895835477562, p=0.95)
        downstream = apply_rotation(upstream=upstream, angle=1.1, sign=-1.0)
        self.assertAlmostEqual(downstream.rho, upstream.rho, places=12)
        self.assertAlmostEqual(downstream.u, upstream.u, places=12)
        self.assertAlmostEqual(downstream.p, upstream.p, places=12)
        self.assertAlmostEqual(downstream.bt, upstream.bt, places=12)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
