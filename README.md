# MHDRiemannSolver (Ægir; Norse god of the sea, presiding over its waves)

An exact solver for the 1D ideal-MHD Riemann problem: given two constant states either side of an initial discontinuity, it root-finds the full 7-wave fan (a fast wave, a rotational discontinuity, and a slow wave on each side of a contact discontinuity) and returns each wave's propagation speed and the resolved state between it and its neighbour. Unlike the approximate Riemann solvers (HLLD, Roe, ...) used inside a CFD code's per-timestep flux update, this one is deliberately slow and exact: it's built for generating reference solutions to validate MHD codes against, not for use inside a simulation's main loop.

This assumes the standard, non-degenerate 7-wave topology: a genuinely nonzero normal field, and initial states whose transverse fields aren't coplanar (coplanar problems, e.g. Brio-Wu, degenerate the rotational discontinuities and aren't reliably solved by this formulation).

Here is the exact solution to the Ryu & Jones (1995) "2a" shock tube, one of the standard non-coplanar test problems for ideal-MHD Riemann solvers:

<div align="center">
  <img src="https://raw.githubusercontent.com/AstroKriel/MHDRiemannSolver/refs/heads/main/gallery/ryu_jones_2a.png" width="70%" />
</div>

## Getting setup

Clone the repository and set up a development environment with [uv](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:AstroKriel/MHDRiemannSolver.git
cd MHDRiemannSolver
uv sync
```

To also install the plotting dependencies needed to run the demo, use:

```bash
uv sync --group demo
```

Run the demo with:

```bash
uv run demos/demo_rj2a.py
```

## Quick start

`solve_riemann_problem` is the main entry point: give it a left/right `PrimitiveState` and a normal magnetic field, and it root-finds the resolved `RiemannSolution`. `sample_profile` then evaluates that solution at any `(position, time)`.

```python
import numpy
from mhd_riemann_solver import PrimitiveState, sample_profile, solve_riemann_problem

left_state = PrimitiveState(
    density=1.08,
    velocity_normal=1.2,
    velocity_transverse_1=0.01,
    velocity_transverse_2=0.5,
    magnetic_field_transverse_1=1.0155412503859613,
    magnetic_field_transverse_2=0.5641895835477562,
    pressure=0.95,
)
right_state = PrimitiveState(
    density=1.0,
    velocity_normal=0.0,
    velocity_transverse_1=0.0,
    velocity_transverse_2=0.0,
    magnetic_field_transverse_1=1.1283791670955125,
    magnetic_field_transverse_2=0.5641895835477562,
    pressure=1.0,
)
magnetic_field_normal = 0.5641895835477562
gamma = 5.0 / 3.0

solution = solve_riemann_problem(
    left_state=left_state,
    right_state=right_state,
    magnetic_field_normal=magnetic_field_normal,
    gamma=gamma,
)
positions = numpy.linspace(0.0, 1.0, 2001)
profile = sample_profile(
    riemann_solution=solution,
    positions=positions,
    time=0.2,
    discontinuity_position=0.5,
)
```

## File structure

```bash
MHDRiemannSolver/                        # project root
├── src/
│   └── mhd_riemann_solver/               # package root (named after Ægir, Norse sea god)
│       ├── __init__.py                   # public API
│       ├── py.typed                      # marker for type checkers (PEP 561)
│       ├── mhd_state.py                  # primitive state, wave speeds, conserved/flux conversions
│       ├── rankine_hugoniot.py           # jump-condition residual, used to validate wave relations
│       ├── rotational_discontinuity.py   # Alfven discontinuity relation
│       ├── solve_shock.py                # MHD shock jump-condition solver
│       ├── solve_rarefaction.py          # MHD rarefaction ODE solver
│       └── exact_solution.py             # coupled 7-wave root-find; the main entry point
├── demos/
│   └── demo_rj2a.py                      # solves and plots the Ryu & Jones (1995) 2a shock tube
├── gallery/
│   └── ryu_jones_2a.png
├── utests/
│   └── test_*.py                         # unit tests, one file per src module
├── pyproject.toml                        # project metadata and dependencies
├── uv.lock                               # lock file (used by uv to pin dependencies)
├── LICENSE                               # terms of use and distribution
└── README.md                             # this file
```

## References

- Ryu, D. & Jones, T. W. (1995), ApJ 442, 228: the "2a" shock tube used in the demo.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
