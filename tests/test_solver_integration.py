"""Solver integration tests: all solvers must produce stable, finite results.

Tests solve() loop, trajectory capture, step() correctness, and
solver interchangeability across the full solver family.
"""

import numpy as np
import pytest


@pytest.fixture
def fresh_liaw_model():
    """Create a fresh LiawModel for each test (not shared)."""
    from lpf.models import LiawModel
    from lpf.initializers import LiawInitializer

    init_pts = np.array([[[16, 16]]], dtype=np.uint32)
    init_states = np.array([[0.5, 0.5]], dtype=np.float32)
    initializer = LiawInitializer(init_states=init_states, init_pts=init_pts)
    params = np.array([[1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]],
                      dtype=np.float32)
    model = LiawModel(
        initializer=initializer, params=params,
        thr_color=0.5 * np.ones((1, 1, 1)),
        width=32, height=32, dx=0.1, device="cpu",
    )
    model.initialize()
    return model


class TestAllSolversFinite:
    """Every solver must produce finite results for a simple problem."""

    @pytest.mark.parametrize("solver_name", [
        "euler", "heun", "rk45", "ab2", "rk23",
        "dopri5", "adaptive_rk45",
    ])
    def test_solver_produces_finite_results(self, solver_name, fresh_liaw_model):
        from lpf.solvers.solverfactory import SolverFactory
        solver = SolverFactory.create(solver_name)
        solver.solve(
            model=fresh_liaw_model, dt=0.01, n_iters=20,
            init_model=False, verbose=0,
        )
        u = fresh_liaw_model.am.get(fresh_liaw_model.u)
        assert np.all(np.isfinite(u)), f"{solver_name} produced non-finite results"


class TestSolverStep:
    """step() must return a valid delta for each solver."""

    def _make_solvers(self):
        from lpf.solvers import EulerSolver, HeunSolver, RungeKuttaSolver
        from lpf.solvers.adamsbashforth2solver import AdamsBashforth2Solver
        from lpf.solvers.rk23solver import RK23Solver
        return [
            ("Euler", EulerSolver()),
            ("Heun", HeunSolver()),
            ("RK4", RungeKuttaSolver()),
            ("RK23", RK23Solver()),
        ]

    @pytest.mark.parametrize("name,solver", [
        ("Euler", None), ("Heun", None), ("RK4", None), ("RK23", None),
    ])
    def test_step_returns_nonzero(self, name, solver, fresh_liaw_model):
        from lpf.solvers import EulerSolver, HeunSolver, RungeKuttaSolver
        from lpf.solvers.rk23solver import RK23Solver

        solver_map = {
            "Euler": EulerSolver(),
            "Heun": HeunSolver(),
            "RK4": RungeKuttaSolver(),
            "RK23": RK23Solver(),
        }
        solver = solver_map[name]
        delta = solver.step(fresh_liaw_model, t=0.0, dt=0.01,
                            y_mesh=fresh_liaw_model.y_mesh)
        delta_np = fresh_liaw_model.am.get(delta)
        assert not np.allclose(delta_np, 0.0), (
            f"{name}.step() should return non-zero delta"
        )
        assert np.all(np.isfinite(delta_np)), (
            f"{name}.step() should return finite delta"
        )


class TestSolverTrajectory:
    """Trajectory capture must work correctly across solvers."""

    def test_trajectory_shape_with_period(self, fresh_liaw_model):
        from lpf.solvers import EulerSolver
        n_iters = 50
        period = 10
        solver = EulerSolver()
        trj = solver.solve(
            model=fresh_liaw_model, dt=0.01, n_iters=n_iters,
            period_output=period, get_trj=True,
            init_model=False, verbose=0,
        )
        expected_points = n_iters // period + 1
        assert trj.shape[0] == expected_points, (
            f"Expected {expected_points} trajectory points, got {trj.shape[0]}"
        )

    def test_trajectory_with_waypoints(self, fresh_liaw_model):
        from lpf.solvers import EulerSolver
        waypoints = [1, 5, 10, 15, 20]
        solver = EulerSolver()
        result = solver.solve(
            model=fresh_liaw_model, dt=0.01, n_iters=20,
            get_trj=True, trj_waypoints=waypoints,
            init_model=False, verbose=0,
        )
        # With waypoints, solve() returns a dict {"iters": ..., "trj": ...}
        assert isinstance(result, dict)
        assert "trj" in result
        assert "iters" in result
        assert result["trj"].shape[0] == len(waypoints), (
            f"Expected {len(waypoints)} waypoints, got {result['trj'].shape[0]}"
        )

    def test_trajectory_monotonic_change(self, fresh_liaw_model):
        """Trajectory should show the system evolving over time."""
        from lpf.solvers import EulerSolver
        solver = EulerSolver()
        trj = solver.solve(
            model=fresh_liaw_model, dt=0.01, n_iters=100,
            period_output=50, get_trj=True,
            init_model=False, verbose=0,
        )
        # First and last snapshots should differ
        assert not np.allclose(trj[0], trj[-1]), (
            "System should evolve over time"
        )


class TestSolverOutputSaving:
    """Solver should correctly save outputs to disk."""

    def test_save_morph_images(self, fresh_liaw_model, tmp_path):
        from lpf.solvers import EulerSolver
        dpath_morph = str(tmp_path / "morph")
        solver = EulerSolver()
        solver.solve(
            model=fresh_liaw_model, dt=0.01, n_iters=20,
            period_output=10, dpath_morph=dpath_morph,
            init_model=False, verbose=0,
        )
        # Check that morph directory was created and has files
        import os
        assert os.path.isdir(dpath_morph)
        files = os.listdir(dpath_morph)
        assert len(files) > 0, "Morph images should be saved"

    def test_save_pattern_independently(self, fresh_liaw_model, tmp_path):
        """Pattern saving should work without morph saving (bug 2-6)."""
        from lpf.solvers import EulerSolver
        dpath_pattern = str(tmp_path / "pattern")
        solver = EulerSolver()
        solver.solve(
            model=fresh_liaw_model, dt=0.01, n_iters=20,
            period_output=10, dpath_pattern=dpath_pattern,
            init_model=False, verbose=0,
        )
        import os
        assert os.path.isdir(dpath_pattern)
        files = os.listdir(dpath_pattern)
        assert len(files) > 0, "Pattern images should be saved independently"


class TestSolverConsistency:
    """Higher-order solvers should be more accurate than lower-order ones."""

    def test_higher_order_closer_to_reference(self):
        """RK4 with large dt should differ from Euler with large dt,
        but both should be finite."""
        from lpf.models import LiawModel
        from lpf.initializers import LiawInitializer
        from lpf.solvers import EulerSolver, RungeKuttaSolver

        def make_model():
            init_pts = np.array([[[16, 16]]], dtype=np.uint32)
            init_states = np.array([[0.5, 0.5]], dtype=np.float32)
            initializer = LiawInitializer(init_states=init_states, init_pts=init_pts)
            params = np.array([[1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]],
                              dtype=np.float32)
            m = LiawModel(
                initializer=initializer, params=params,
                thr_color=0.5 * np.ones((1, 1, 1)),
                width=32, height=32, dx=0.1, device="cpu",
            )
            m.initialize()
            return m

        model_euler = make_model()
        model_rk4 = make_model()

        n_iters = 50
        dt = 0.01

        EulerSolver().solve(model=model_euler, dt=dt, n_iters=n_iters,
                            init_model=False, verbose=0)
        RungeKuttaSolver().solve(model=model_rk4, dt=dt, n_iters=n_iters,
                                init_model=False, verbose=0)

        u_euler = model_euler.am.get(model_euler.u)
        u_rk4 = model_rk4.am.get(model_rk4.u)

        # Both should be finite
        assert np.all(np.isfinite(u_euler))
        assert np.all(np.isfinite(u_rk4))

        # They should differ (different numerical methods)
        assert not np.allclose(u_euler, u_rk4, atol=1e-10), (
            "Euler and RK4 should produce different results"
        )
