"""CPU-only solver tests. Runs on all platforms (Windows, Linux, macOS).

Covers:
  - All solver types produce finite results on CPU
  - Trajectory capture (period_output, waypoints)
  - File I/O (morph, pattern, states saving)
  - Solver consistency (Euler vs RK4 differ)
  - step() returns valid delta
  - Base class CUDA dispatch returns None on CPU (no crash)
  - Constructor param forwarding
  - AB2 state reset between solves
"""

import numpy as np
import pytest
import os


# ---- helpers ----

ALL_SOLVERS = ["EulerSolver", "HeunSolver", "RungeKuttaSolver",
               "RK23Solver", "AdamsBashforth2Solver"]

ALL_MODELS = [
    ("LiawModel", 8, [1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]),
    ("GrayScottModel", 4, [1e-3, 1e-2, 0.04, 0.06]),
    ("BrusselatorModel", 4, [1e-3, 1e-2, 1.0, 3.0]),
    ("FitzHughNagumoModel", 5, [1e-3, 1e-1, 0.01, 0.5, 0.1]),
]


def _get_solver_class(name):
    import lpf.solvers as S
    return getattr(S, name)


def _make_model(model_name="LiawModel", n_params=8,
                sample_params=None, batch_size=1):
    import lpf.models as M
    from lpf.initializers import LiawInitializer

    if sample_params is None:
        sample_params = [1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]

    np.random.seed(42)
    init_pts = np.array([[[16, 16]]] * batch_size, dtype=np.uint32)
    init_states = np.array([[0.5, 0.5]] * batch_size, dtype=np.float32)
    params = np.array([sample_params] * batch_size, dtype=np.float32)

    cls = getattr(M, model_name)
    model = cls(
        initializer=LiawInitializer(init_states=init_states, init_pts=init_pts),
        n_init_pts=1, params=params,
        thr_color=0.5 * np.ones((batch_size, 1, 1)),
        width=32, height=32, dx=0.1, device="cpu",
    )
    model.initialize()
    return model


# ---- all solvers produce finite results ----

class TestAllSolversFinite:
    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_finite_results(self, solver_name):
        model = _make_model()
        cls = _get_solver_class(solver_name)
        cls(dt=0.01, n_iters=20).solve(model, init_model=False, verbose=0)
        u = np.asarray(model.u)
        assert np.all(np.isfinite(u)), f"{solver_name} produced non-finite"


# ---- all solvers × all models ----

class TestSolverModelMatrix:
    @pytest.mark.parametrize("solver_name", ["EulerSolver", "RungeKuttaSolver"])
    @pytest.mark.parametrize("model_name,n_params,sample_params", ALL_MODELS)
    def test_solver_model_combination(self, solver_name, model_name,
                                       n_params, sample_params):
        model = _make_model(model_name, n_params, sample_params)
        cls = _get_solver_class(solver_name)
        cls(dt=0.01, n_iters=20).solve(model, init_model=False, verbose=0)
        u = np.asarray(model.u)
        assert np.all(np.isfinite(u))


# ---- step() ----

class TestStep:
    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_step_returns_finite_nonzero(self, solver_name):
        model = _make_model()
        cls = _get_solver_class(solver_name)
        solver = cls()
        delta = solver.step(model, t=0.0, dt=0.01, y_mesh=model.y_mesh)
        delta_np = np.asarray(delta)
        assert np.all(np.isfinite(delta_np)), f"{solver_name}.step() non-finite"
        assert not np.allclose(delta_np, 0.0), f"{solver_name}.step() all zeros"


# ---- trajectory capture ----

class TestTrajectory:
    def test_trajectory_with_period(self):
        from lpf.solvers import EulerSolver
        model = _make_model()
        trj = EulerSolver(dt=0.01, n_iters=50).solve(
            model, period_output=10, get_trj=True, init_model=False, verbose=0)
        assert trj.shape[0] == 6  # iter 1, 10, 20, 30, 40, 50

    def test_trajectory_with_waypoints(self):
        from lpf.solvers import EulerSolver
        model = _make_model()
        result = EulerSolver(dt=0.01, n_iters=20).solve(
            model, trj_waypoints=[5, 10, 20], init_model=False, verbose=0)
        assert isinstance(result, dict)
        assert result["trj"].shape[0] == 3
        assert result["iters"] == [5, 10, 20]

    def test_return_none_without_get_trj(self):
        from lpf.solvers import EulerSolver
        model = _make_model()
        result = EulerSolver(dt=0.01, n_iters=20).solve(
            model, init_model=False, verbose=0)
        assert result is None

    def test_return_array_with_get_trj(self):
        from lpf.solvers import EulerSolver
        model = _make_model()
        result = EulerSolver(dt=0.01, n_iters=20).solve(
            model, period_output=10, get_trj=True, init_model=False, verbose=0)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 5  # (time, states, batch, height, width)

    def test_return_dict_with_waypoints(self):
        from lpf.solvers import EulerSolver
        model = _make_model()
        result = EulerSolver(dt=0.01, n_iters=20).solve(
            model, trj_waypoints=[5, 10, 20], init_model=False, verbose=0)
        assert isinstance(result, dict)
        assert "trj" in result
        assert "iters" in result
        assert isinstance(result["iters"], list)
        assert isinstance(result["trj"], np.ndarray)

    def test_trajectory_evolves(self):
        from lpf.solvers import EulerSolver
        model = _make_model()
        trj = EulerSolver(dt=0.01, n_iters=100).solve(
            model, period_output=50, get_trj=True, init_model=False, verbose=0)
        assert not np.allclose(trj[0], trj[-1]), "System should evolve"


# ---- file I/O ----

class TestFileIO:
    def test_save_morph(self, tmp_path):
        from lpf.solvers import EulerSolver
        model = _make_model()
        dpath = str(tmp_path / "morph")
        EulerSolver().solve(
            model, dt=0.01, n_iters=20, period_output=10,
            dpath_morph=dpath, init_model=False, verbose=0)
        assert os.path.isdir(dpath)
        files = []
        for root, dirs, fnames in os.walk(dpath):
            files.extend(fnames)
        assert len(files) > 0

    def test_save_states(self, tmp_path):
        from lpf.solvers import EulerSolver
        model = _make_model()
        dpath = str(tmp_path / "states")
        EulerSolver().solve(
            model, dt=0.01, n_iters=20, period_output=10,
            dpath_states=dpath, init_model=False, verbose=0)
        npz_files = []
        for root, dirs, fnames in os.walk(str(tmp_path)):
            npz_files.extend(f for f in fnames if f.endswith('.npz'))
        assert len(npz_files) > 0

    def test_save_pattern_independently(self, tmp_path):
        """Pattern saving should work without morph saving."""
        from lpf.solvers import EulerSolver
        model = _make_model()
        dpath = str(tmp_path / "pattern")
        EulerSolver().solve(
            model, dt=0.01, n_iters=20, period_output=10,
            dpath_pattern=dpath, init_model=False, verbose=0)
        assert os.path.isdir(dpath)
        assert len(os.listdir(dpath)) > 0


# ---- solver consistency ----

class TestConsistency:
    def test_euler_rk4_differ(self):
        from lpf.solvers import EulerSolver, RungeKuttaSolver
        m1 = _make_model()
        m2 = _make_model()
        EulerSolver(dt=0.01, n_iters=50).solve(m1, init_model=False, verbose=0)
        RungeKuttaSolver(dt=0.01, n_iters=50).solve(m2, init_model=False, verbose=0)
        assert not np.allclose(np.asarray(m1.u), np.asarray(m2.u), atol=1e-10)


# ---- base class CUDA dispatch on CPU (no crash) ----

class TestCUDADispatchOnCPU:
    def test_make_cuda_solver_returns_none_on_cpu(self):
        from lpf.solvers import EulerSolver
        solver = EulerSolver()
        model = _make_model()
        # _is_cuda should be False for CPU model
        assert not solver._is_cuda(model)
        # solve should work normally via Python path
        solver.solve(model, dt=0.01, n_iters=5, init_model=False, verbose=0)
        assert np.all(np.isfinite(np.asarray(model.u)))


# ---- constructor param forwarding ----

class TestParamForwarding:
    def test_constructor_dt_n_iters(self):
        from lpf.solvers import EulerSolver
        model = _make_model()
        solver = EulerSolver(dt=0.01, n_iters=10)
        solver.solve(model, init_model=False, verbose=0)
        assert np.all(np.isfinite(np.asarray(model.u)))


# ---- AB2 state reset ----

class TestAB2StateReset:
    def test_consecutive_solves_independent(self):
        from lpf.solvers import AdamsBashforth2Solver
        solver = AdamsBashforth2Solver(dt=0.01, n_iters=20)

        m1 = _make_model()
        solver.solve(m1, init_model=False, verbose=0)
        y1 = np.asarray(m1.u).copy()

        m2 = _make_model()
        solver.solve(m2, init_model=False, verbose=0)
        y2 = np.asarray(m2.u).copy()

        np.testing.assert_allclose(y1, y2, atol=1e-7,
            err_msg="Consecutive AB2 solves should produce identical results")
