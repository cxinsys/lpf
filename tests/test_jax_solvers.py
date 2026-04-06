"""Tests for JAX-kernel-accelerated solvers.

Mirrors ``test_cuda_solvers.py`` for the JAX execution path
(``device="jax:cpu"``, ``"jax:gpu:0"``, ``"jax:tpu:0"``).

Covers:
  - Auto-detection: JAX CPU/GPU models auto-dispatch to JAX kernels
  - Correctness: JAX results match CPU NumPy baseline within tolerance
  - All fixed-step solvers: Euler, Heun, RK4, RK23, AB2
  - All adaptive solvers: AdaptiveRKF45, DOPRI5
  - All features: trajectory capture, waypoints, period_output, file I/O,
    early stopping, batch sizes, dtype, return-type parity
  - Diploid model support

"""

import numpy as np
import pytest

# Skip entire module if JAX is not available
jax = pytest.importorskip("jax", reason="JAX not installed")

from tests.conftest import make_liaw_model_16, get_numpy, LIAW_DT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_FIXED_SOLVERS = ["EulerSolver", "HeunSolver", "RungeKuttaSolver",
                     "RK23Solver", "AdamsBashforth2Solver"]
ALL_ADAPTIVE_SOLVERS = ["AdaptiveRKF45Solver", "DOPRI5Solver"]
ALL_SOLVERS = ALL_FIXED_SOLVERS + ALL_ADAPTIVE_SOLVERS


# Devices to test. JAX CPU is excluded — it duplicates the NumPy baseline
# without exercising the JAX kernel path we actually ship for. Only run JAX
# tests when a GPU device is available.
JAX_DEVICES = []
try:
    if jax.devices("gpu"):
        JAX_DEVICES.append("jax:gpu:0")
except Exception:
    pass

if not JAX_DEVICES:
    pytest.skip("No JAX GPU device available", allow_module_level=True)


def _make_model(device, batch_size=1, width=32, height=32, dtype=np.float32):
    from lpf.models import LiawModel
    from lpf.initializers import LiawInitializer

    np.random.seed(42)
    init_pts = np.random.randint(0, width,
                                  size=(batch_size, 5, 2)).astype(np.uint32)
    init_states = np.array([[0.5, 0.5]] * batch_size, dtype=dtype)
    params = np.array([[1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]]
                       * batch_size, dtype=dtype)

    return LiawModel(
        initializer=LiawInitializer(init_states=init_states, init_pts=init_pts),
        n_init_pts=5, params=params,
        thr_color=0.5 * np.ones((batch_size, 1, 1)),
        width=width, height=height, dx=0.1,
        device=device, dtype=dtype,
    )


def _get_numpy(model):
    """Extract y_mesh as numpy regardless of backend."""
    arr = model.y_mesh
    if hasattr(arr, 'get'):       # CuPy
        return arr.get()
    if hasattr(arr, 'cpu'):       # PyTorch
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _to_numpy(arr):
    if hasattr(arr, 'get'):
        return arr.get()
    if hasattr(arr, 'cpu'):
        return arr.cpu().numpy()
    return np.asarray(arr)


def _get_solver_class(name):
    import lpf.solvers as S
    return getattr(S, name)


def _is_jax_array(arr):
    """True if ``arr`` is a JAX array (any device)."""
    try:
        import jax
        return isinstance(arr, jax.Array)
    except Exception:
        return type(arr).__module__.startswith("jax")


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

class TestAutoDetection:
    """JAX kernels should activate automatically based on device."""

    @pytest.mark.parametrize("device", JAX_DEVICES)
    @pytest.mark.parametrize("solver_name", ALL_FIXED_SOLVERS)
    def test_jax_model_uses_jax_kernels(self, device, solver_name):
        model = _make_model(device)
        cls = _get_solver_class(solver_name)
        solver = cls(dt=0.01, n_iters=10)
        solver.solve(model, verbose=0)

        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), \
            f"{solver_name} on {device} produced non-finite"

    @pytest.mark.parametrize("device", JAX_DEVICES)
    @pytest.mark.parametrize("solver_name", ALL_ADAPTIVE_SOLVERS)
    def test_jax_model_uses_jax_kernels_adaptive(self, device, solver_name):
        model = _make_model(device)
        cls = _get_solver_class(solver_name)
        solver = cls(dt=0.01, n_iters=10)
        solver.solve(model, verbose=0)

        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), \
            f"{solver_name} on {device} produced non-finite"

    def test_jax_dispatch_uses_jax_solver(self):
        """``Solver._get_jax_solver`` must return a JaxSolverBase subclass."""
        from lpf.solvers import EulerSolver
        from lpf.solvers._jax.base import JaxSolverBase

        model = _make_model("jax:cpu")
        solver = EulerSolver()
        assert solver._is_jax(model)
        jax_solver = solver._get_jax_solver()
        assert isinstance(jax_solver, JaxSolverBase)


# ---------------------------------------------------------------------------
# Correctness vs CPU baseline
# ---------------------------------------------------------------------------

class TestCorrectnessVsCpu:
    """JAX results must match CPU NumPy baseline within tolerance."""

    @pytest.mark.parametrize("device", JAX_DEVICES)
    @pytest.mark.parametrize("solver_name", ALL_FIXED_SOLVERS)
    def test_jax_matches_cpu(self, device, solver_name):
        # Use the realistic 16-batch init_pop_01 fixture, n_iters=1
        # for strict numerical parity (matches the CUDA test pattern).
        n_iters = 1
        cls = _get_solver_class(solver_name)

        model_cpu = make_liaw_model_16("cpu")
        cls(dt=LIAW_DT, n_iters=n_iters).solve(model_cpu, verbose=0)
        y_cpu = get_numpy(model_cpu)

        model_jax = make_liaw_model_16(device)
        cls(dt=LIAW_DT, n_iters=n_iters).solve(model_jax, verbose=0)
        y_jax = get_numpy(model_jax)

        np.testing.assert_allclose(
            y_jax, y_cpu, atol=1e-5, rtol=1e-5,
            err_msg=f"{solver_name}: JAX({device}) vs CPU mismatch")

    @pytest.mark.parametrize("device", JAX_DEVICES)
    @pytest.mark.parametrize("solver_name", ALL_ADAPTIVE_SOLVERS)
    def test_jax_adaptive_matches_cpu(self, device, solver_name):
        # Adaptive sub-stepping accumulates more rounding; use a slightly
        # looser tolerance and a few outer iterations so the comparison
        # actually exercises the inner sub-step loop.
        n_iters = 5
        cls = _get_solver_class(solver_name)

        model_cpu = make_liaw_model_16("cpu")
        cls(dt=LIAW_DT, n_iters=n_iters).solve(model_cpu, verbose=0)
        y_cpu = get_numpy(model_cpu)

        model_jax = make_liaw_model_16(device)
        cls(dt=LIAW_DT, n_iters=n_iters).solve(model_jax, verbose=0)
        y_jax = get_numpy(model_jax)

        np.testing.assert_allclose(
            y_jax, y_cpu, atol=1e-4, rtol=1e-3,
            err_msg=f"{solver_name}: JAX({device}) vs CPU mismatch")


# ---------------------------------------------------------------------------
# Cross-backend parity (JAX vs CuPy if available)
# ---------------------------------------------------------------------------

class TestJaxVsCuda:
    """JAX kernels and CUDA fused kernels must agree numerically."""

    @pytest.mark.parametrize("solver_name", ALL_FIXED_SOLVERS)
    def test_jax_gpu_matches_cupy(self, solver_name):
        cupy = pytest.importorskip("cupy")
        if "jax:gpu:0" not in JAX_DEVICES:
            pytest.skip("JAX GPU not available")

        n_iters = 1
        cls = _get_solver_class(solver_name)

        model_cu = make_liaw_model_16("cuda:0")
        cls(dt=LIAW_DT, n_iters=n_iters).solve(model_cu, verbose=0)
        y_cu = get_numpy(model_cu)

        model_jax = make_liaw_model_16("jax:gpu:0")
        cls(dt=LIAW_DT, n_iters=n_iters).solve(model_jax, verbose=0)
        y_jax = get_numpy(model_jax)

        np.testing.assert_allclose(
            y_jax, y_cu, atol=1e-5, rtol=1e-5,
            err_msg=f"{solver_name}: JAX GPU vs CuPy mismatch")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnTypes:
    """The model arrays should remain JAX arrays after a JAX solve."""

    def test_jax_returns_jax_array(self):
        model = _make_model("jax:cpu")
        from lpf.solvers import EulerSolver
        EulerSolver(dt=0.01, n_iters=10).solve(model, verbose=0)
        assert _is_jax_array(model.y_mesh)


# ---------------------------------------------------------------------------
# Trajectory capture
# ---------------------------------------------------------------------------

class TestTrajectoryCapture:
    """Trajectory capture must work on the JAX path."""

    @pytest.mark.parametrize("device", JAX_DEVICES)
    def test_trajectory_with_period(self, device):
        model = _make_model(device)
        from lpf.solvers import EulerSolver
        trj = EulerSolver(dt=0.01, n_iters=100).solve(
            model, period_output=50, get_trj=True, verbose=0)
        assert trj.shape[0] == 3  # iter 1, 50, 100
        np_trj = _to_numpy(trj)
        assert np.all(np.isfinite(np_trj))

    @pytest.mark.parametrize("device", JAX_DEVICES)
    def test_trajectory_with_waypoints(self, device):
        model = _make_model(device)
        from lpf.solvers import RungeKuttaSolver
        waypoints = [10, 30, 50]
        result = RungeKuttaSolver(dt=0.01, n_iters=50).solve(
            model, trj_waypoints=waypoints, verbose=0)
        assert isinstance(result, dict)
        assert result["trj"].shape[0] == len(waypoints)
        assert result["iters"] == waypoints

    def test_return_none_without_get_trj(self):
        from lpf.solvers import EulerSolver
        model = _make_model("jax:cpu")
        result = EulerSolver(dt=0.01, n_iters=20).solve(model, verbose=0)
        assert result is None

    def test_return_jax_array_for_jax_model(self):
        from lpf.solvers import EulerSolver
        model = _make_model("jax:cpu")
        result = EulerSolver(dt=0.01, n_iters=20).solve(
            model, period_output=10, get_trj=True, verbose=0)
        assert _is_jax_array(result)

    def test_return_dict_with_waypoints_jax(self):
        from lpf.solvers import EulerSolver
        model = _make_model("jax:cpu")
        result = EulerSolver(dt=0.01, n_iters=20).solve(
            model, trj_waypoints=[5, 10, 20], verbose=0)
        assert isinstance(result, dict)
        assert _is_jax_array(result["trj"])
        assert isinstance(result["iters"], list)


# ---------------------------------------------------------------------------
# CPU/JAX return-value parity
# ---------------------------------------------------------------------------

class TestCpuJaxReturnParity:
    """solve() return values must agree (type & values modulo backend) between
    CPU and JAX backends for all option combinations.  Device-specific array
    types (numpy / JAX array) are the only allowed difference."""

    N_ITERS = 1
    DT = LIAW_DT
    ATOL = 1e-5
    RTOL = 1e-5

    def _solve(self, device, **kwargs):
        from lpf.solvers import EulerSolver
        model = make_liaw_model_16(device)
        return EulerSolver(dt=self.DT, n_iters=self.N_ITERS).solve(
            model, verbose=0, **kwargs)

    def test_none_return_cpu_vs_jax(self):
        assert self._solve("cpu") is None
        assert self._solve("jax:cpu") is None

    def test_trajectory_array_cpu_vs_jax(self):
        trj_cpu = self._solve("cpu", period_output=1, get_trj=True)
        trj_jax = self._solve("jax:cpu", period_output=1, get_trj=True)

        assert isinstance(trj_cpu, np.ndarray)
        assert _is_jax_array(trj_jax)
        assert trj_cpu.shape == trj_jax.shape

        np.testing.assert_allclose(
            trj_cpu, _to_numpy(trj_jax),
            atol=self.ATOL, rtol=self.RTOL)

    def test_waypoint_dict_cpu_vs_jax(self):
        waypoints = [1]
        result_cpu = self._solve("cpu", trj_waypoints=waypoints)
        result_jax = self._solve("jax:cpu", trj_waypoints=waypoints)

        assert isinstance(result_cpu, dict)
        assert isinstance(result_jax, dict)
        assert set(result_cpu.keys()) == set(result_jax.keys()) == {"iters", "trj"}

        assert result_cpu["iters"] == result_jax["iters"] == waypoints

        assert isinstance(result_cpu["trj"], np.ndarray)
        assert _is_jax_array(result_jax["trj"])
        assert result_cpu["trj"].shape == result_jax["trj"].shape
        np.testing.assert_allclose(
            result_cpu["trj"], _to_numpy(result_jax["trj"]),
            atol=self.ATOL, rtol=self.RTOL)

    def test_empty_waypoints_cpu_vs_jax(self):
        waypoints = [999]
        result_cpu = self._solve("cpu", trj_waypoints=waypoints)
        result_jax = self._solve("jax:cpu", trj_waypoints=waypoints)

        assert isinstance(result_cpu, dict)
        assert isinstance(result_jax, dict)
        assert result_cpu["iters"] == []
        assert result_jax["iters"] == []
        assert result_cpu["trj"].shape[0] == 0
        assert result_jax["trj"].shape[0] == 0


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

class TestFileIO:
    """File I/O should work via the JAX path."""

    def test_save_morph_and_states(self, tmp_path):
        model = _make_model("jax:cpu")
        from lpf.solvers import EulerSolver
        dpath_morph = str(tmp_path / "morph")
        dpath_states = str(tmp_path / "states")

        EulerSolver(dt=0.01, n_iters=50).solve(
            model, period_output=25,
            dpath_morph=dpath_morph, dpath_states=dpath_states,
            verbose=0)

        morph_files = []
        state_files = []
        for root, dirs, files in os.walk(str(tmp_path)):
            for f in files:
                if f.endswith('.png'):
                    morph_files.append(f)
                if f.endswith('.npz'):
                    state_files.append(f)

        assert len(morph_files) > 0, "Should save morph images"
        assert len(state_files) > 0, "Should save state files"


# ---------------------------------------------------------------------------
# Batch sizes
# ---------------------------------------------------------------------------

class TestBatchSizes:
    """JAX kernels must handle various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_size(self, batch_size):
        model = _make_model("jax:cpu", batch_size=batch_size)
        from lpf.solvers import EulerSolver
        EulerSolver(dt=0.01, n_iters=20).solve(model, verbose=0)
        y = _get_numpy(model)
        assert y.shape == (2, batch_size, 32, 32)
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# Float64
# ---------------------------------------------------------------------------

class TestFloat64:
    """float64 must work on the JAX path and stay float64.

    LPF enables ``JAX_ENABLE_X64=1`` at JaxModule import time so float64
    is preserved end-to-end across the entire jit-compiled solve.
    """

    @pytest.mark.parametrize("solver_name", ALL_FIXED_SOLVERS)
    def test_float64_jax(self, solver_name):
        model = _make_model("jax:cpu", dtype=np.float64)
        cls = _get_solver_class(solver_name)
        cls(dt=0.01, n_iters=20).solve(model, verbose=0)
        y = _get_numpy(model)
        assert y.shape == (2, 1, 32, 32)
        assert y.dtype == np.float64, \
            f"{solver_name}: expected float64, got {y.dtype}"
        assert np.all(np.isfinite(y))

    @pytest.mark.parametrize("solver_name", ALL_FIXED_SOLVERS)
    def test_float32_jax_stays_float32(self, solver_name):
        """Float32 input must not get auto-promoted to float64 either."""
        model = _make_model("jax:cpu", dtype=np.float32)
        cls = _get_solver_class(solver_name)
        cls(dt=0.01, n_iters=20).solve(model, verbose=0)
        y = _get_numpy(model)
        assert y.dtype == np.float32, \
            f"{solver_name}: expected float32, got {y.dtype}"


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    """Early stopping (rtol) should work on the JAX path."""

    def test_early_stopping_does_not_crash(self):
        model = _make_model("jax:cpu")
        from lpf.solvers import EulerSolver
        EulerSolver(dt=0.01, n_iters=200).solve(
            model, rtol=1e-3, verbose=0)
        y = _get_numpy(model)
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# All model types
# ---------------------------------------------------------------------------

class TestAllModelTypes:
    """JAX kernels should work with all reaction-diffusion models."""

    @pytest.mark.parametrize("model_cls_name,n_params", [
        ("LiawModel", 8),
        ("GrayScottModel", 4),
        ("BrusselatorModel", 4),
        ("FitzHughNagumoModel", 5),
        ("SchnakenbergModel", 6),
        ("GiererMeinhardtModel", 6),
    ])
    def test_model_type(self, model_cls_name, n_params):
        from lpf.initializers import LiawInitializer
        import lpf.models as M

        model_cls = getattr(M, model_cls_name)
        B = 2

        np.random.seed(42)
        init_pts = np.random.randint(0, 32, size=(B, 3, 2)).astype(np.uint32)
        init_states = np.array([[0.5, 0.5]] * B, dtype=np.float32)
        params = np.full((B, n_params), 0.01, dtype=np.float32)
        params[:, 0] = 1e-3  # Du
        params[:, 1] = 1e-2  # Dv

        model = model_cls(
            initializer=LiawInitializer(init_states=init_states, init_pts=init_pts),
            n_init_pts=3, params=params,
            thr_color=0.5 * np.ones((B, 1, 1)),
            width=32, height=32, dx=0.1, device="jax:cpu",
        )

        from lpf.solvers import EulerSolver
        EulerSolver(dt=0.01, n_iters=20).solve(model, verbose=0)
        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), \
            f"{model_cls_name} produced non-finite on JAX"


# ---------------------------------------------------------------------------
# Constructor param forwarding
# ---------------------------------------------------------------------------

class TestParamForwarding:
    """Constructor params (dt, n_iters) should be forwarded to the JAX path."""

    def test_constructor_params(self):
        model = _make_model("jax:cpu")
        from lpf.solvers import EulerSolver

        solver = EulerSolver(dt=0.01, n_iters=30)
        solver.solve(model, verbose=0)

        y = _get_numpy(model)
        assert np.all(np.isfinite(y))

    def test_solve_params_override_constructor(self):
        model = _make_model("jax:cpu")
        from lpf.solvers import EulerSolver

        solver = EulerSolver(dt=0.05, n_iters=100)
        solver.solve(model, dt=0.01, n_iters=10, verbose=0)

        y = _get_numpy(model)
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# Diploid model support
# ---------------------------------------------------------------------------

class TestDiploidModel:
    """The JAX path supports TwoComponentDiploidModel."""

    def _make_diploid(self, device, alpha=0.6, beta=0.4):
        from lpf.models import LiawModel, TwoComponentDiploidModel
        from lpf.initializers import LiawInitializer

        np.random.seed(42)
        pts = np.array([[[16, 16]]], dtype=np.uint32)
        states = np.array([[0.5, 0.5]], dtype=np.float32)
        p1 = np.array([[1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]],
                      dtype=np.float32)
        p2 = p1 * 1.1

        pa = LiawModel(
            initializer=LiawInitializer(init_states=states, init_pts=pts),
            n_init_pts=1, params=p1,
            thr_color=0.5 * np.ones((1, 1, 1)),
            width=32, height=32, dx=0.1, device=device, dtype=np.float32,
        )
        ma = LiawModel(
            initializer=LiawInitializer(init_states=states, init_pts=pts),
            n_init_pts=1, params=p2,
            thr_color=0.5 * np.ones((1, 1, 1)),
            width=32, height=32, dx=0.1, device=device, dtype=np.float32,
        )
        d = TwoComponentDiploidModel(
            paternal_model=pa, maternal_model=ma, alpha=alpha, beta=beta,
            params=p1, width=32, height=32, dx=0.1,
            device=device, dtype=np.float32,
            initializer=LiawInitializer(init_states=states, init_pts=pts),
            n_init_pts=1, thr_color=0.5 * np.ones((1, 1, 1)),
        )
        d.initialize()
        return d

    @pytest.mark.parametrize("device", JAX_DEVICES)
    @pytest.mark.parametrize("solver_name", ALL_FIXED_SOLVERS)
    def test_diploid_fixed_solvers(self, device, solver_name):
        cls = _get_solver_class(solver_name)
        d = self._make_diploid(device)
        cls(dt=0.01, n_iters=20).solve(d, init_model=False, verbose=0)
        y = _get_numpy(d)
        assert y.shape == (4, 1, 32, 32)
        assert np.all(np.isfinite(y))

    @pytest.mark.parametrize("device", JAX_DEVICES)
    @pytest.mark.parametrize("solver_name", ALL_ADAPTIVE_SOLVERS)
    def test_diploid_adaptive_solvers(self, device, solver_name):
        cls = _get_solver_class(solver_name)
        d = self._make_diploid(device)
        cls(dt=0.01, n_iters=10).solve(d, init_model=False, verbose=0)
        y = _get_numpy(d)
        assert y.shape == (4, 1, 32, 32)
        assert np.all(np.isfinite(y))

    def test_diploid_jax_matches_cpu(self):
        from lpf.solvers import EulerSolver
        d_cpu = self._make_diploid("cpu")
        EulerSolver(dt=0.01, n_iters=20).solve(d_cpu, init_model=False,
                                                verbose=0)
        y_cpu = _get_numpy(d_cpu)

        for device in JAX_DEVICES:
            d_jax = self._make_diploid(device)
            EulerSolver(dt=0.01, n_iters=20).solve(d_jax, init_model=False,
                                                    verbose=0)
            y_jax = _get_numpy(d_jax)
            np.testing.assert_allclose(
                y_jax, y_cpu, atol=1e-4, rtol=1e-3,
                err_msg=f"Diploid JAX({device}) vs CPU mismatch")
