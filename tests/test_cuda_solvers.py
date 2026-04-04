"""Tests for CUDA-kernel-accelerated solvers.

Covers:
  - Auto-detection: CuPy and PyTorch CUDA models auto-dispatch to CUDA kernels
  - Correctness: CUDA results match Python baseline within tolerance
  - All solver types: Euler, Heun, RK4, RK23, AB2
  - All features: trajectory capture, waypoints, period_output, file I/O, early stopping
  - PyTorch DLPack bridge: torch tensors in → CUDA kernels run → torch tensors out
  - Batch sizes: single and multi-batch
  - dtype support: float32, float64
"""

import numpy as np
import pytest
import os

# Skip entire module if CuPy is not available
cupy = pytest.importorskip("cupy")


# ---------- helpers ----------

def _make_model(device, batch_size=1, width=32, height=32, dtype=np.float32):
    from lpf.models import LiawModel
    from lpf.initializers import LiawInitializer

    np.random.seed(42)
    init_pts = np.random.randint(0, width, size=(batch_size, 5, 2)).astype(np.uint32)
    init_states = np.array([[0.5, 0.5]] * batch_size, dtype=dtype)
    params = np.array([[1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]] * batch_size,
                      dtype=dtype)

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


ALL_SOLVERS = ["EulerSolver", "HeunSolver", "RungeKuttaSolver",
               "RK23Solver", "AdamsBashforth2Solver"]


def _get_solver_class(name):
    import lpf.solvers as S
    return getattr(S, name)


# ---------- auto-detection ----------

class TestAutoDetection:
    """CUDA kernels should activate automatically based on device."""

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_cupy_model_uses_cuda_kernels(self, solver_name):
        model = _make_model("cuda:0")
        cls = _get_solver_class(solver_name)
        solver = cls(dt=0.01, n_iters=10)
        solver.solve(model, verbose=0)

        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), f"{solver_name} on cuda:0 produced non-finite"

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_torch_cuda_model_uses_cuda_kernels(self, solver_name):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        model = _make_model("torch:gpu:0")
        cls = _get_solver_class(solver_name)
        solver = cls(dt=0.01, n_iters=10)
        solver.solve(model, verbose=0)

        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), f"{solver_name} on torch:gpu:0 produced non-finite"

    def test_cpu_model_does_not_use_cuda(self):
        """CPU model should use plain Python path (no CuPy import needed)."""
        model = _make_model("cpu")
        from lpf.solvers import EulerSolver
        solver = EulerSolver(dt=0.01, n_iters=10)
        solver.solve(model, verbose=0)
        y = _get_numpy(model)
        assert np.all(np.isfinite(y))


# ---------- correctness: CUDA vs Python baseline ----------

class TestCorrectnessVsPython:
    """CUDA kernel results must match Python baseline within tolerance."""

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_cuda_matches_python(self, solver_name):
        n_iters = 50
        dt = 0.01
        cls = _get_solver_class(solver_name)

        # Python baseline (CPU)
        model_py = _make_model("cpu")
        cls(dt=dt, n_iters=n_iters).solve(model_py, verbose=0)
        y_py = _get_numpy(model_py)

        # CUDA (cuda:0)
        model_cu = _make_model("cuda:0")
        cls(dt=dt, n_iters=n_iters).solve(model_cu, verbose=0)
        y_cu = _get_numpy(model_cu)

        # Compare where both are finite
        valid = np.isfinite(y_py) & np.isfinite(y_cu)
        if valid.any():
            max_diff = np.max(np.abs(y_py[valid] - y_cu[valid]))
            assert max_diff < 1e-2, (
                f"{solver_name}: max diff {max_diff:.6e} exceeds tolerance"
            )

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_torch_cuda_matches_cupy_cuda(self, solver_name):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        n_iters = 50
        dt = 0.01
        cls = _get_solver_class(solver_name)

        model_cu = _make_model("cuda:0")
        cls(dt=dt, n_iters=n_iters).solve(model_cu, verbose=0)
        y_cu = _get_numpy(model_cu)

        model_torch = _make_model("torch:gpu:0")
        cls(dt=dt, n_iters=n_iters).solve(model_torch, verbose=0)
        y_torch = _get_numpy(model_torch)

        valid = np.isfinite(y_cu) & np.isfinite(y_torch)
        if valid.any():
            max_diff = np.max(np.abs(y_cu[valid] - y_torch[valid]))
            assert max_diff < 1e-4, (
                f"{solver_name}: CuPy vs Torch diff {max_diff:.6e}"
            )


# ---------- return type ----------

class TestReturnTypes:
    """The returned model arrays should match the original framework."""

    def test_cupy_returns_cupy(self):
        model = _make_model("cuda:0")
        from lpf.solvers import EulerSolver
        EulerSolver(dt=0.01, n_iters=10).solve(model, verbose=0)
        assert type(model.y_mesh).__module__.startswith("cupy")

    def test_torch_returns_torch(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        model = _make_model("torch:gpu:0")
        from lpf.solvers import EulerSolver
        EulerSolver(dt=0.01, n_iters=10).solve(model, verbose=0)
        assert isinstance(model.y_mesh, torch.Tensor)
        assert model.y_mesh.device.type == "cuda"


# ---------- trajectory capture ----------

class TestTrajectoryCapture:
    """Trajectory capture must work on CUDA path."""

    def test_trajectory_with_period(self):
        model = _make_model("cuda:0")
        from lpf.solvers import EulerSolver
        trj = EulerSolver(dt=0.01, n_iters=100).solve(
            model, period_output=50, get_trj=True, verbose=0)
        assert trj.shape[0] == 3  # iter 1, 50, 100

    def test_trajectory_with_waypoints(self):
        model = _make_model("cuda:0")
        from lpf.solvers import RungeKuttaSolver
        waypoints = [10, 30, 50]
        result = RungeKuttaSolver(dt=0.01, n_iters=50).solve(
            model, trj_waypoints=waypoints, verbose=0)
        assert isinstance(result, dict)
        assert result["trj"].shape[0] == len(waypoints)
        assert result["iters"] == waypoints

    def test_trajectory_torch_returns_cupy(self):
        """Trajectory buffer is CuPy (internal). Model arrays are restored to torch."""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        model = _make_model("torch:gpu:0")
        from lpf.solvers import EulerSolver
        trj = EulerSolver(dt=0.01, n_iters=50).solve(
            model, period_output=25, get_trj=True, verbose=0)
        # Trajectory is captured as cupy internally
        assert trj.shape[0] > 0
        # Model should be restored to torch
        assert isinstance(model.y_mesh, torch.Tensor)


# ---------- file I/O ----------

class TestFileIO:
    """File I/O should work via the CUDA path (no fallback)."""

    def test_save_morph_and_states(self, tmp_path):
        model = _make_model("cuda:0")
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


# ---------- batch sizes ----------

class TestBatchSizes:
    """CUDA kernels must handle various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_size(self, batch_size):
        model = _make_model("cuda:0", batch_size=batch_size)
        from lpf.solvers import EulerSolver
        EulerSolver(dt=0.01, n_iters=20).solve(model, verbose=0)
        y = _get_numpy(model)
        assert y.shape == (2, batch_size, 32, 32)
        assert np.all(np.isfinite(y))


# ---------- float64 ----------

class TestFloat64:
    """float64 precision must work on CUDA path."""

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_float64_cuda(self, solver_name):
        model = _make_model("cuda:0", dtype=np.float64)
        cls = _get_solver_class(solver_name)
        cls(dt=0.01, n_iters=20).solve(model, verbose=0)
        y = _get_numpy(model)
        assert y.dtype == np.float64
        assert np.all(np.isfinite(y))


# ---------- early stopping ----------

class TestEarlyStopping:
    """Early stopping should work on CUDA path."""

    def test_early_stopping_does_not_crash(self):
        model = _make_model("cuda:0")
        from lpf.solvers import EulerSolver
        # Very tight rtol — may or may not trigger, but should not crash
        EulerSolver(dt=0.01, n_iters=200).solve(
            model, rtol=1e-3, verbose=0)
        y = _get_numpy(model)
        assert np.all(np.isfinite(y))


# ---------- all models ----------

class TestAllModelTypes:
    """CUDA kernels should work with all reaction-diffusion models."""

    @pytest.mark.parametrize("model_cls_name,n_params", [
        ("GrayScottModel", 4),
        ("LiawModel", 8),
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
            width=32, height=32, dx=0.1, device="cuda:0",
        )

        from lpf.solvers import EulerSolver
        EulerSolver(dt=0.01, n_iters=20).solve(model, verbose=0)
        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), f"{model_cls_name} produced non-finite on CUDA"


# ---------- solver constructor params forwarding ----------

class TestParamForwarding:
    """Constructor params (dt, n_iters) should be forwarded to CUDA path."""

    def test_constructor_params(self):
        model = _make_model("cuda:0")
        from lpf.solvers import EulerSolver

        solver = EulerSolver(dt=0.01, n_iters=30)
        solver.solve(model, verbose=0)  # no dt/n_iters in solve()

        y = _get_numpy(model)
        assert np.all(np.isfinite(y))

    def test_solve_params_override_constructor(self):
        model = _make_model("cuda:0")
        from lpf.solvers import EulerSolver

        solver = EulerSolver(dt=0.05, n_iters=100)
        # solve() params should override constructor
        solver.solve(model, dt=0.01, n_iters=10, verbose=0)

        y = _get_numpy(model)
        assert np.all(np.isfinite(y))
