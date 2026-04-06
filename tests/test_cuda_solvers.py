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
from types import SimpleNamespace

# Skip entire module if CuPy is not available
cupy = pytest.importorskip("cupy")

from tests.conftest import make_liaw_model_16, get_numpy, LIAW_DT


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
    """CUDA kernel results must match Python baseline within tolerance.
    Uses 16-batch init_pop_01 parameters for realistic coverage.
    """

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_cuda_matches_python(self, solver_name):
        # Use n_iters=1 for strict parity; diverse init_pop_01 params
        # can diverge numerically at higher iterations due to stiffness.
        n_iters = 1
        cls = _get_solver_class(solver_name)

        # Python baseline (CPU)
        model_py = make_liaw_model_16("cpu")
        cls(dt=LIAW_DT, n_iters=n_iters).solve(model_py, verbose=0)
        y_py = get_numpy(model_py)

        # CUDA (cuda:0)
        model_cu = make_liaw_model_16("cuda:0")
        cls(dt=LIAW_DT, n_iters=n_iters).solve(model_cu, verbose=0)
        y_cu = get_numpy(model_cu)

        np.testing.assert_allclose(y_cu, y_py, atol=1e-5, rtol=1e-5,
                                   err_msg=f"{solver_name}: CUDA vs CPU mismatch")

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_torch_cuda_matches_cupy_cuda(self, solver_name):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        n_iters = 1
        cls = _get_solver_class(solver_name)

        model_cu = make_liaw_model_16("cuda:0")
        cls(dt=LIAW_DT, n_iters=n_iters).solve(model_cu, verbose=0)
        y_cu = get_numpy(model_cu)

        model_torch = make_liaw_model_16("torch:gpu:0")
        cls(dt=LIAW_DT, n_iters=n_iters).solve(model_torch, verbose=0)
        y_torch = get_numpy(model_torch)

        np.testing.assert_allclose(y_torch, y_cu, atol=1e-5, rtol=1e-5,
                                   err_msg=f"{solver_name}: CuPy vs Torch mismatch")


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

    def test_return_none_without_get_trj(self):
        """solve() without get_trj should return None on CUDA path."""
        from lpf.solvers import EulerSolver
        model = _make_model("cuda:0")
        result = EulerSolver(dt=0.01, n_iters=20).solve(model, verbose=0)
        assert result is None

    def test_return_cupy_array_for_cupy_model(self):
        """CuPy model should return CuPy array trajectory."""
        from lpf.solvers import EulerSolver
        model = _make_model("cuda:0")
        result = EulerSolver(dt=0.01, n_iters=20).solve(
            model, period_output=10, get_trj=True, verbose=0)
        assert isinstance(result, cupy.ndarray)

    def test_return_dict_with_waypoints_cupy(self):
        """CuPy model with waypoints should return dict with CuPy array."""
        from lpf.solvers import EulerSolver
        model = _make_model("cuda:0")
        result = EulerSolver(dt=0.01, n_iters=20).solve(
            model, trj_waypoints=[5, 10, 20], verbose=0)
        assert isinstance(result, dict)
        assert isinstance(result["trj"], cupy.ndarray)
        assert isinstance(result["iters"], list)

    def test_return_torch_tensor_for_torch_model(self):
        """Torch model should return torch.Tensor trajectory."""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        from lpf.solvers import EulerSolver
        model = _make_model("torch:gpu:0")
        trj = EulerSolver(dt=0.01, n_iters=50).solve(
            model, period_output=25, get_trj=True, verbose=0)
        assert isinstance(trj, torch.Tensor)
        assert trj.device.type == "cuda"

    def test_return_dict_with_waypoints_torch(self):
        """Torch model with waypoints should return dict with torch.Tensor."""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        from lpf.solvers import EulerSolver
        model = _make_model("torch:gpu:0")
        result = EulerSolver(dt=0.01, n_iters=20).solve(
            model, trj_waypoints=[5, 10, 20], verbose=0)
        assert isinstance(result, dict)
        assert isinstance(result["trj"], torch.Tensor)
        assert result["trj"].device.type == "cuda"
        assert isinstance(result["iters"], list)

    def test_native_matches_python_with_nonzero_iter_begin(self, monkeypatch):
        import lpf.solvers._cuda.base as cuda_base
        native = cuda_base._get_native()
        if not native.is_available():
            pytest.skip("Native AOT solver not available")

        from lpf.solvers import EulerSolver

        model_native = _make_model("cuda:0", width=16, height=16)
        trj_native = EulerSolver(dt=0.01, n_iters=3).solve(
            model_native,
            iter_begin=5,
            period_output=2,
            get_trj=True,
            verbose=0,
        )

        monkeypatch.setattr(
            cuda_base,
            "_native_solver",
            SimpleNamespace(is_available=lambda: False),
        )
        model_python = _make_model("cuda:0", width=16, height=16)
        trj_python = EulerSolver(dt=0.01, n_iters=3).solve(
            model_python,
            iter_begin=5,
            period_output=2,
            get_trj=True,
            verbose=0,
        )

        np.testing.assert_allclose(trj_native.get(), trj_python.get(), atol=1e-6, rtol=1e-6)


# ---------- CPU/CUDA return-value parity ----------

def _to_numpy(arr):
    """Convert CuPy array, torch.Tensor, or numpy array to numpy."""
    if hasattr(arr, 'get'):        # CuPy
        return arr.get()
    if hasattr(arr, 'cpu'):        # torch
        return arr.cpu().numpy()
    return np.asarray(arr)


class TestCpuCudaReturnParity:
    """solve() return values must agree (type & values) between CPU and CUDA
    backends for all option combinations. Device-specific array types
    (numpy / CuPy / torch.Tensor) are the only allowed difference."""

    N_ITERS = 1  # strict numerical parity at n_iters=1 (same as TestCorrectnessVsPython)
    DT = LIAW_DT
    ATOL = 1e-5
    RTOL = 1e-5

    def _solve(self, device, **kwargs):
        from lpf.solvers import EulerSolver
        model = make_liaw_model_16(device)
        return EulerSolver(dt=self.DT, n_iters=self.N_ITERS).solve(
            model, verbose=0, **kwargs)

    # ---- get_trj=False ----

    def test_none_return_cpu_vs_cupy(self):
        assert self._solve("cpu") is None
        assert self._solve("cuda:0") is None

    def test_none_return_cpu_vs_torch(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")
        assert self._solve("cpu") is None
        assert self._solve("torch:gpu:0") is None

    # ---- get_trj=True with period_output ----

    def test_trajectory_array_cpu_vs_cupy(self):
        trj_cpu = self._solve("cpu", period_output=1, get_trj=True)
        trj_cu = self._solve("cuda:0", period_output=1, get_trj=True)

        # Type follows device
        assert isinstance(trj_cpu, np.ndarray)
        assert isinstance(trj_cu, cupy.ndarray)

        # Shape equal
        assert trj_cpu.shape == trj_cu.shape

        # Values equal (within tolerance)
        np.testing.assert_allclose(
            trj_cpu, _to_numpy(trj_cu),
            atol=self.ATOL, rtol=self.RTOL)

    def test_trajectory_array_cpu_vs_torch(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        trj_cpu = self._solve("cpu", period_output=1, get_trj=True)
        trj_torch = self._solve("torch:gpu:0", period_output=1, get_trj=True)

        assert isinstance(trj_cpu, np.ndarray)
        assert isinstance(trj_torch, torch.Tensor)
        assert trj_torch.device.type == "cuda"

        assert tuple(trj_cpu.shape) == tuple(trj_torch.shape)
        np.testing.assert_allclose(
            trj_cpu, _to_numpy(trj_torch),
            atol=self.ATOL, rtol=self.RTOL)

    # ---- trj_waypoints ----

    def test_waypoint_dict_cpu_vs_cupy(self):
        waypoints = [1]  # must be within [1, N_ITERS]
        result_cpu = self._solve("cpu", trj_waypoints=waypoints)
        result_cu = self._solve("cuda:0", trj_waypoints=waypoints)

        # Same dict structure
        assert isinstance(result_cpu, dict)
        assert isinstance(result_cu, dict)
        assert set(result_cpu.keys()) == set(result_cu.keys()) == {"iters", "trj"}

        # iters is a list and identical
        assert isinstance(result_cpu["iters"], list)
        assert isinstance(result_cu["iters"], list)
        assert result_cpu["iters"] == result_cu["iters"] == waypoints

        # trj types follow device, values match
        assert isinstance(result_cpu["trj"], np.ndarray)
        assert isinstance(result_cu["trj"], cupy.ndarray)
        assert result_cpu["trj"].shape == result_cu["trj"].shape
        np.testing.assert_allclose(
            result_cpu["trj"], _to_numpy(result_cu["trj"]),
            atol=self.ATOL, rtol=self.RTOL)

    def test_waypoint_dict_cpu_vs_torch(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        waypoints = [1]
        result_cpu = self._solve("cpu", trj_waypoints=waypoints)
        result_torch = self._solve("torch:gpu:0", trj_waypoints=waypoints)

        assert isinstance(result_cpu, dict)
        assert isinstance(result_torch, dict)
        assert set(result_cpu.keys()) == set(result_torch.keys()) == {"iters", "trj"}

        assert result_cpu["iters"] == result_torch["iters"] == waypoints

        assert isinstance(result_cpu["trj"], np.ndarray)
        assert isinstance(result_torch["trj"], torch.Tensor)
        assert result_torch["trj"].device.type == "cuda"
        assert tuple(result_cpu["trj"].shape) == tuple(result_torch["trj"].shape)
        np.testing.assert_allclose(
            result_cpu["trj"], _to_numpy(result_torch["trj"]),
            atol=self.ATOL, rtol=self.RTOL)

    # ---- empty waypoints / out-of-range ----

    def test_empty_waypoints_cpu_vs_cupy(self):
        # All waypoints out of range → empty trajectory + empty iters on both
        waypoints = [999]
        result_cpu = self._solve("cpu", trj_waypoints=waypoints)
        result_cu = self._solve("cuda:0", trj_waypoints=waypoints)

        assert isinstance(result_cpu, dict)
        assert isinstance(result_cu, dict)
        assert result_cpu["iters"] == []
        assert result_cu["iters"] == []
        assert result_cpu["trj"].shape[0] == 0
        assert result_cu["trj"].shape[0] == 0


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

    def test_torch_cuda_save_states(self, tmp_path):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        model = _make_model("torch:gpu:0")
        from lpf.solvers import EulerSolver

        dpath_states = str(tmp_path / "states")
        EulerSolver(dt=0.01, n_iters=20).solve(
            model,
            period_output=10,
            dpath_states=dpath_states,
            verbose=0,
        )

        state_files = []
        for _, _, files in os.walk(str(tmp_path)):
            for fname in files:
                if fname.endswith(".npz"):
                    state_files.append(fname)

        assert state_files, "Should save state files for torch CUDA models"
        assert isinstance(model.y_mesh, torch.Tensor)
        assert model.y_mesh.device.type == "cuda"


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

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_solver_reuse_across_shape_change_matches_fresh_solver(self, solver_name):
        cls = _get_solver_class(solver_name)

        solver_reuse = cls(dt=0.01, n_iters=1)
        solver_reuse.solve(_make_model("cuda:0", width=16, height=16), verbose=0)

        model_reuse = _make_model("cuda:0", width=24, height=24)
        solver_reuse.solve(model_reuse, verbose=0)
        y_reuse = _get_numpy(model_reuse)

        model_fresh = _make_model("cuda:0", width=24, height=24)
        cls(dt=0.01, n_iters=1).solve(model_fresh, verbose=0)
        y_fresh = _get_numpy(model_fresh)

        np.testing.assert_allclose(y_reuse, y_fresh, atol=1e-6, rtol=1e-6)


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
