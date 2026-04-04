"""Tests for 5 new reaction-diffusion models:
Brusselator, FitzHugh-Nagumo, Lengyel-Epstein, Thomas, Barkley.

Covers:
  - Python (CPU) solve: finite results
  - CUDA (cuda:0) solve: finite results + matches CPU
  - Torch CUDA solve: finite results
  - All solvers: Euler, Heun, RK4 on each model
  - Model I/O: to_dict, parse_params round-trip
  - ModelFactory registration
"""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")


# ---- Model specs: (class_name, n_params, sample_params) ----
MODEL_SPECS = [
    ("BrusselatorModel", 4,
     [1e-3, 1e-2, 1.0, 3.0]),
    ("FitzHughNagumoModel", 5,
     [1e-3, 1e-1, 0.01, 0.5, 0.1]),
    ("LengyelEpsteinModel", 4,
     [1e-3, 1e-2, 5.0, 1.0]),
    ("ThomasModel", 7,
     [1e-3, 1e-2, 0.5, 0.5, 1.0, 0.5, 0.01]),
    ("BarkleyModel", 5,
     [1e-3, 1e-2, 0.02, 0.7, 0.05]),
]


def _get_model_class(name):
    import lpf.models as M
    return getattr(M, name)


def _make_model(model_name, n_params, sample_params, device="cpu",
                batch_size=2, dtype=np.float32):
    from lpf.initializers import LiawInitializer

    np.random.seed(42)
    init_pts = np.random.randint(0, 32, size=(batch_size, 3, 2)).astype(np.uint32)
    init_states = np.array([[0.5, 0.5]] * batch_size, dtype=dtype)
    params = np.array([sample_params] * batch_size, dtype=dtype)

    cls = _get_model_class(model_name)
    return cls(
        initializer=LiawInitializer(init_states=init_states, init_pts=init_pts),
        n_init_pts=3, params=params,
        thr_color=0.5 * np.ones((batch_size, 1, 1)),
        width=32, height=32, dx=0.1,
        device=device, dtype=dtype,
    )


def _get_numpy(model):
    arr = model.y_mesh
    if hasattr(arr, 'get'):
        return arr.get()
    if hasattr(arr, 'cpu'):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


# ---- CPU Python solve ----

class TestCPUSolve:
    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_cpu_euler_finite(self, model_name, n_params, sample_params):
        from lpf.solvers import EulerSolver
        model = _make_model(model_name, n_params, sample_params, "cpu")
        EulerSolver(dt=0.01, n_iters=50).solve(model, verbose=0)
        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), f"{model_name} CPU produced non-finite"

    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_cpu_rk4_finite(self, model_name, n_params, sample_params):
        from lpf.solvers import RungeKuttaSolver
        model = _make_model(model_name, n_params, sample_params, "cpu")
        RungeKuttaSolver(dt=0.01, n_iters=50).solve(model, verbose=0)
        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), f"{model_name} CPU RK4 produced non-finite"


# ---- CUDA solve ----

class TestCUDASolve:
    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_cuda_euler_finite(self, model_name, n_params, sample_params):
        from lpf.solvers import EulerSolver
        model = _make_model(model_name, n_params, sample_params, "cuda:0")
        EulerSolver(dt=0.01, n_iters=50).solve(model, verbose=0)
        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), f"{model_name} CUDA Euler non-finite"

    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_cuda_heun_finite(self, model_name, n_params, sample_params):
        from lpf.solvers import HeunSolver
        model = _make_model(model_name, n_params, sample_params, "cuda:0")
        HeunSolver(dt=0.01, n_iters=50).solve(model, verbose=0)
        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), f"{model_name} CUDA Heun non-finite"

    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_cuda_rk4_finite(self, model_name, n_params, sample_params):
        from lpf.solvers import RungeKuttaSolver
        model = _make_model(model_name, n_params, sample_params, "cuda:0")
        RungeKuttaSolver(dt=0.01, n_iters=50).solve(model, verbose=0)
        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), f"{model_name} CUDA RK4 non-finite"


# ---- CUDA vs CPU correctness ----

class TestCUDAvsCSPU:
    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_cuda_matches_cpu(self, model_name, n_params, sample_params):
        from lpf.solvers import EulerSolver
        n_iters, dt = 50, 0.01

        m_cpu = _make_model(model_name, n_params, sample_params, "cpu")
        EulerSolver(dt=dt, n_iters=n_iters).solve(m_cpu, verbose=0)
        y_cpu = _get_numpy(m_cpu)

        m_cuda = _make_model(model_name, n_params, sample_params, "cuda:0")
        EulerSolver(dt=dt, n_iters=n_iters).solve(m_cuda, verbose=0)
        y_cuda = _get_numpy(m_cuda)

        valid = np.isfinite(y_cpu) & np.isfinite(y_cuda)
        if valid.any():
            assert np.max(np.abs(y_cpu[valid] - y_cuda[valid])) < 1e-2, \
                f"{model_name}: CUDA vs CPU diff too large"


# ---- Torch CUDA ----

class TestTorchCUDA:
    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_torch_cuda_finite(self, model_name, n_params, sample_params):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        from lpf.solvers import EulerSolver
        model = _make_model(model_name, n_params, sample_params, "torch:gpu:0")
        EulerSolver(dt=0.01, n_iters=50).solve(model, verbose=0)
        y = _get_numpy(model)
        assert np.all(np.isfinite(y)), f"{model_name} torch:gpu non-finite"
        assert isinstance(model.y_mesh, torch.Tensor)


# ---- Model I/O: to_dict / parse_params ----

class TestModelIO:
    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_to_dict_has_params(self, model_name, n_params, sample_params):
        model = _make_model(model_name, n_params, sample_params, "cpu")
        d = model.to_dict(index=0)
        assert "Du" in d
        assert "Dv" in d
        assert d["model"] == model_name

    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_parse_params_roundtrip(self, model_name, n_params, sample_params):
        model = _make_model(model_name, n_params, sample_params, "cpu")
        d = model.to_dict(index=0)
        cls = _get_model_class(model_name)
        parsed = cls.parse_params([d])
        assert parsed.shape == (1, n_params)
        np.testing.assert_allclose(parsed[0, 0], sample_params[0], rtol=1e-5)


# ---- ModelFactory ----

class TestModelFactory:
    @pytest.mark.parametrize("name", [
        "brusselator", "fitzhughnagumo", "lengyelepstein", "thomas", "barkley",
    ])
    def test_factory_creates(self, name):
        from lpf.models import ModelFactory
        from lpf.initializers import LiawInitializer

        np.random.seed(42)
        init_pts = np.array([[[8, 8]]], dtype=np.uint32)
        init_states = np.array([[0.5, 0.5]], dtype=np.float32)

        # Get n_params from the spec
        spec = {s[0].lower().replace("model", ""): s for s in MODEL_SPECS}
        _, n_p, sp = spec[name]
        params = np.array([sp], dtype=np.float32)

        model = ModelFactory.create(
            name,
            initializer=LiawInitializer(init_states=init_states, init_pts=init_pts),
            n_init_pts=1, params=params,
            thr_color=0.5 * np.ones((1, 1, 1)),
            width=16, height=16, dx=0.1, device="cpu",
        )
        assert model is not None
        model.initialize()


# ---- Float64 ----

class TestFloat64NewModels:
    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_float64_cuda(self, model_name, n_params, sample_params):
        from lpf.solvers import EulerSolver
        model = _make_model(model_name, n_params, sample_params,
                            "cuda:0", dtype=np.float64)
        EulerSolver(dt=0.01, n_iters=20).solve(model, verbose=0)
        y = _get_numpy(model)
        assert y.dtype == np.float64
        assert np.all(np.isfinite(y))
