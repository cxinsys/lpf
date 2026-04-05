"""Tests for 5 new reaction-diffusion models:
Brusselator, FitzHugh-Nagumo, Lengyel-Epstein, Thomas, Barkley.

All tests in this file run on CPU (no GPU required).
GPU-specific tests are in test_cuda_solvers.py.
"""

import numpy as np
import pytest


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


def _make_model(model_name, n_params, sample_params,
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
        device="cpu", dtype=dtype,
    )


# ---- CPU Python solve ----

class TestCPUSolve:
    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_cpu_euler_finite(self, model_name, n_params, sample_params):
        from lpf.solvers import EulerSolver
        model = _make_model(model_name, n_params, sample_params)
        EulerSolver(dt=0.01, n_iters=50).solve(model, verbose=0)
        y = np.asarray(model.y_mesh)
        assert np.all(np.isfinite(y)), f"{model_name} CPU Euler non-finite"

    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_cpu_rk4_finite(self, model_name, n_params, sample_params):
        from lpf.solvers import RungeKuttaSolver
        model = _make_model(model_name, n_params, sample_params)
        RungeKuttaSolver(dt=0.01, n_iters=50).solve(model, verbose=0)
        y = np.asarray(model.y_mesh)
        assert np.all(np.isfinite(y)), f"{model_name} CPU RK4 non-finite"

    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_cpu_heun_finite(self, model_name, n_params, sample_params):
        from lpf.solvers import HeunSolver
        model = _make_model(model_name, n_params, sample_params)
        HeunSolver(dt=0.01, n_iters=50).solve(model, verbose=0)
        y = np.asarray(model.y_mesh)
        assert np.all(np.isfinite(y)), f"{model_name} CPU Heun non-finite"


# ---- Model I/O: to_dict / parse_params ----

class TestModelIO:
    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_to_dict_has_params(self, model_name, n_params, sample_params):
        model = _make_model(model_name, n_params, sample_params)
        d = model.to_dict(index=0)
        assert "Du" in d
        assert "Dv" in d
        assert d["model"] == model_name

    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_parse_params_roundtrip(self, model_name, n_params, sample_params):
        model = _make_model(model_name, n_params, sample_params)
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


# ---- Float64 (CPU) ----

class TestFloat64CPU:
    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_float64_cpu(self, model_name, n_params, sample_params):
        from lpf.solvers import EulerSolver
        model = _make_model(model_name, n_params, sample_params, dtype=np.float64)
        EulerSolver(dt=0.01, n_iters=20).solve(model, verbose=0)
        y = np.asarray(model.y_mesh)
        assert y.dtype == np.float64
        assert np.all(np.isfinite(y))


# ---- Trajectory (CPU) ----

class TestTrajectoryCPU:
    def test_trajectory_with_waypoints(self):
        from lpf.solvers import EulerSolver
        model = _make_model("BrusselatorModel", 4, [1e-3, 1e-2, 1.0, 3.0])
        result = EulerSolver(dt=0.01, n_iters=50).solve(
            model, trj_waypoints=[10, 30, 50], verbose=0)
        assert isinstance(result, dict)
        assert result["trj"].shape[0] == 3

    def test_trajectory_with_period(self):
        from lpf.solvers import EulerSolver
        model = _make_model("FitzHughNagumoModel", 5,
                            [1e-3, 1e-1, 0.01, 0.5, 0.1])
        trj = EulerSolver(dt=0.01, n_iters=100).solve(
            model, period_output=50, get_trj=True, verbose=0)
        assert trj.shape[0] == 3  # iter 1, 50, 100


# ---- Reactions produce non-zero derivatives ----

class TestReactions:
    @pytest.mark.parametrize("model_name,n_params,sample_params", MODEL_SPECS)
    def test_reactions_nonzero(self, model_name, n_params, sample_params):
        """Reaction terms should be non-trivial for typical inputs."""
        model = _make_model(model_name, n_params, sample_params)
        model.initialize()
        u_c = model.u[:, 1:-1, 1:-1]
        v_c = model.v[:, 1:-1, 1:-1]
        f, g = model.reactions(0.0, u_c, v_c)
        # At least one of f, g should be non-zero
        assert not (np.allclose(f, 0) and np.allclose(g, 0)), \
            f"{model_name} reactions returned all zeros"
