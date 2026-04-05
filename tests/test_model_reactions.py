"""Model reaction functions must produce correct kinetics.

Tests the core mathematical correctness of each reaction-diffusion model's
reaction terms, which are critical for simulation accuracy.
"""

import numpy as np
import pytest


def _make_model(ModelClass, params, n_params):
    """Helper to create a model with given params."""
    from lpf.initializers import LiawInitializer

    init_pts = np.array([[[8, 8]]], dtype=np.uint32)
    init_states = np.array([[0.5, 0.5]], dtype=np.float32)
    initializer = LiawInitializer(init_states=init_states, init_pts=init_pts)
    params_arr = np.array([params], dtype=np.float32)

    model = ModelClass(
        initializer=initializer, params=params_arr,
        thr_color=0.5 * np.ones((1, 1, 1)),
        width=16, height=16, dx=0.1, device="cpu",
    )
    model.initialize()
    return model


class TestLiawReactions:
    """Liaw model: f = ru*(u²v/(1+ku²)) + su - mu*u, g = -rv*(u²v/(1+ku²)) + sv"""

    def test_reactions_shape(self):
        from lpf.models import LiawModel
        model = _make_model(LiawModel, [1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01], 8)
        u = model.u
        v = model.v
        f, g = model.reactions(t=0.0, u_c=u, v_c=v)
        assert f.shape == u.shape
        assert g.shape == v.shape

    def test_reactions_finite(self):
        from lpf.models import LiawModel
        model = _make_model(LiawModel, [1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01], 8)
        f, g = model.reactions(t=0.0, u_c=model.u, v_c=model.v)
        assert np.all(np.isfinite(f))
        assert np.all(np.isfinite(g))


class TestGrayScottReactions:
    """Gray-Scott: f = -uv² + F(1-u), g = uv² - (F+k)v"""

    def test_reactions_shape(self):
        from lpf.models.grayscottmodel import GrayScottModel
        model = _make_model(GrayScottModel, [2e-5, 1e-5, 0.04, 0.06], 4)
        f, g = model.reactions(t=0.0, u_c=model.u, v_c=model.v)
        assert f.shape == model.u.shape
        assert g.shape == model.v.shape

    def test_uniform_steady_state(self):
        """At u=1, v=0: f = F(1-1) = 0, g = -(F+k)*0 = 0 → steady state."""
        from lpf.models.grayscottmodel import GrayScottModel
        F, k = 0.04, 0.06
        model = _make_model(GrayScottModel, [2e-5, 1e-5, F, k], 4)
        model._y_mesh[0, :, :, :] = 1.0  # u = 1
        model._y_mesh[1, :, :, :] = 0.0  # v = 0
        f, g = model.reactions(t=0.0, u_c=model.u, v_c=model.v)
        np.testing.assert_allclose(f, 0.0, atol=1e-7)
        np.testing.assert_allclose(g, 0.0, atol=1e-7)


class TestSchnakenbergReactions:
    """Schnakenberg: f = su - mu*u + rho*u²v, g = sv - rho*u²v"""

    def test_reactions_shape(self):
        from lpf.models.schnakenbergmodel import SchnakenbergModel
        model = _make_model(SchnakenbergModel, [1e-3, 1e-2, 1.0, 0.5, 0.5, 0.1], 6)
        f, g = model.reactions(t=0.0, u_c=model.u, v_c=model.v)
        assert f.shape == model.u.shape

    def test_reactions_known_values(self):
        """Verify reaction terms for known u, v values."""
        from lpf.models.schnakenbergmodel import SchnakenbergModel
        rho, su, sv, mu = 2.0, 0.1, 0.5, 0.3
        model = _make_model(SchnakenbergModel, [1e-3, 1e-2, rho, su, sv, mu], 6)

        u_val, v_val = 1.0, 2.0
        model._y_mesh[0, :, :, :] = u_val
        model._y_mesh[1, :, :, :] = v_val

        f, g = model.reactions(t=0.0, u_c=model.u, v_c=model.v)
        expected_f = su - mu * u_val + rho * u_val**2 * v_val
        expected_g = sv - rho * u_val**2 * v_val
        np.testing.assert_allclose(f[0, 0, 0], expected_f, rtol=1e-5)
        np.testing.assert_allclose(g[0, 0, 0], expected_g, rtol=1e-5)


class TestGiererMeinhardtReactions:
    """Gierer-Meinhardt: f = ru*u²/v - mu*u, g = rv*u² - nu*v"""

    def test_reactions_shape(self):
        from lpf.models.gierermeinhardtmodel import GiererMeinhardtModel
        model = _make_model(GiererMeinhardtModel, [1e-3, 1e-2, 1.0, 1.0, 0.5, 0.3], 6)
        f, g = model.reactions(t=0.0, u_c=model.u, v_c=model.v)
        assert f.shape == model.u.shape

    def test_reactions_known_values(self):
        from lpf.models.gierermeinhardtmodel import GiererMeinhardtModel
        ru, rv, mu, nu = 2.0, 1.5, 0.3, 0.2
        model = _make_model(GiererMeinhardtModel, [1e-3, 1e-2, ru, rv, mu, nu], 6)

        u_val, v_val = 1.0, 2.0
        model._y_mesh[0, :, :, :] = u_val
        model._y_mesh[1, :, :, :] = v_val

        f, g = model.reactions(t=0.0, u_c=model.u, v_c=model.v)
        expected_f = ru * u_val**2 / v_val - mu * u_val
        expected_g = rv * u_val**2 - nu * v_val
        np.testing.assert_allclose(f[0, 0, 0], expected_f, rtol=1e-5)
        np.testing.assert_allclose(g[0, 0, 0], expected_g, rtol=1e-5)


class TestModelToDict:
    """to_dict() must produce correct serialization for all models."""

    def test_liaw_to_dict_roundtrip(self):
        from lpf.models import LiawModel
        params = [1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]
        model = _make_model(LiawModel, params, 8)

        d = model.to_dict(index=0, generation=5, fitness=0.95)
        assert d["Du"] == pytest.approx(1e-3)
        assert d["Dv"] == pytest.approx(1e-2)
        assert d["generation"] == 5
        assert d["fitness"] == pytest.approx(0.95)

    def test_schnakenberg_to_dict_has_all_params(self):
        from lpf.models.schnakenbergmodel import SchnakenbergModel
        params = [1e-3, 1e-2, 1.0, 0.5, 0.5, 0.1]
        model = _make_model(SchnakenbergModel, params, 6)

        d = model.to_dict(index=0)
        for key in ["Du", "Dv", "rho", "su", "sv", "mu"]:
            assert key in d, f"Missing key: {key}"

    def test_gierer_meinhardt_to_dict_has_all_params(self):
        from lpf.models.gierermeinhardtmodel import GiererMeinhardtModel
        params = [1e-3, 1e-2, 1.0, 1.0, 0.5, 0.3]
        model = _make_model(GiererMeinhardtModel, params, 6)

        d = model.to_dict(index=0)
        for key in ["Du", "Dv", "ru", "rv", "mu", "nu"]:
            assert key in d, f"Missing key: {key}"
