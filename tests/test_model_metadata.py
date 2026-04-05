"""Model classes must have correct docstrings and name attributes.

Regression tests for Section 4 issues:
- 4-1: SchnakenbergModel docstring said "Gierer-Meinhardt model"
- 4-2/4-3: End comments said "GrayScottModel"
"""

import numpy as np
import pytest


def _make_model(ModelClass, params_list):
    from lpf.initializers import LiawInitializer
    init_pts = np.array([[[5, 5]]], dtype=np.uint32)
    init_states = np.array([[0.5, 0.5]], dtype=np.float32)
    initializer = LiawInitializer(init_states=init_states, init_pts=init_pts)
    params = np.array([params_list], dtype=np.float32)
    return ModelClass(
        initializer=initializer, params=params,
        thr_color=0.5 * np.ones((1, 1, 1)),
        width=16, height=16, dx=0.1, device="cpu",
    )


class TestSchnakenbergModelMetadata:

    def test_docstring_says_schnakenberg(self):
        from lpf.models.schnakenbergmodel import SchnakenbergModel
        doc = SchnakenbergModel.__doc__
        assert "Schnakenberg" in doc
        assert "Gierer-Meinhardt" not in doc

    def test_model_name(self):
        from lpf.models.schnakenbergmodel import SchnakenbergModel
        model = _make_model(SchnakenbergModel, [1e-3, 1e-2, 1.0, 0.5, 0.5, 0.1])
        assert model._name == "SchnakenbergModel"


class TestGiererMeinhardtModelMetadata:

    def test_docstring_says_gierer_meinhardt(self):
        from lpf.models.gierermeinhardtmodel import GiererMeinhardtModel
        assert "Gierer-Meinhardt" in GiererMeinhardtModel.__doc__

    def test_model_name(self):
        from lpf.models.gierermeinhardtmodel import GiererMeinhardtModel
        model = _make_model(GiererMeinhardtModel, [1e-3, 1e-2, 1.0, 1.0, 0.5, 0.3])
        assert model._name == "GiererMeinhardtModel"
