"""End-to-end tests derived from tutorial notebooks.

These tests run the core simulation workflows from tutorials 01 and 02
with reduced iterations to verify the full pipeline works correctly:
  import → params → initializer → model → solver → output → visualization
"""

import os
import os.path as osp
from os.path import join as pjoin
import json
import tempfile

import numpy as np
import pytest
from PIL import Image

from lpf.initializers import LiawInitializer
from lpf.models import LiawModel
from lpf.solvers import EulerSolver, RungeKuttaSolver


class TestTutorial01SingleParamset:
    """Tutorial 01: Solve a single parameter set end-to-end."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Replicate tutorial01 setup with reduced iterations."""
        batch_size = 1
        device = "cpu"
        dt = 0.01
        n_iters = 100  # reduced from 500000
        dx = 0.1
        width = 64   # reduced from 128
        height = 64

        param_dict = {
            "u0": 2.0, "v0": 1.0,
            "Du": 0.0005, "Dv": 0.075,
            "ru": 0.18, "rv": 0.02874,
            "su": 0.001, "sv": 0.025,
            "k": 0.084,
            "mu": 0.08,
        }

        np.random.seed(42)
        for i in range(25):
            param_dict["init_pts_%d" % (i + 1)] = (
                np.random.randint(0, height),
                np.random.randint(0, width),
            )

        return {
            "param_dict": param_dict,
            "dt": dt, "n_iters": n_iters, "dx": dx,
            "width": width, "height": height,
            "device": device, "dpath_output": str(tmp_path),
        }

    def test_parse_params(self, setup):
        """LiawModel.parse_params should accept model dict."""
        model_dicts = [setup["param_dict"]]
        params = LiawModel.parse_params(model_dicts)
        assert params.shape == (1, 8)
        assert params.dtype == np.float32

    def test_initializer_update(self, setup):
        """LiawInitializer.update should parse init_pts from dict."""
        model_dicts = [setup["param_dict"]]
        initializer = LiawInitializer()
        initializer.update(model_dicts)
        assert initializer.init_pts is not None
        assert initializer.init_states is not None

    def test_model_creation(self, setup):
        """LiawModel should initialize without errors."""
        model_dicts = [setup["param_dict"]]
        initializer = LiawInitializer()
        initializer.update(model_dicts)
        params = LiawModel.parse_params(model_dicts)

        model = LiawModel(
            initializer=initializer, params=params,
            dx=setup["dx"], width=setup["width"], height=setup["height"],
            device=setup["device"],
        )
        model.initialize()

        assert model.u.shape == (1, setup["height"], setup["width"])
        assert model.v.shape == (1, setup["height"], setup["width"])

    def test_euler_solve(self, setup):
        """EulerSolver.solve should run without errors and produce output."""
        model_dicts = [setup["param_dict"]]
        initializer = LiawInitializer()
        initializer.update(model_dicts)
        params = LiawModel.parse_params(model_dicts)

        model = LiawModel(
            initializer=initializer, params=params,
            dx=setup["dx"], width=setup["width"], height=setup["height"],
            device=setup["device"],
        )
        model.initialize()

        dpath = setup["dpath_output"]
        solver = EulerSolver()
        solver.solve(
            model=model, dt=setup["dt"], n_iters=setup["n_iters"],
            period_output=50,
            dpath_model=dpath, dpath_morph=dpath, dpath_pattern=dpath,
            verbose=0,
        )

        # Check finite results
        u = model.am.get(model.u)
        assert np.all(np.isfinite(u)), "Solution should be finite"

        # Check output directories were created
        assert osp.isdir(pjoin(dpath, "models"))
        assert osp.isdir(pjoin(dpath, "model_1"))

        # Check model JSON was saved
        fpath_model = pjoin(dpath, "models", "model_1.json")
        assert osp.isfile(fpath_model)

        with open(fpath_model, "rt") as fin:
            model_dict = json.load(fin)
        assert "Du" in model_dict
        assert "solver" in model_dict

    def test_rk4_solve(self, setup):
        """RungeKuttaSolver should also work end-to-end."""
        model_dicts = [setup["param_dict"]]
        initializer = LiawInitializer()
        initializer.update(model_dicts)
        params = LiawModel.parse_params(model_dicts)

        model = LiawModel(
            initializer=initializer, params=params,
            dx=setup["dx"], width=setup["width"], height=setup["height"],
            device=setup["device"],
        )
        model.initialize()

        solver = RungeKuttaSolver()
        solver.solve(
            model=model, dt=setup["dt"], n_iters=setup["n_iters"],
            verbose=0,
        )

        u = model.am.get(model.u)
        assert np.all(np.isfinite(u))

    def test_colorize_and_save_image(self, setup):
        """Model colorize + save_image pipeline."""
        model_dicts = [setup["param_dict"]]
        initializer = LiawInitializer()
        initializer.update(model_dicts)
        params = LiawModel.parse_params(model_dicts)

        model = LiawModel(
            initializer=initializer, params=params,
            dx=setup["dx"], width=setup["width"], height=setup["height"],
            device=setup["device"],
        )
        model.initialize()

        EulerSolver().solve(
            model=model, dt=setup["dt"], n_iters=50,
            verbose=0,
        )

        arr_color = model.colorize(thr_color=0.5)
        assert arr_color.shape == (1, setup["height"], setup["width"], 3)

        img_morph, img_pattern = model.create_image(0, arr_color)
        assert isinstance(img_morph, Image.Image)
        assert isinstance(img_pattern, Image.Image)

        dpath = setup["dpath_output"]
        fpath_morph = pjoin(dpath, "morph.png")
        fpath_pattern = pjoin(dpath, "pattern.png")
        model.save_image(index=0, fpath_morph=fpath_morph,
                         fpath_pattern=fpath_pattern)

        assert osp.isfile(fpath_morph)
        assert osp.isfile(fpath_pattern)

    def test_model_json_roundtrip(self, setup):
        """Saved model JSON should be loadable and contain all parameters."""
        model_dicts = [setup["param_dict"]]
        initializer = LiawInitializer()
        initializer.update(model_dicts)
        params = LiawModel.parse_params(model_dicts)

        model = LiawModel(
            initializer=initializer, params=params,
            dx=setup["dx"], width=setup["width"], height=setup["height"],
            device=setup["device"],
        )
        model.initialize()

        dpath = setup["dpath_output"]
        EulerSolver().solve(
            model=model, dt=setup["dt"], n_iters=50,
            period_output=50, dpath_model=dpath,
            verbose=0,
        )

        fpath_model = pjoin(dpath, "models", "model_1.json")
        with open(fpath_model, "rt") as fin:
            d = json.load(fin)

        # All kinetic parameters should be present
        for key in ["Du", "Dv", "ru", "rv", "k", "su", "sv", "mu"]:
            assert key in d, f"Missing key '{key}' in saved model"

        # Solver info should be present
        assert "solver" in d
        assert "dt" in d
        assert "n_iters" in d

    def test_merge_single_timeseries(self, setup):
        """merge_single_timeseries should produce a merged image."""
        from lpf.visualization import merge_single_timeseries

        model_dicts = [setup["param_dict"]]
        initializer = LiawInitializer()
        initializer.update(model_dicts)
        params = LiawModel.parse_params(model_dicts)

        model = LiawModel(
            initializer=initializer, params=params,
            dx=setup["dx"], width=setup["width"], height=setup["height"],
            device=setup["device"],
        )
        model.initialize()

        dpath = setup["dpath_output"]
        EulerSolver().solve(
            model=model, dt=setup["dt"], n_iters=setup["n_iters"],
            period_output=50,
            dpath_morph=dpath, dpath_pattern=dpath,
            verbose=0,
        )

        dpath_images = pjoin(dpath, "model_1")
        img_merged = merge_single_timeseries(
            dpath_input=dpath_images,
            n_cols=5,
            infile_header="morph",
            ratio_resize=0.5,
        )
        assert isinstance(img_merged, Image.Image)
        assert img_merged.width > 0
        assert img_merged.height > 0


class TestTutorial02VisualizeSingleMorph:
    """Tutorial 02: Solve and visualize a single morph."""

    def test_full_pipeline(self, tmp_path):
        """Full pipeline: params → model → solve → colorize → create_image."""
        param_dict = {
            "u0": 1.953, "v0": 2.394,
            "Du": 0.000498, "Dv": 0.0786,
            "ru": 0.19, "rv": 0.026,
            "su": 0.00217, "sv": 0.0279,
            "k": 0.093,
            "mu": 0.0692,
        }

        # Add init points
        for i in range(20):
            param_dict["init_pts_%d" % (i + 1)] = (
                np.random.randint(0, 64),
                np.random.randint(0, 64),
            )

        model_dicts = [param_dict]
        initializer = LiawInitializer()
        initializer.update(model_dicts)
        params = LiawModel.parse_params(model_dicts)

        model = LiawModel(
            initializer=initializer, params=params,
            dx=0.1, width=64, height=64, device="cpu",
        )
        model.initialize()

        solver = EulerSolver()
        solver.solve(model=model, dt=0.01, n_iters=50, verbose=0)

        # Colorize (mimics tutorial02 cell 10)
        arr_color = model.colorize(thr_color=0.5)
        assert arr_color.shape == (1, 64, 64, 3)

        img_morph, img_pattern = model.create_image(0, arr_color)
        assert isinstance(img_morph, Image.Image)
        assert isinstance(img_pattern, Image.Image)

        # Save (mimics tutorial02 cell 13)
        fpath_morph = str(tmp_path / "morph.png")
        fpath_pattern = str(tmp_path / "pattern.png")
        model.save_image(
            index=0,
            fpath_morph=fpath_morph,
            fpath_pattern=fpath_pattern,
        )

        assert osp.isfile(fpath_morph)
        assert osp.isfile(fpath_pattern)

        # Verify saved images are valid
        img = Image.open(fpath_morph)
        assert img.size[0] > 0


class TestTutorial03MultiParams:
    """Tutorial 03: Solve with multiple parameter sets (batch)."""

    def test_batch_solve(self, tmp_path):
        """Multiple parameter sets should solve in batch."""
        batch_size = 3
        param_dicts = []
        for b in range(batch_size):
            d = {
                "u0": 2.0, "v0": 1.0,
                "Du": 0.0005, "Dv": 0.075,
                "ru": 0.18 + b * 0.01, "rv": 0.02874,
                "su": 0.001, "sv": 0.025,
                "k": 0.084,
                "mu": 0.08,
            }
            np.random.seed(b)
            for i in range(5):
                d["init_pts_%d" % (i + 1)] = (
                    np.random.randint(0, 32),
                    np.random.randint(0, 32),
                )
            param_dicts.append(d)

        initializer = LiawInitializer()
        initializer.update(param_dicts)
        params = LiawModel.parse_params(param_dicts)
        assert params.shape == (batch_size, 8)

        model = LiawModel(
            initializer=initializer, params=params,
            dx=0.1, width=32, height=32, device="cpu",
        )
        model.initialize()
        assert model.batch_size == batch_size

        EulerSolver().solve(
            model=model, dt=0.01, n_iters=50, verbose=0,
        )

        u = model.am.get(model.u)
        assert u.shape == (batch_size, 32, 32)
        assert np.all(np.isfinite(u))

        # Colorize all
        arr_color = model.colorize(thr_color=0.5)
        assert arr_color.shape == (batch_size, 32, 32, 3)

        # Create image for each
        from lpf.visualization import merge_multiple
        imgs = []
        for i in range(batch_size):
            img_morph, img_pattern = model.create_image(i, arr_color)
            imgs.append(img_morph)

        img_merged = merge_multiple(
            imgs=imgs, n_cols=2, ratio_resize=0.5,
        )
        assert isinstance(img_merged, Image.Image)


class TestTutorial05EvoSearchSetup:
    """Tutorial 05/06: Verify the evosearch components can be created.

    We don't run the full evolutionary search (requires pygmo),
    but verify all components initialize correctly.
    """

    def test_model_factory(self):
        from lpf.models import ModelFactory
        model = ModelFactory.create(
            name="Liaw", n_init_pts=5,
            width=32, height=32, dx=0.1, device="cpu",
        )
        assert model is not None
        assert model._width == 32

    def test_solver_factory(self):
        from lpf.solvers.solverfactory import SolverFactory
        solver = SolverFactory.create(name="Euler", dt=0.01, n_iters=100)
        assert solver is not None
        assert solver._dt == 0.01

    def test_converter_factory(self):
        from lpf.converters import ConverterFactory
        converter = ConverterFactory.create("LiawInitializer")
        assert converter is not None

    def test_load_targets(self):
        """load_targets should load ladybird images."""
        try:
            from lpf.data import load_targets
            targets = load_targets("haxyridis", ["axyridis"])
            assert len(targets) > 0
            for img in targets:
                assert isinstance(img, Image.Image)
        except FileNotFoundError:
            pytest.skip("Target images not found")
