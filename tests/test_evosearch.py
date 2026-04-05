"""EvoSearch pipeline tests: fitness evaluation, caching, save, and GPU support.

Regression tests for:
- evosearch.py: compute() expected list of images, but received single PIL Image
- rdmodel.py: params setter failed when _params was None (ModelFactory path)
- rdmodel.py: params setter failed with NumPy array on CuPy model (cross-device)
"""

import os
import shutil

import numpy as np
import pytest

from lpf.data import load_model_dicts, load_targets
from lpf.solvers import SolverFactory
from lpf.search import EvoSearch
from lpf.objectives import ObjectiveFactory
from lpf.models import ModelFactory
from lpf.converters import ConverterFactory


LPF_ROOT = os.path.join(os.path.dirname(__file__), "..")
DPATH_POP = os.path.join(LPF_ROOT, "population", "init_pop_axyridis")


def _has_cupy_gpu():
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _has_lpips():
    try:
        import lpips
        return True
    except Exception:
        return False


@pytest.fixture
def cpu_search(tmp_path):
    """Create an EvoSearch instance with CPU model and lightweight objectives."""
    model = ModelFactory.create(
        name="Liaw", n_init_pts=25,
        width=128, height=128, dx=0.1, device="cpu",
    )
    solver = SolverFactory.create(name="Euler", dt=0.01, n_iters=100)
    converter = ConverterFactory.create("LiawInitializer")
    obj_config = [
        ["MeanMeanSquareError", "1e-1", "cpu"],
        ["MeanColorProportion", "1e0", "cpu"],
    ]
    objectives = ObjectiveFactory.create(obj_config)
    targets = load_targets("haxyridis", ["axyridis"])

    search = EvoSearch(
        model=model, solver=solver, converter=converter,
        targets=targets, objectives=objectives,
        droot_output=str(tmp_path / "output"),
    )
    model_dicts = load_model_dicts(DPATH_POP)
    dv = converter.to_dv(model_dicts[0], 25)
    return search, converter, model_dicts, dv


class TestEvoSearchFitness:
    """fitness() must return a list with a single finite scalar."""

    def test_fitness_returns_scalar_list(self, cpu_search):
        search, _, _, dv = cpu_search
        result = search.fitness(dv)
        assert isinstance(result, list)
        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_fitness_cache_returns_same_result(self, cpu_search):
        search, _, _, dv = cpu_search
        f1 = search.fitness(dv)
        f2 = search.fitness(dv)
        assert f1 == f2

    def test_fitness_different_dv_different_result(self, cpu_search):
        search, converter, model_dicts, dv1 = cpu_search
        dv2 = converter.to_dv(model_dicts[1], 25)
        f1 = search.fitness(dv1)
        f2 = search.fitness(dv2)
        assert f1 != f2


class TestEvoSearchSave:
    """save() must write model JSON, morph PNG, and pattern PNG."""

    def test_save_best(self, cpu_search):
        search, _, _, dv = cpu_search
        f = search.fitness(dv)
        ok = search.save("best", dv, generation=1, fitness=f[0])
        assert ok is True
        files = os.listdir(search.dpath_best)
        extensions = {os.path.splitext(f)[1] for f in files}
        assert ".json" in extensions
        assert ".png" in extensions
        assert len(files) == 3  # model, morph, pattern

    def test_save_pop(self, cpu_search):
        search, _, _, dv = cpu_search
        f = search.fitness(dv)
        ok = search.save("pop", dv, generation=1, fitness=f[0])
        assert ok is True
        files = os.listdir(search.dpath_population)
        assert len(files) == 3

    def test_save_invalid_mode_raises(self, cpu_search):
        search, _, _, dv = cpu_search
        with pytest.raises(ValueError, match="mode"):
            search.save("invalid", dv)


@pytest.mark.skipif(not _has_cupy_gpu(), reason="CuPy GPU not available")
class TestEvoSearchGPU:
    """EvoSearch must work with GPU (CuPy) models."""

    @pytest.fixture
    def gpu_search(self, tmp_path):
        model = ModelFactory.create(
            name="Liaw", n_init_pts=25,
            width=128, height=128, dx=0.1, device="cuda:0",
        )
        solver = SolverFactory.create(name="Euler", dt=0.01, n_iters=100)
        converter = ConverterFactory.create("LiawInitializer")
        obj_config = [
            ["MeanMeanSquareError", "1e-1", "cpu"],
            ["MeanColorProportion", "1e0", "cpu"],
        ]
        objectives = ObjectiveFactory.create(obj_config)
        targets = load_targets("haxyridis", ["axyridis"])

        search = EvoSearch(
            model=model, solver=solver, converter=converter,
            targets=targets, objectives=objectives,
            droot_output=str(tmp_path / "output"),
        )
        model_dicts = load_model_dicts(DPATH_POP)
        dv = converter.to_dv(model_dicts[0], 25)
        return search, dv

    def test_gpu_fitness(self, gpu_search):
        search, dv = gpu_search
        result = search.fitness(dv)
        assert isinstance(result, list)
        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_gpu_save(self, gpu_search):
        search, dv = gpu_search
        f = search.fitness(dv)
        ok = search.save("best", dv, generation=1, fitness=f[0])
        assert ok is True
        assert len(os.listdir(search.dpath_best)) == 3


@pytest.mark.skipif(
    not (_has_cupy_gpu() and _has_torch_cuda() and _has_lpips()),
    reason="Requires CuPy GPU + PyTorch CUDA + lpips",
)
class TestEvoSearchGPUObjectives:
    """Full pipeline with GPU objectives (VGG16, LPIPS)."""

    def test_all_objectives_on_gpu(self, tmp_path):
        model = ModelFactory.create(
            name="Liaw", n_init_pts=25,
            width=128, height=128, dx=0.1, device="cuda:0",
        )
        solver = SolverFactory.create(name="Euler", dt=0.01, n_iters=100)
        converter = ConverterFactory.create("LiawInitializer")
        obj_config = [
            ["MeanMeanSquareError", "1e-1", "cpu"],
            ["MeanColorProportion", "1e0", "cpu"],
            ["MeanVgg16PerceptualLoss", "1e-4", "cuda:0"],
            ["MeanLearnedPerceptualImagePatchSimilarity:vgg", "1.5e1", "cuda:0"],
            ["MeanLearnedPerceptualImagePatchSimilarity:alex", "4e0", "cuda:0"],
        ]
        objectives = ObjectiveFactory.create(obj_config)
        targets = load_targets("haxyridis", ["axyridis"])

        search = EvoSearch(
            model=model, solver=solver, converter=converter,
            targets=targets, objectives=objectives,
            droot_output=str(tmp_path / "output"),
        )
        model_dicts = load_model_dicts(DPATH_POP)
        dv = converter.to_dv(model_dicts[0], 25)

        result = search.fitness(dv)
        assert isinstance(result, list)
        assert len(result) == 1
        assert np.isfinite(result[0])
        assert result[0] > 0
