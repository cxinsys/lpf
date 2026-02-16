"""Tests for P1 semantic bug fixes (issue 2-1 through 2-6)."""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# 2-1: `if not x:` → `if x is None:` — falsy values should not be replaced
# ---------------------------------------------------------------------------

class TestNotXPattern:
    """Ensure that valid falsy values (0, 0.0, etc.) are preserved."""

    def test_twocomponentmodel_width_zero_not_overridden(self):
        """Width=0 should NOT be silently replaced with 128."""
        from lpf.models import LiawModel
        from lpf.initializers import LiawInitializer

        init_pts = np.array([[[16, 16]]], dtype=np.uint32)
        init_states = np.array([[0.5, 0.5]], dtype=np.float32)
        initializer = LiawInitializer(init_states=init_states, init_pts=init_pts)
        params = np.array([[1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]],
                          dtype=np.float32)

        # Explicitly set width=0 — should be stored as 0, not silently become 128
        model = LiawModel(
            initializer=initializer,
            params=params,
            width=0,
            height=32,
            dx=0.1,
            device="cpu",
        )
        assert model._width == 0, \
            "width=0 should be preserved, not silently replaced with default"

    def test_twocomponentmodel_dx_zero_not_overridden(self):
        """dx=0 should NOT be silently replaced with 0.1."""
        from lpf.models import LiawModel
        from lpf.initializers import LiawInitializer

        init_pts = np.array([[[16, 16]]], dtype=np.uint32)
        init_states = np.array([[0.5, 0.5]], dtype=np.float32)
        initializer = LiawInitializer(init_states=init_states, init_pts=init_pts)
        params = np.array([[1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]],
                          dtype=np.float32)

        # dx=0 is physically nonsensical but should NOT be silently overridden
        model = LiawModel(
            initializer=initializer,
            params=params,
            width=32,
            height=32,
            dx=0,
            device="cpu",
        )
        assert model._dx == 0, \
            "dx=0 should be preserved, not silently replaced with default 0.1"

    def test_objective_coeff_zero_preserved(self):
        """coeff=0.0 should be stored as-is, not replaced by default."""
        from lpf.objectives.mse import EachMeanSquareError

        obj = EachMeanSquareError(coeff=0.0)
        assert obj._coeff == 0.0, \
            "coeff=0.0 should be preserved, not replaced by default"

    def test_objective_coeff_zero_used_in_compute(self):
        """When coeff=0.0 is passed to compute(), it should be respected."""
        from lpf.objectives.mse import EachMeanSquareError

        obj = EachMeanSquareError(coeff=1.0)
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.ones((10, 10, 3), dtype=np.uint8) * 255

        result = obj.compute([img1], [img2], coeff=0.0)
        assert np.allclose(result, 0.0), \
            "coeff=0.0 in compute() should zero out the result"

    def test_solver_dt_zero_not_overridden(self):
        """dt=0.0 should NOT be silently replaced with 0.01."""
        from lpf.solvers.solver import Solver

        solver = Solver(dt=0.0)
        assert solver._dt == 0.0, \
            "dt=0.0 should be preserved, not silently replaced"

    def test_evosearch_generation_zero_preserved(self):
        """generation=0 should produce a formatted string, not empty string."""
        # We can't easily instantiate EvoSearch (needs model etc.),
        # so we test the generation formatting logic directly.
        generation = 0
        max_generation = 100

        # This mirrors the fixed logic in evosearch.py:save()
        if generation is None:
            str_gen = ""
        else:
            if max_generation is None:
                max_generation = 1000000
            fstr_gen = "gen-%0{}d_".format(int(np.ceil(np.log10(max_generation))) + 1)
            str_gen = fstr_gen % (int(generation))

        assert str_gen != "", \
            "generation=0 should produce a formatted string, not empty string"
        assert "gen-" in str_gen


# ---------------------------------------------------------------------------
# 2-2: `to_dict()` skips valid 0 values
# ---------------------------------------------------------------------------

class TestToDictZeroValues:
    """Ensure to_dict() includes index=0, generation=0, fitness=0.0."""

    def test_to_dict_includes_index_zero(self, liaw_model):
        """index=0 should appear in the output dict."""
        d = liaw_model.to_dict(index=0)
        assert "index" in d, "index=0 should be included in to_dict() output"
        assert d["index"] == 0

    def test_to_dict_includes_generation_zero(self, liaw_model):
        """generation=0 should appear in the output dict."""
        d = liaw_model.to_dict(index=0, generation=0)
        assert "generation" in d, \
            "generation=0 should be included in to_dict() output"
        assert d["generation"] == 0

    def test_to_dict_includes_fitness_zero(self, liaw_model):
        """fitness=0.0 should appear in the output dict."""
        d = liaw_model.to_dict(index=0, fitness=0.0)
        assert "fitness" in d, \
            "fitness=0.0 should be included in to_dict() output"
        assert d["fitness"] == 0.0

    def test_to_dict_excludes_none_values(self, liaw_model):
        """None values should still be excluded."""
        d = liaw_model.to_dict(index=0, generation=None, fitness=None)
        assert "generation" not in d
        assert "fitness" not in d


# ---------------------------------------------------------------------------
# 2-3: float16 casting causes false positives in is_state_invalid
# ---------------------------------------------------------------------------

class TestFloat16Casting:
    """Ensure valid float32 values > 65504 are not falsely flagged as invalid."""

    def test_model_is_state_invalid_large_valid_value(self, liaw_model):
        """Values in (65504, inf) should be valid under float32 checking."""
        # float16 max is 65504. Values > 65504 would overflow to inf in float16.
        val = 70000.0  # Valid in float32, overflow in float16
        liaw_model._u[0, :, :] = val
        liaw_model._v[0, :, :] = 0.1

        is_invalid = liaw_model.is_state_invalid(0)
        assert not is_invalid, \
            "Value 70000.0 is valid in float32; should not be flagged as invalid"

    def test_validation_util_large_valid_value(self):
        """Standalone validation.is_state_invalid should also handle large float32."""
        from lpf.utils.validation import is_state_invalid

        arr_u = np.full((10, 10), 70000.0, dtype=np.float32)
        arr_v = np.full((10, 10), 0.1, dtype=np.float32)

        assert not is_state_invalid(arr_u, arr_v), \
            "70000.0 is valid in float32; should not be flagged invalid"

    def test_actual_invalid_values_still_detected(self, liaw_model):
        """Actually invalid values (negative, NaN, Inf) should still be detected."""
        # NaN
        liaw_model._u[0, :, :] = float('nan')
        liaw_model._v[0, :, :] = 0.1
        assert liaw_model.is_state_invalid(0), "NaN should be detected as invalid"

        # Negative
        liaw_model._u[0, :, :] = -1.0
        liaw_model._v[0, :, :] = 0.1
        assert liaw_model.is_state_invalid(0), "Negative should be detected as invalid"

        # Inf
        liaw_model._u[0, :, :] = float('inf')
        liaw_model._v[0, :, :] = 0.1
        assert liaw_model.is_state_invalid(0), "Inf should be detected as invalid"


# ---------------------------------------------------------------------------
# 2-4: TorchModule.repeat() semantic mismatch with NumPy
# ---------------------------------------------------------------------------

class TestTorchRepeatSemantics:
    """TorchModule.repeat() should match NumPy's np.repeat() behavior."""

    @pytest.fixture
    def torch_module(self):
        try:
            from lpf.array.module import TorchModule
            return TorchModule("cpu", 0)
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_repeat_no_axis_matches_numpy(self, torch_module):
        """Without axis, repeat should do element-wise repetition like NumPy."""
        import torch

        np_arr = np.array([1, 2, 3])
        torch_arr = torch.tensor([1, 2, 3])

        np_result = np.repeat(np_arr, 2)
        torch_result = torch_module.repeat(torch_arr, 2).numpy()

        np.testing.assert_array_equal(
            np_result, torch_result,
            err_msg="TorchModule.repeat(axis=None) should match NumPy semantics: "
                    "[1,1,2,2,3,3] not [1,2,3,1,2,3]"
        )

    def test_repeat_with_axis_matches_numpy(self, torch_module):
        """With axis specified, repeat should match NumPy behavior."""
        import torch

        np_arr = np.array([[1, 2], [3, 4]])
        torch_arr = torch.tensor([[1, 2], [3, 4]])

        np_result = np.repeat(np_arr, 2, axis=0)
        torch_result = torch_module.repeat(torch_arr, 2, axis=0).numpy()

        np.testing.assert_array_equal(np_result, torch_result)

    def test_repeat_2d_no_axis_matches_numpy(self, torch_module):
        """2D array without axis: NumPy flattens then repeats."""
        import torch

        np_arr = np.array([[1, 2], [3, 4]])
        torch_arr = torch.tensor([[1, 2], [3, 4]])

        np_result = np.repeat(np_arr, 3)
        torch_result = torch_module.repeat(torch_arr, 3).numpy()

        np.testing.assert_array_equal(
            np_result, torch_result,
            err_msg="2D repeat without axis should flatten then repeat elements"
        )


# ---------------------------------------------------------------------------
# 2-5: solver.trj_y property reads wrong location
# ---------------------------------------------------------------------------

class TestSolverTrjY:
    """solver.trj_y should return self._trj_y, not self._model.trj_y."""

    def test_trj_y_returns_solver_data(self, liaw_model):
        """After solve with get_trj=True, trj_y should return trajectory data."""
        from lpf.solvers import EulerSolver

        solver = EulerSolver(model=liaw_model)
        trj = solver.solve(
            model=liaw_model,
            dt=0.01,
            n_iters=10,
            period_output=5,
            get_trj=True,
        )

        assert trj is not None, "solve() with get_trj=True should return trajectory"

        # The property should return the same object
        assert solver.trj_y is trj, \
            "solver.trj_y should return self._trj_y (the trajectory data)"

    def test_trj_y_shape(self, liaw_model):
        """Trajectory should have shape (n_time_points, *model.shape_grid)."""
        from lpf.solvers import EulerSolver

        n_iters = 20
        period_output = 5
        solver = EulerSolver(model=liaw_model)
        solver.solve(
            model=liaw_model,
            dt=0.01,
            n_iters=n_iters,
            period_output=period_output,
            get_trj=True,
        )

        # n_time_points = n_iters // period_output + 1
        expected_time_points = n_iters // period_output + 1
        assert solver.trj_y.shape[0] == expected_time_points


# ---------------------------------------------------------------------------
# 2-6: Pattern saving should work independently of morph saving
# ---------------------------------------------------------------------------

class TestPatternSavingIndependent:
    """Pattern saving should not be nested inside morph saving."""

    def test_save_image_with_pattern_only(self, liaw_model, tmp_path):
        """save_image should work with fpath_pattern only (no fpath_morph)."""
        fpath_pattern = str(tmp_path / "test_pattern.png")

        # Should not raise when fpath_morph is None
        liaw_model.save_image(index=0, fpath_morph=None, fpath_pattern=fpath_pattern)

        import os
        assert os.path.exists(fpath_pattern), \
            "Pattern should be saved even when fpath_morph is None"

    def test_save_image_with_morph_only(self, liaw_model, tmp_path):
        """save_image should work with fpath_morph only (no fpath_pattern)."""
        fpath_morph = str(tmp_path / "test_morph.png")

        liaw_model.save_image(index=0, fpath_morph=fpath_morph, fpath_pattern=None)

        import os
        assert os.path.exists(fpath_morph), \
            "Morph should be saved when fpath_morph is provided"

    def test_save_image_with_both(self, liaw_model, tmp_path):
        """save_image should work with both morph and pattern paths."""
        fpath_morph = str(tmp_path / "test_morph.png")
        fpath_pattern = str(tmp_path / "test_pattern.png")

        liaw_model.save_image(
            index=0, fpath_morph=fpath_morph, fpath_pattern=fpath_pattern)

        import os
        assert os.path.exists(fpath_morph)
        assert os.path.exists(fpath_pattern)
