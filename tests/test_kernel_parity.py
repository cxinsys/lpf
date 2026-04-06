"""Test that CPU, JIT, and AOT paths produce identical results.

Verifies numerical parity across all execution paths:
  - CPU (NumPy, Python loop)
  - CUDA JIT (shared-memory tiled, NVRTC-compiled)
  - CUDA AOT (pre-compiled fatbin, if available)
  - PyTorch CUDA (DLPack bridge)

Uses 16 parameter sets from init_pop_01 as a batch test.
Tests float32 and float64 precision with n_iters=100.

Note on AB2 tolerance
---------------------
AdamsBashforth2Solver uses absolute tolerance (N * eps * max_val) instead of
relative tolerance (N * eps) used by other solvers.  This is because AB2 has a
smaller stability region than single-step methods (Euler, RK4, Heun, RK23).

With dt=0.01, some init_pop_01 parameter sets (batch 7, 12: Dv~0.138) sit
near the AB2 stability boundary (dt_crit ~ 4*Dv/dx^2 ~ 0.018).  The solution
remains bounded but oscillates with larger amplitude than Euler (e.g. v_max
15 vs 4), and these oscillations amplify the CPU-GPU rounding difference
(FMA, operation order) by orders of magnitude.

This is NOT a bug in AB2: reducing dt (e.g. dt=0.005) eliminates the
oscillation and brings the CPU-GPU diff back to 1-2 ULP.  The test uses
absolute tolerance to accommodate this expected behavior without masking
real implementation errors.
"""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from tests.conftest import (
    make_liaw_model_16, get_numpy,
    LIAW_BATCH_SIZE_16, LIAW_WIDTH_16, LIAW_HEIGHT_16,
)

ALL_SOLVERS = ["EulerSolver", "HeunSolver", "RungeKuttaSolver",
               "RK23Solver", "AdamsBashforth2Solver"]

# dt=0.005 keeps all 16 init_pop_01 batches within every solver's
# stability region (most restrictive: batch 2, Dv=0.228,
# dt_crit_ab2 ~ 0.011).  200 iters gives T=1.0.
DT = 0.005
N_ITERS = 200


def _get_solver_class(name):
    import lpf.solvers as S
    return getattr(S, name)


# ---------- helpers ----------

def _solve(model, solver_name="EulerSolver", n_iters=N_ITERS, dt=DT):
    cls = _get_solver_class(solver_name)
    cls(dt=dt, n_iters=n_iters).solve(model, verbose=0)
    return get_numpy(model)


def _force_jit(model):
    """Patch the CUDA solver to bypass AOT and use JIT kernels."""
    from lpf.kernels.compiler import _KERNEL_CACHE
    _KERNEL_CACHE.clear()

    from lpf.solvers._cuda.euler import CuEulerSolver
    cu_solver = CuEulerSolver(dt=DT, n_iters=N_ITERS)
    cu_solver._ensure_cuda(model)
    cu_solver._km._use_aot = False
    cu_solver._km._aot = None

    key = (model.name, np.dtype(model.dtype).str, False)
    _KERNEL_CACHE[key] = cu_solver._km


def _max_ulp(a, b, dtype):
    """Max difference in ULPs (Units in the Last Place)."""
    eps = np.finfo(dtype).eps
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    return np.max(diff / scale) / eps


# ---------- dtype parity: CPU vs GPU across all solvers ----------

class TestDtypeParity:
    """CPU and GPU must produce matching dtype and values for n_iters=100."""

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cpu_gpu_dtype_match(self, solver_name, dtype):
        """Output dtype must match input dtype on both CPU and GPU."""
        y_cpu = _solve(make_liaw_model_16("cpu", dtype=dtype), solver_name)
        y_gpu = _solve(make_liaw_model_16("cuda:0", dtype=dtype), solver_name)

        assert y_cpu.dtype == dtype, f"CPU output dtype {y_cpu.dtype} != {dtype}"
        assert y_gpu.dtype == dtype, f"GPU output dtype {y_gpu.dtype} != {dtype}"

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cpu_gpu_relative_error(self, solver_name, dtype):
        """CPU vs GPU relative error must stay near machine epsilon."""
        y_cpu = _solve(make_liaw_model_16("cpu", dtype=dtype), solver_name)
        y_gpu = _solve(make_liaw_model_16("cuda:0", dtype=dtype), solver_name)

        valid = np.isfinite(y_cpu) & np.isfinite(y_gpu)
        if not valid.any():
            pytest.skip("All values non-finite")

        max_abs = np.max(np.abs(y_cpu[valid] - y_gpu[valid]))
        eps = np.finfo(dtype).eps

        # Absolute tolerance: N * eps * max_val.
        #
        # For well-conditioned solvers (Euler, RK4, Heun, RK23), errors
        # accumulate as O(N * eps) and this bound is conservative.
        #
        # For AB2, some init_pop_01 batches (7, 12) sit near the AB2
        # stability boundary at dt=0.01 (dt_crit ~ 0.018 for Dv~0.138).
        # The solution oscillates with larger amplitude, amplifying the
        # CPU-GPU rounding difference.  Absolute tolerance absorbs this
        # without requiring a separate tolerance per solver.
        # See module docstring for details.
        max_val = np.max(np.abs(y_cpu[valid]))
        tol = N_ITERS * eps * max(max_val, 1.0)
        assert max_abs < tol, (
            f"{solver_name} {dtype.__name__}: max abs diff {max_abs:.2e} "
            f"exceeds {tol:.2e} (N*eps*max_val)"
        )


# ---------- JIT vs AOT parity ----------

class TestJitAotParity:
    """JIT and AOT kernels must produce identical results."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_jit_matches_cpu(self, dtype):
        y_cpu = _solve(make_liaw_model_16("cpu", dtype=dtype))

        model_gpu = make_liaw_model_16("cuda:0", dtype=dtype)
        _force_jit(model_gpu)
        from lpf.solvers import EulerSolver
        EulerSolver(dt=DT, n_iters=N_ITERS).solve(model_gpu, verbose=0)
        y_jit = get_numpy(model_gpu)

        assert y_jit.dtype == dtype
        valid = np.isfinite(y_cpu) & np.isfinite(y_jit)
        diff = np.abs(y_cpu[valid] - y_jit[valid])
        scale = np.maximum(np.abs(y_cpu[valid]), 1e-30)
        max_rel = np.max(diff / scale)
        eps = np.finfo(dtype).eps
        tol = N_ITERS * eps
        assert max_rel < tol, (
            f"JIT vs CPU: max rel error {max_rel:.2e} ({dtype.__name__})")

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_aot_matches_cpu(self, dtype):
        from lpf.kernels.aot.loader import AOTKernelLoader
        if not AOTKernelLoader().is_available():
            pytest.skip("AOT kernels not built")

        from lpf.kernels.compiler import _KERNEL_CACHE
        _KERNEL_CACHE.clear()

        y_cpu = _solve(make_liaw_model_16("cpu", dtype=dtype))
        y_aot = _solve(make_liaw_model_16("cuda:0", dtype=dtype))

        assert y_aot.dtype == dtype
        valid = np.isfinite(y_cpu) & np.isfinite(y_aot)
        diff = np.abs(y_cpu[valid] - y_aot[valid])
        scale = np.maximum(np.abs(y_cpu[valid]), 1e-30)
        max_rel = np.max(diff / scale)
        eps = np.finfo(dtype).eps
        tol = N_ITERS * eps
        assert max_rel < tol, (
            f"AOT vs CPU: max rel error {max_rel:.2e} ({dtype.__name__})")

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_jit_matches_aot(self, dtype):
        from lpf.kernels.aot.loader import AOTKernelLoader
        if not AOTKernelLoader().is_available():
            pytest.skip("AOT kernels not built")

        from lpf.kernels.compiler import _KERNEL_CACHE
        _KERNEL_CACHE.clear()
        y_aot = _solve(make_liaw_model_16("cuda:0", dtype=dtype))

        _KERNEL_CACHE.clear()
        model_jit = make_liaw_model_16("cuda:0", dtype=dtype)
        _force_jit(model_jit)
        from lpf.solvers import EulerSolver
        EulerSolver(dt=DT, n_iters=N_ITERS).solve(model_jit, verbose=0)
        y_jit = get_numpy(model_jit)

        assert y_jit.dtype == dtype
        assert y_aot.dtype == dtype
        np.testing.assert_array_equal(y_jit, y_aot,
                                      err_msg=f"JIT vs AOT not bit-exact ({dtype.__name__})")


# ---------- JAX kernel parity ----------

class TestJaxParity:
    """JAX kernels (CPU + GPU) must match CuPy CUDA fused kernel results.

    These tests document the cross-backend numerical agreement that the
    JAX dispatch path is expected to maintain.  Skip cleanly if JAX is
    not installed.
    """

    @pytest.fixture(autouse=True)
    def _require_jax(self):
        pytest.importorskip("jax", reason="JAX not installed")

    @staticmethod
    def _jax_devices():
        import jax
        devs = ["jax:cpu"]
        try:
            if jax.devices("gpu"):
                devs.append("jax:gpu:0")
        except Exception:
            pass
        return devs

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    @pytest.mark.parametrize("dtype", [np.float32])
    def test_jax_cpu_matches_cpu(self, solver_name, dtype):
        """JAX CPU path must match the NumPy CPU baseline within N*eps."""
        y_cpu = _solve(make_liaw_model_16("cpu", dtype=dtype), solver_name)
        y_jax = _solve(make_liaw_model_16("jax:cpu", dtype=dtype), solver_name)

        valid = np.isfinite(y_cpu) & np.isfinite(y_jax)
        if not valid.any():
            pytest.skip("All values non-finite")

        max_abs = np.max(np.abs(y_cpu[valid] - y_jax[valid]))
        eps = np.finfo(dtype).eps
        max_val = np.max(np.abs(y_cpu[valid]))
        tol = N_ITERS * eps * max(max_val, 1.0)
        assert max_abs < tol, (
            f"{solver_name} {dtype.__name__}: JAX CPU vs NumPy CPU "
            f"max abs diff {max_abs:.2e} exceeds {tol:.2e}"
        )

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    @pytest.mark.parametrize("dtype", [np.float32])
    def test_jax_gpu_matches_cpu(self, solver_name, dtype):
        """JAX GPU path must match the NumPy CPU baseline within N*eps."""
        if "jax:gpu:0" not in self._jax_devices():
            pytest.skip("JAX GPU not available")

        y_cpu = _solve(make_liaw_model_16("cpu", dtype=dtype), solver_name)
        y_jax = _solve(
            make_liaw_model_16("jax:gpu:0", dtype=dtype), solver_name)

        valid = np.isfinite(y_cpu) & np.isfinite(y_jax)
        if not valid.any():
            pytest.skip("All values non-finite")

        max_abs = np.max(np.abs(y_cpu[valid] - y_jax[valid]))
        eps = np.finfo(dtype).eps
        max_val = np.max(np.abs(y_cpu[valid]))
        tol = N_ITERS * eps * max(max_val, 1.0)
        assert max_abs < tol, (
            f"{solver_name} {dtype.__name__}: JAX GPU vs NumPy CPU "
            f"max abs diff {max_abs:.2e} exceeds {tol:.2e}"
        )

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    @pytest.mark.parametrize("dtype", [np.float32])
    def test_jax_gpu_matches_cupy(self, solver_name, dtype):
        """JAX GPU and CuPy CUDA fused kernel must agree within N*eps."""
        if "jax:gpu:0" not in self._jax_devices():
            pytest.skip("JAX GPU not available")

        y_cu = _solve(make_liaw_model_16("cuda:0", dtype=dtype), solver_name)
        y_jax = _solve(
            make_liaw_model_16("jax:gpu:0", dtype=dtype), solver_name)

        valid = np.isfinite(y_cu) & np.isfinite(y_jax)
        if not valid.any():
            pytest.skip("All values non-finite")

        max_abs = np.max(np.abs(y_cu[valid] - y_jax[valid]))
        eps = np.finfo(dtype).eps
        max_val = np.max(np.abs(y_cu[valid]))
        tol = N_ITERS * eps * max(max_val, 1.0)
        assert max_abs < tol, (
            f"{solver_name} {dtype.__name__}: JAX GPU vs CuPy CUDA "
            f"max abs diff {max_abs:.2e} exceeds {tol:.2e}"
        )


# ---------- PyTorch CUDA parity ----------

class TestTorchParity:
    """PyTorch CUDA backend must match CuPy results."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_torch_matches_cpu(self, dtype):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        y_cpu = _solve(make_liaw_model_16("cpu", dtype=dtype))
        y_torch = _solve(make_liaw_model_16("torch:gpu:0", dtype=dtype))

        assert y_torch.dtype == dtype
        valid = np.isfinite(y_cpu) & np.isfinite(y_torch)
        diff = np.abs(y_cpu[valid] - y_torch[valid])
        scale = np.maximum(np.abs(y_cpu[valid]), 1e-30)
        max_rel = np.max(diff / scale)
        eps = np.finfo(dtype).eps
        tol = N_ITERS * eps
        assert max_rel < tol, (
            f"Torch vs CPU: max rel error {max_rel:.2e} ({dtype.__name__})")

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_torch_matches_cupy(self, dtype):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for torch")

        y_cupy = _solve(make_liaw_model_16("cuda:0", dtype=dtype))
        y_torch = _solve(make_liaw_model_16("torch:gpu:0", dtype=dtype))

        np.testing.assert_array_equal(y_cupy, y_torch,
                                      err_msg=f"CuPy vs Torch not bit-exact ({dtype.__name__})")


# ---------- cross-solver consistency ----------

class TestCrossSolverConsistency:
    """Different solvers must converge to similar solutions.

    All solvers integrate the same PDE, so their results must agree
    within each solver's truncation error order:
      - Euler (order 1): O(dt)   ~ 1e-2
      - AB2   (order 2): O(dt^2) ~ 1e-4
      - Heun  (order 2): O(dt^2) ~ 1e-4
      - RK23  (order 3): O(dt^3) ~ 1e-6
      - RK4   (order 4): O(dt^4) ~ 1e-8
    """

    # Use dt=0.0001 so truncation error is small enough to compare
    # solvers meaningfully.  At this dt, all 16 init_pop_01 batches
    # are well within every solver's stability region (most restrictive:
    # batch 2, Dv=0.228, dt_crit_ab2 ~ 0.011 >> 0.0001).
    #
    # 100 iterations at dt=0.0001 gives T=0.01, enough to observe the
    # convergence order hierarchy:
    #   Euler vs RK4:  O(dt)   ~ 5e-4
    #   Heun  vs RK4:  O(dt^2) ~ 4e-7
    #   AB2   vs RK4:  O(dt^2) ~ 4e-6
    #   RK23  vs RK4:  O(dt^3) ~ 8e-10
    _CS_DT = 0.0001
    _CS_ITERS = 100

    def _cs_solve(self, device, solver_name, dtype):
        cls = _get_solver_class(solver_name)
        m = make_liaw_model_16(device, dtype=dtype)
        cls(dt=self._CS_DT, n_iters=self._CS_ITERS).solve(m, verbose=0)
        return get_numpy(m)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_high_order_solvers_agree(self, dtype):
        """Heun, RK4, and RK23 (order 2-4) must converge to similar solutions."""
        y_heun = self._cs_solve("cuda:0", "HeunSolver", dtype)
        y_rk4 = self._cs_solve("cuda:0", "RungeKuttaSolver", dtype)
        y_rk23 = self._cs_solve("cuda:0", "RK23Solver", dtype)

        # float32 precision (~1e-7) limits how small the difference can be,
        # regardless of solver order.  float64 can resolve finer differences.
        eps = np.finfo(dtype).eps
        for name_a, y_a, name_b, y_b, tol in [
            ("Heun", y_heun, "RK4",  y_rk4,  max(1e-5, 100 * eps)),  # O(dt^2)
            ("Heun", y_heun, "RK23", y_rk23, max(1e-5, 100 * eps)),  # O(dt^2)
            ("RK4",  y_rk4,  "RK23", y_rk23, max(1e-7, 100 * eps)),  # O(dt^3)
        ]:
            max_diff = np.max(np.abs(y_a - y_b))
            assert max_diff < tol, (
                f"{name_a} vs {name_b} ({dtype.__name__}): "
                f"max diff {max_diff:.2e} exceeds {tol:.0e}"
            )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_euler_within_first_order(self, dtype):
        """Euler (order 1) vs RK4: difference bounded by O(dt) ~ 5e-4."""
        y_euler = self._cs_solve("cuda:0", "EulerSolver", dtype)
        y_rk4 = self._cs_solve("cuda:0", "RungeKuttaSolver", dtype)

        max_diff = np.max(np.abs(y_euler - y_rk4))
        assert max_diff < 1e-2, (
            f"Euler vs RK4 ({dtype.__name__}): "
            f"max diff {max_diff:.2e} exceeds 1e-2"
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_ab2_within_second_order(self, dtype):
        """AB2 (order 2) vs RK4: difference bounded by O(dt^2) ~ 4e-6."""
        y_ab2 = self._cs_solve("cuda:0", "AdamsBashforth2Solver", dtype)
        y_rk4 = self._cs_solve("cuda:0", "RungeKuttaSolver", dtype)

        max_diff = np.max(np.abs(y_ab2 - y_rk4))
        assert max_diff < 1e-4, (
            f"AB2 vs RK4 ({dtype.__name__}): "
            f"max diff {max_diff:.2e} exceeds 1e-4"
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_solvers_converge_at_500k_iters(self, dtype):
        """All CUDA solvers must converge to similar solutions over 500K iterations.

        Uses dt=0.005 to keep all batches within AB2 stability region.
        Compares only batches where all solvers remain finite and bounded,
        since some parameter sets have strong reactions that cause large
        transient differences even with correct implementations.
        """
        dt = 0.005
        n_iters = 500000
        solver_names = ["EulerSolver", "HeunSolver", "RungeKuttaSolver",
                         "RK23Solver", "AdamsBashforth2Solver"]

        results = {}
        for name in solver_names:
            cls = _get_solver_class(name)
            m = make_liaw_model_16("cuda:0", dtype=dtype)
            cls(dt=dt, n_iters=n_iters).solve(m, verbose=0)
            results[name] = get_numpy(m)

        ref = results["RungeKuttaSolver"]

        # Select batches where ALL solvers are finite and bounded
        n_batches = ref.shape[1]
        stable = []
        for b in range(n_batches):
            if all(np.all(np.isfinite(r[:, b])) and np.max(np.abs(r[:, b])) < 100
                   for r in results.values()):
                stable.append(b)
        assert len(stable) >= 8, f"Only {len(stable)} stable batches, expected >= 8"

        # High-order solvers (Heun, RK23) must be close to RK4
        for name in ["HeunSolver", "RK23Solver"]:
            max_diff = np.max(np.abs(results[name][:, stable] - ref[:, stable]))
            assert max_diff < 1e-2, (
                f"{name} vs RK4 at 500K iters ({dtype.__name__}): "
                f"max diff {max_diff:.2e} in {len(stable)} stable batches"
            )

        # Euler (1st order) and AB2 (2nd order) may have larger truncation
        # error but must still be in the same ballpark
        for name in ["EulerSolver", "AdamsBashforth2Solver"]:
            max_diff = np.max(np.abs(results[name][:, stable] - ref[:, stable]))
            assert max_diff < 1.0, (
                f"{name} vs RK4 at 500K iters ({dtype.__name__}): "
                f"max diff {max_diff:.2e} in {len(stable)} stable batches"
            )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_ab2_first_step_equals_euler(self, dtype):
        """AB2 bootstraps with Euler, so first step must be identical."""
        y_euler = _solve(make_liaw_model_16("cuda:0", dtype=dtype), "EulerSolver", n_iters=1)
        y_ab2 = _solve(make_liaw_model_16("cuda:0", dtype=dtype), "AdamsBashforth2Solver", n_iters=1)

        np.testing.assert_array_equal(y_euler, y_ab2,
                                      err_msg=f"AB2 first step != Euler ({dtype.__name__})")


# ---------- shape and finiteness ----------

class TestBasicSanity:
    """Basic checks: shape, dtype, finiteness."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cpu_output_shape_and_finite(self, dtype):
        y = _solve(make_liaw_model_16("cpu", dtype=dtype))
        assert y.shape == (2, LIAW_BATCH_SIZE_16, LIAW_HEIGHT_16, LIAW_WIDTH_16)
        assert y.dtype == dtype
        assert np.all(np.isfinite(y))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_gpu_output_shape_and_finite(self, dtype):
        y = _solve(make_liaw_model_16("cuda:0", dtype=dtype))
        assert y.shape == (2, LIAW_BATCH_SIZE_16, LIAW_HEIGHT_16, LIAW_WIDTH_16)
        assert y.dtype == dtype
        assert np.all(np.isfinite(y))
