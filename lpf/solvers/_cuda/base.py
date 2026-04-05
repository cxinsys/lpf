"""
Base class for CUDA-accelerated solvers.

Execution priority (auto-selected):
  1. Native .so  — entire solve loop in C/CUDA, zero Python during iteration
  2. Python loop — tight loop calling CUDA kernel launches via CuPy

Both paths support ALL features: trajectory, waypoints, early stopping,
verbose, AND file I/O (morph/pattern/states saving).

File I/O strategy: the CUDA loop captures state at every output point,
then Python writes all files AFTER the loop completes.  The solve loop
itself is never interrupted.
"""

import gc
import os
import time
from os.path import join as pjoin

import numpy as np

from lpf.solvers.solver import Solver
from lpf.kernels.compiler import get_kernel_manager


# ---- lazy native singleton ----
_native_solver = None

def _get_native():
    global _native_solver
    if _native_solver is None:
        try:
            from lpf.kernels.aot.native import NativeSolver
            _native_solver = NativeSolver()
        except Exception:
            from types import SimpleNamespace
            _native_solver = SimpleNamespace(is_available=lambda: False)
    return _native_solver


_SOLVER_CODES = {
    "CuEulerSolver": 0,
    "CuHeunSolver": 1,
    "CuRungekuttaSolver": 2,
    "CuRK23Solver": 3,
    "CuAdamsBashforth2Solver": 4,
}


class CuSolverBase(Solver):
    """Base for CUDA-kernel solvers with unified fast-path solve().

    Subclasses must implement:
        _alloc_buffers(model)
        _fast_step(model, t, dt, y_cur, y_next)
        _get_work_bufs()
    """

    def __init__(self, *args, fast_math=False, block_size=256, **kwargs):
        super().__init__(*args, **kwargs)
        self._fast_math = fast_math
        self._block_size = block_size
        self._km = None
        self._bufs_ready = False

    def _ensure_cuda(self, model):
        if self._km is None or self._km._model_name != model.name:
            self._km = get_kernel_manager(
                model.name, model.dtype, self._fast_math)
            self._bufs_ready = False

    # ---- PyTorch ↔ CuPy zero-copy bridge ----

    @staticmethod
    def _maybe_bridge_torch(model):
        """If model is PyTorch CUDA, replace its arrays with CuPy views.
        Returns a restore dict (or None if no bridge was needed)."""
        from lpf.array.module import TorchModule
        if not isinstance(model.am, TorchModule):
            return None
        import torch
        if not (hasattr(model.am, '_device')
                and 'cuda' in str(model.am._device)):
            return None

        import cupy as cp

        saved = {
            'y_mesh': model._y_mesh,
            'params': model._params,
            'u': model._u,
            'v': model._v,
        }

        model._y_mesh = cp.from_dlpack(model._y_mesh.detach())
        model._params = cp.from_dlpack(model._params.detach())
        model._u = model._y_mesh[0, :, :, :]
        model._v = model._y_mesh[1, :, :, :]

        return saved

    @staticmethod
    def _restore_torch(model, saved):
        """Restore PyTorch references after CUDA solve."""
        if saved is None:
            return
        # GPU memory was modified in-place (shared via DLPack).
        # Just restore the PyTorch tensor references.
        model._y_mesh = saved['y_mesh']
        model._params = saved['params']
        model._u = saved['y_mesh'][0, :, :, :]
        model._v = saved['y_mesh'][1, :, :, :]

    # ---- subclass hooks ----
    def _alloc_buffers(self, model):
        raise NotImplementedError
    def _fast_step(self, model, t, dt, y_cur, y_next):
        raise NotImplementedError
    def _on_solve_begin(self):
        pass
    def _get_work_bufs(self):
        return []

    # ==================================================================
    # Unified solve
    # ==================================================================

    def solve(self,
              model=None, dt=None, n_iters=None, rtol=None,
              period_output=None,
              dpath_model=None, dpath_morph=None,
              dpath_pattern=None, dpath_states=None,
              init_model=True, iter_begin=0, iter_end=None,
              get_trj=False, trj_waypoints=None, verbose=0):

        import cupy as cp

        # ---- resolve params ----
        if model is None:
            if self._model is None:
                raise ValueError("model should be defined.")
            model = self._model

        if dt is None:
            dt = self._dt if self._dt is not None else 0.01

        if rtol is None and self._rtol is not None:
            rtol = self._rtol
        if rtol is not None and rtol < 0:
            raise ValueError("rtol should be greater than 0.")

        if period_output is None and self._period_output is not None:
            period_output = self._period_output
        if period_output is not None and period_output < 1:
            raise ValueError("period_output should be greater than 0.")

        if n_iters is None and self._n_iters is not None:
            n_iters = self._n_iters

        if not model.has_initializer():
            raise ValueError("model should have an initializer.")

        if init_model:
            model.initialize()

        # ---- PyTorch CUDA bridge (zero-copy via DLPack) ----
        torch_bridge = self._maybe_bridge_torch(model)

        if iter_end is None:
            if n_iters is None:
                raise ValueError("n_iters should be defined.")
            iter_end = iter_begin + n_iters

        # ---- CUDA setup ----
        self._ensure_cuda(model)
        if not self._bufs_ready:
            self._alloc_buffers(model)
        self._on_solve_begin()

        has_file_io = any([dpath_morph, dpath_pattern, dpath_states])
        batch_size = model.batch_size
        actual_iters = iter_end - iter_begin

        # ---- compute ALL output waypoints ----
        # Merge user trj_waypoints + period_output points into one sorted list.
        if trj_waypoints is not None:
            output_iters = set(trj_waypoints)
            get_trj = True
        else:
            output_iters = set()

        if period_output is not None:
            for i in range(iter_begin, iter_end):
                if i == iter_begin or (i + 1) % period_output == 0:
                    output_iters.add(i + 1)

        output_iters_sorted = sorted(output_iters)
        n_output = len(output_iters_sorted)

        # ---- pre-create directories for file I/O ----
        dname_model = fstr_morph = fstr_pattern = fstr_states = None
        if has_file_io or dpath_model:
            dname_model = "model_%0{}d".format(
                int(np.floor(np.log10(batch_size))) + 1)

        if dpath_model:
            fstr_fname_model = "model_%0{}d.json".format(
                int(np.floor(np.log10(batch_size))) + 1)
            dict_solver = self.to_dict()
            dict_solver["solver"] = self.name
            dict_solver["dt"] = dt
            dict_solver["n_iters"] = n_iters
            if rtol is not None:
                dict_solver["rtol"] = rtol
            for j in range(batch_size):
                dpath_models = pjoin(dpath_model, "models")
                os.makedirs(dpath_models, exist_ok=True)
                model.save_model(index=j,
                                 fpath=pjoin(dpath_models, fstr_fname_model % (j+1)),
                                 initializer=model.initializer,
                                 solver=dict_solver)

        if dpath_morph:
            for j in range(batch_size):
                os.makedirs(pjoin(dpath_morph, dname_model % (j+1)), exist_ok=True)
            fstr_morph = "morph_%0{}d.png".format(
                int(np.floor(np.log10(n_iters))) + 1)

        if dpath_pattern:
            for j in range(batch_size):
                os.makedirs(pjoin(dpath_pattern, dname_model % (j+1)), exist_ok=True)
            fstr_pattern = "pattern_%0{}d.png".format(
                int(np.floor(np.log10(n_iters))) + 1)

        if dpath_states:
            for j in range(batch_size):
                os.makedirs(pjoin(dpath_states, dname_model % (j+1)), exist_ok=True)
            fstr_states = "states_%0{}d".format(
                int(np.floor(np.log10(n_iters))) + 1)

        # ---- allocate trajectory / snapshot buffer ----
        # Captures state at ALL output iterations (for both trj and file I/O).
        trj_y = None
        if n_output > 0:
            trj_y = cp.zeros((n_output, *model.shape_grid),
                             dtype=model.y_mesh.dtype)

        # ---- run the CUDA solve loop ----
        B, H, W = model.batch_size, model.height, model.width
        dx2_inv = 1.0 / (model.dx ** 2)

        native = _get_native()
        can_native = (
            native.is_available()
            and self._name in _SOLVER_CODES
        )

        t_total = time.time()

        if can_native and n_output > 0:
            result = self._solve_native(
                model, dt, dx2_inv, actual_iters, B, H, W,
                trj_y, output_iters_sorted, rtol)
        elif can_native and n_output == 0:
            result = self._solve_native(
                model, dt, dx2_inv, actual_iters, B, H, W,
                None, None, rtol)
        else:
            result = self._solve_python_loop(
                model, dt, dx2_inv, iter_begin, iter_end,
                output_iters_sorted, trj_y, rtol, verbose)

        # ---- post-loop: write files from captured snapshots ----
        if has_file_io and trj_y is not None:
            self._write_files(
                model, trj_y, output_iters_sorted, batch_size,
                dpath_morph, dpath_pattern, dpath_states,
                dname_model, fstr_morph, fstr_pattern, fstr_states)

        if verbose >= 1:
            print("- [Duration] : %.5e sec." % (time.time() - t_total))

        # ---- restore PyTorch references if bridged ----
        self._restore_torch(model, torch_bridge)

        gc.collect()

        # ---- return trajectory if requested ----
        if get_trj and trj_y is not None:
            if trj_waypoints is not None:
                return {"iters": sorted(trj_waypoints), "trj": trj_y}
            return trj_y

        return result

    # ==================================================================
    # Native .so path
    # ==================================================================

    def _solve_native(self, model, dt, dx2_inv, n_iters, B, H, W,
                      trj_y, output_iters_sorted, rtol):
        import cupy as cp

        native = _get_native()
        solver_code = _SOLVER_CODES[self._name]

        y_a = model.y_mesh
        y_b = cp.empty_like(y_a)

        trj_iters_np = None
        if output_iters_sorted:
            trj_iters_np = np.array(output_iters_sorted, dtype=np.int32)

        res = native.solve(
            solver_type=solver_code,
            model_name=model.name,
            dtype=model.dtype,
            y_a=y_a, y_b=y_b, params=model.params,
            B=B, H=H, W=W, dt=dt, dx2_inv=dx2_inv,
            n_iters=n_iters,
            work_bufs=self._get_work_bufs(),
            trj_buf=trj_y,
            trj_iters=trj_iters_np,
            rtol=rtol if rtol else 0.0,
            rtol_period=1000,
        )

        final = y_a if res['result_buf'] == 0 else y_b
        cp.cuda.Stream.null.synchronize()
        with model.am:
            model._y_mesh[...] = final
            model._u = model._y_mesh[0, :, :, :]
            model._v = model._y_mesh[1, :, :, :]

        return res

    # ==================================================================
    # Python fast-path loop (fallback when .so not available)
    # ==================================================================

    def _solve_python_loop(self, model, dt, dx2_inv,
                           iter_begin, iter_end,
                           output_iters_sorted, trj_y, rtol, verbose):
        import cupy as cp

        output_set = set(output_iters_sorted) if output_iters_sorted else set()

        y_cur = model.y_mesh
        y_next = cp.empty_like(y_cur)

        ix_trj = 0
        t = 0.0
        t_seg = time.time()
        n_tp = len(output_iters_sorted) if output_iters_sorted else 0

        for i in range(iter_begin, iter_end):
            t += dt
            self._fast_step(model, t, dt, y_cur, y_next)
            y_cur, y_next = y_next, y_cur

            if (i + 1) in output_set:
                if trj_y is not None and ix_trj < n_tp:
                    trj_y[ix_trj, ...] = y_cur
                    ix_trj += 1
                if verbose >= 1:
                    cp.cuda.Stream.null.synchronize()
                    print("- [Iteration #%d] elapsed time: %.5e sec."
                          % (i + 1, time.time() - t_seg))
                    t_seg = time.time()

            if rtol is not None and (i + 1) % 1000 == 0:
                diff = float(cp.max(cp.abs(y_cur - y_next)))
                scale = float(cp.max(cp.abs(y_next))) + 1e-30
                if diff / scale < rtol:
                    if verbose >= 1:
                        print("- [Early stopping at #%d]" % (i + 1))
                    break

        cp.cuda.Stream.null.synchronize()
        with model.am:
            model._y_mesh[...] = y_cur
            model._u = model._y_mesh[0, :, :, :]
            model._v = model._y_mesh[1, :, :, :]

        return None

    # ==================================================================
    # Post-loop file I/O — writes all captured snapshots
    # ==================================================================

    @staticmethod
    def _write_files(model, trj_y, output_iters, batch_size,
                     dpath_morph, dpath_pattern, dpath_states,
                     dname_model, fstr_morph, fstr_pattern, fstr_states):
        """Write image/state files from captured trajectory snapshots."""
        for snap_idx, iter_num in enumerate(output_iters):
            # Temporarily set model state to the snapshot
            snapshot = trj_y[snap_idx]
            u_snap = snapshot[0]
            v_snap = snapshot[1]

            if dpath_morph or dpath_pattern:
                arr_color = model.colorize(arr_u=u_snap)
                for j in range(batch_size):
                    fpath_m = None
                    if dpath_morph:
                        fpath_m = pjoin(dpath_morph,
                                        dname_model % (j + 1),
                                        fstr_morph % iter_num)
                    fpath_p = None
                    if dpath_pattern:
                        fpath_p = pjoin(dpath_pattern,
                                        dname_model % (j + 1),
                                        fstr_pattern % iter_num)
                    model.save_image(index=j,
                                     fpath_morph=fpath_m,
                                     fpath_pattern=fpath_p,
                                     arr_color=arr_color)

            if dpath_states:
                for j in range(batch_size):
                    fpath_s = pjoin(dpath_states,
                                    dname_model % (j + 1),
                                    fstr_states % iter_num)
                    model.save_states(index=j, fpath=fpath_s,
                                      u=u_snap, v=v_snap)
