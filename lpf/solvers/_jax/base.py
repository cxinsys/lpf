"""JaxSolverBase — unified JIT+scan solve loop for all JAX-accelerated solvers.

Execution model:
    1. Build a single Python closure that runs ``n_iters`` steps via
       ``lax.scan``.  It captures the RHS function, the step function, and
       the trajectory output policy (period_output / waypoints).
    2. ``jax.jit`` compiles it once per unique (model, step, n_iters,
       output-pattern, record-traj, rtol-on/off) signature.  Subsequent
       calls with the same signature reuse the compiled kernel — so
       repeated solves are free.
    3. One call into the compiled kernel → single XLA program → single
       GPU/TPU launch for all ``n_iters`` steps.

Supports the full ``Solver.solve()`` API:
    - n_iters / iter_end / iter_begin
    - dt, rtol (rtol: early-stopping check every 1000 steps, matching CPU)
    - period_output, trj_waypoints, get_trj
    - dpath_morph / dpath_pattern / dpath_states / dpath_model
    - init_model, verbose
"""

import gc
import os
import time
from os.path import join as pjoin

import numpy as np

from lpf.solvers.solver import Solver
from lpf.solvers._jax.rhs import (get_rhs, is_supported as rhs_supported,
                                  build_diploid_rhs, is_diploid_model,
                                  is_diploid_supported)
from lpf.solvers._jax.steps import get_step, is_supported as step_supported


# ---------------------------------------------------------------------------
# Compile cache: (model_name, solver_name, n_iters, dt, dx_inv2, n_record,
#                 rtol_on) -> compiled callable
# ---------------------------------------------------------------------------

_COMPILE_CACHE = {}


def _ensure_jax_imports():
    import jax
    import jax.numpy as jnp
    from jax import lax
    return jax, jnp, lax


def _build_integrator_no_trj(rhs_fn, solver_name, n_iters):
    """Fast path: no trajectory output.

    Uses ``lax.fori_loop`` and returns only ``y_final``. Avoids the scan
    output tensor entirely, so the compiled program's GPU memory footprint
    is O(state size) regardless of n_iters.
    """
    _, jnp, lax = _ensure_jax_imports()
    step_fn, _ = get_step(solver_name)

    def integrate(y0, params, state0, dt, dx_inv2):
        def body(i, carry):
            y, state = carry
            y_new, new_state = step_fn(rhs_fn, y, params, dt, dx_inv2, state)
            return (y_new, new_state)

        y_final, _ = lax.fori_loop(0, n_iters, body, (y0, state0))
        return y_final

    return integrate


def _build_integrator_with_trj(rhs_fn, solver_name):
    """Trajectory path: uses lax.scan so we can collect state on selected steps.

    ``rec_mask`` is a bool array of length ``n_iters``. True means "record
    state at the END of this step". We always pass all states through the
    scan output (padded with the current state when rec_mask is False) and
    let the caller slice by ``rec_mask`` afterwards.
    """
    _, jnp, lax = _ensure_jax_imports()
    step_fn, _ = get_step(solver_name)

    def integrate(y0, params, state0, dt, dx_inv2, rec_mask):
        def body(carry, rec_flag_unused):
            y, state = carry
            y_new, new_state = step_fn(rhs_fn, y, params, dt, dx_inv2, state)
            return (y_new, new_state), y_new

        (y_final, _), trj = lax.scan(body, (y0, state0), rec_mask)
        return y_final, trj

    return integrate


def _build_integrator_rtol_no_trj(rhs_fn, solver_name, n_iters):
    """rtol early-stop path without trajectory output."""
    _, jnp, lax = _ensure_jax_imports()
    step_fn, _ = get_step(solver_name)

    def integrate(y0, params, state0, dt, dx_inv2, rtol):
        def cond(carry):
            _, _, step_idx, stopped = carry
            return (step_idx < n_iters) & (~stopped)

        def body(carry):
            y, state, step_idx, stopped = carry
            y_old_abs_max = jnp.max(jnp.abs(y))
            y_new, new_state = step_fn(rhs_fn, y, params, dt, dx_inv2, state)
            delta_abs_max = jnp.max(jnp.abs(y_new - y))
            rel_change = delta_abs_max / (y_old_abs_max + 1e-30)
            do_check = ((step_idx + 1) % 1000 == 0)
            should_stop = do_check & (rel_change < rtol)
            return (y_new, new_state, step_idx + 1, stopped | should_stop)

        init = (y0, state0, jnp.int32(0), jnp.bool_(False))
        y_final, _, _, _ = lax.while_loop(cond, body, init)
        return y_final

    return integrate


def _build_integrator_rtol_with_trj(rhs_fn, solver_name):
    """rtol early-stop + trajectory output path."""
    _, jnp, lax = _ensure_jax_imports()
    step_fn, _ = get_step(solver_name)

    def integrate(y0, params, state0, dt, dx_inv2, rec_mask, rtol):
        def body(carry, _):
            y, state, step_idx, stopped = carry
            y_old_abs_max = jnp.max(jnp.abs(y))
            y_new, new_state = step_fn(rhs_fn, y, params, dt, dx_inv2, state)
            delta_abs_max = jnp.max(jnp.abs(y_new - y))
            rel_change = delta_abs_max / (y_old_abs_max + 1e-30)

            do_check = ((step_idx + 1) % 1000 == 0)
            should_stop = do_check & (rel_change < rtol)
            stopped_new = stopped | should_stop

            y_out = jnp.where(stopped_new, y, y_new)
            return (y_out, new_state, step_idx + 1, stopped_new), y_out

        init = (y0, state0, jnp.int32(0), jnp.bool_(False))
        (y_final, _, _, _), trj = lax.scan(body, init, rec_mask)
        return y_final, trj

    return integrate


# ---------------------------------------------------------------------------
# JaxSolverBase
# ---------------------------------------------------------------------------

class JaxSolverBase(Solver):
    """Base for JAX-JIT solvers; one subclass per integration method.

    Subclasses only need to set ``self._name`` via ``__init__``.
    """

    def __init__(self, solver_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = solver_name
        # Each solver instance keeps its own jit cache. We also share a
        # module-level cache keyed by signature to survive subclass
        # re-instantiation.

    # ------------------------------------------------------------------
    # Compile cache access
    # ------------------------------------------------------------------

    def _get_compiled(self, rhs_fn, cache_key, n_iters, has_rtol, want_trj,
                      dtype):
        """Return a jit-compiled integrator for the given RHS.

        ``cache_key`` identifies the RHS so that repeated calls on the
        same model/solver configuration reuse the compiled kernel.
        ``dtype`` is included so float32 and float64 models do not share
        a compiled body.
        """
        import jax
        key = (cache_key, self._name, n_iters, has_rtol, want_trj,
               np.dtype(dtype).str)
        fn = _COMPILE_CACHE.get(key)
        if fn is not None:
            return fn

        if want_trj:
            if has_rtol:
                builder = _build_integrator_rtol_with_trj(
                    rhs_fn, self._name)
            else:
                builder = _build_integrator_with_trj(rhs_fn, self._name)
        else:
            if has_rtol:
                builder = _build_integrator_rtol_no_trj(
                    rhs_fn, self._name, n_iters)
            else:
                builder = _build_integrator_no_trj(
                    rhs_fn, self._name, n_iters)

        fn = jax.jit(builder)
        _COMPILE_CACHE[key] = fn
        return fn

    # ------------------------------------------------------------------
    # Main entry point (replaces Solver.solve for JAX models)
    # ------------------------------------------------------------------

    def solve(self,
              model=None, dt=None, n_iters=None, rtol=None,
              period_output=None,
              dpath_model=None, dpath_morph=None,
              dpath_pattern=None, dpath_states=None,
              init_model=True, iter_begin=0, iter_end=None,
              get_trj=False, trj_waypoints=None, verbose=0):

        _, jnp, _ = _ensure_jax_imports()
        import jax

        # ---- resolve params (mirror Solver.solve) ----
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

        # ---- resolve n_iters / iter_end ----
        if iter_end is None:
            if n_iters is None:
                raise ValueError("n_iters should be defined.")
            if n_iters < 1:
                raise ValueError("n_iters should be greater than or equal to 1.")
            iter_end = iter_begin + n_iters
        else:
            if iter_end <= iter_begin:
                raise ValueError(
                    f"iter_end ({iter_end}) must be greater than "
                    f"iter_begin ({iter_begin}).")
            if n_iters is None:
                n_iters = iter_end - iter_begin

        # ---- validate model and pick RHS ----
        if not step_supported(self._name):
            raise ValueError(f"JAX solver: unsupported solver '{self._name}'.")

        is_diploid = is_diploid_model(model)
        if is_diploid:
            if not is_diploid_supported(model):
                pa = model.paternal_model.name
                ma = model.maternal_model.name
                raise ValueError(
                    f"JAX solver: unsupported diploid sub-models "
                    f"(paternal={pa}, maternal={ma}).")
            pa_name = model.paternal_model.name
            ma_name = model.maternal_model.name
            rhs_fn = build_diploid_rhs(pa_name, ma_name)
            rhs_cache_key = ("diploid", pa_name, ma_name)
        else:
            if not rhs_supported(model.name):
                raise ValueError(
                    f"JAX solver: unsupported model '{model.name}'. "
                    f"Use device='cpu' for models without JAX kernel support.")
            rhs_fn = get_rhs(model.name)
            rhs_cache_key = model.name

        batch_size = model.batch_size

        # ---- compute recording mask ----
        if trj_waypoints is not None:
            valid_range = range(iter_begin + 1, iter_end + 1)
            waypoints_set = set(w for w in trj_waypoints if w in valid_range)
            get_trj = True
        else:
            waypoints_set = None

        if get_trj and period_output is None and waypoints_set is None:
            period_output = 1

        has_file_io = any([dpath_morph, dpath_pattern, dpath_states])

        # Build per-step record mask (True for steps whose state should be
        # captured).  Index i in the mask corresponds to iteration i+1
        # (1-based) of the original iter_begin..iter_end range.
        rec_mask_np = np.zeros(n_iters, dtype=bool)
        output_iters_sorted = []
        for local_i in range(n_iters):
            absolute_i = iter_begin + local_i  # iteration index BEFORE step
            abs_done = absolute_i + 1          # iteration index AFTER step
            should_record = False
            if waypoints_set is not None:
                if abs_done in waypoints_set:
                    should_record = True
            elif period_output is not None:
                if local_i == 0 or abs_done % period_output == 0:
                    # Match CPU: "(i == iter_begin or (i + 1) % period_output == 0)"
                    should_record = True
            if should_record:
                rec_mask_np[local_i] = True
                output_iters_sorted.append(abs_done)

        # ---- file I/O directory setup ----
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
                                 fpath=pjoin(dpath_models,
                                             fstr_fname_model % (j + 1)),
                                 initializer=model.initializer,
                                 solver=dict_solver)

        if dpath_morph:
            for j in range(batch_size):
                os.makedirs(pjoin(dpath_morph, dname_model % (j + 1)),
                            exist_ok=True)
            fstr_morph = "morph_%0{}d.png".format(
                int(np.floor(np.log10(n_iters))) + 1)

        if dpath_pattern:
            for j in range(batch_size):
                os.makedirs(pjoin(dpath_pattern, dname_model % (j + 1)),
                            exist_ok=True)
            fstr_pattern = "pattern_%0{}d.png".format(
                int(np.floor(np.log10(n_iters))) + 1)

        if dpath_states:
            for j in range(batch_size):
                os.makedirs(pjoin(dpath_states, dname_model % (j + 1)),
                            exist_ok=True)
            fstr_states = "states_%0{}d".format(
                int(np.floor(np.log10(n_iters))) + 1)

        # ---- prepare initial state on the target JAX device ----
        target_device = model.am._device
        model_dtype = np.dtype(model.dtype)
        dx_inv2 = np.asarray(1.0 / (model.dx ** 2), dtype=model_dtype)
        dt_jax = np.asarray(dt, dtype=model_dtype)

        y0 = model.y_mesh  # Already a JAX array on the target device

        if is_diploid:
            # Pass (pa_params, ma_params) as a pytree so jit can handle it
            params = (model.paternal_model.params,
                      model.maternal_model.params)
        else:
            params = model.params

        # solver-local state (e.g. AB2 prev_dydt)
        _, state_init_fn = get_step(self._name)
        state0 = state_init_fn(y0) if self._name == "AdamsBashforth2Solver" \
                                  else None

        # ---- compile (cached) ----
        has_rtol = rtol is not None
        want_trj = bool(rec_mask_np.any())
        compiled = self._get_compiled(
            rhs_fn, rhs_cache_key, n_iters, has_rtol, want_trj, model_dtype)

        t_total = time.time()

        if want_trj:
            rec_mask_jax = jax.device_put(rec_mask_np, target_device)
            if has_rtol:
                y_final, trj_all = compiled(
                    y0, params, state0, dt_jax, dx_inv2,
                    rec_mask_jax, np.asarray(rtol, dtype=model_dtype))
            else:
                y_final, trj_all = compiled(
                    y0, params, state0, dt_jax, dx_inv2, rec_mask_jax)
            y_final.block_until_ready()

            # Select the frames we actually want
            idxs = np.nonzero(rec_mask_np)[0]
            trj_y = trj_all[idxs]
            trj_y.block_until_ready()
        else:
            if has_rtol:
                y_final = compiled(
                    y0, params, state0, dt_jax, dx_inv2,
                    np.asarray(rtol, dtype=model_dtype))
            else:
                y_final = compiled(y0, params, state0, dt_jax, dx_inv2)
            y_final.block_until_ready()
            trj_y = None

        # ---- commit y_final back to the model ----
        model._y_mesh = y_final
        if is_diploid:
            # Phenotypic u/v = alpha * pa + beta * ma
            # Matches TwoComponentDiploidModel.pdefunc's formula.
            alpha = model.alpha
            beta = model.beta
            model._u = alpha * y_final[0] + beta * y_final[2]
            model._v = alpha * y_final[1] + beta * y_final[3]
        else:
            model._u = y_final[0, :, :, :]
            model._v = y_final[1, :, :, :]

        # ---- file I/O (uses host numpy) ----
        if has_file_io and trj_y is not None and len(output_iters_sorted) > 0:
            self._write_files(model, trj_y, output_iters_sorted, batch_size,
                              dpath_morph, dpath_pattern, dpath_states,
                              dname_model, fstr_morph, fstr_pattern,
                              fstr_states)

        if verbose >= 1:
            print("- [Duration] : %.5e sec." % (time.time() - t_total))

        # ---- return trajectory if requested ----
        if get_trj:
            if trj_y is None:
                # return empty array with correct trailing shape
                empty = jax.numpy.zeros((0, *model.shape_grid),
                                        dtype=model.y_mesh.dtype)
                trj_y = empty
            if trj_waypoints is not None:
                captured_wp = sorted(
                    set(trj_waypoints) & set(output_iters_sorted))
                return {"iters": captured_wp, "trj": trj_y}
            return trj_y

        return None

    # ------------------------------------------------------------------
    # File I/O (runs on host after scan completes)
    # ------------------------------------------------------------------

    @staticmethod
    def _write_files(model, trj_y, output_iters, batch_size,
                     dpath_morph, dpath_pattern, dpath_states,
                     dname_model, fstr_morph, fstr_pattern, fstr_states):
        trj_np = np.asarray(trj_y)
        for snap_idx, iter_num in enumerate(output_iters):
            snapshot = trj_np[snap_idx]
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
                    model.save_states(j, fpath_s, u=u_snap, v=v_snap)
