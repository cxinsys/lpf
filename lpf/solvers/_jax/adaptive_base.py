"""Base class for JAX-accelerated adaptive solvers (RKF45, DOPRI5).

These solvers cannot use the simple ``step_fn`` registry in
``lpf.solvers._jax.steps`` because their step has variable-length
internal sub-stepping with adaptive step-size control.  The full
adaptive integrator is built directly here using ``lax.while_loop``.

Like ``JaxSolverBase`` it provides the full ``Solver.solve()`` API
(trajectory, period_output, waypoints, file I/O) and dispatches once
into a jit-compiled XLA program for the entire ``n_iters``-step run.
"""

import os
import time
from os.path import join as pjoin

import numpy as np

from lpf.solvers.solver import Solver
from lpf.solvers._jax.rhs import (get_rhs, is_supported as rhs_supported,
                                  build_diploid_rhs, is_diploid_model,
                                  is_diploid_supported)
from lpf.solvers._jax.adaptive_steps import (make_rkf45_outer_step,
                                              make_dopri5_outer_step)


_ADAPT_COMPILE_CACHE = {}


def _ensure_jax_imports():
    import jax
    import jax.numpy as jnp
    from jax import lax
    return jax, jnp, lax


# ---------------------------------------------------------------------------
# Integrator builders for adaptive solvers
# ---------------------------------------------------------------------------

def _build_rkf45_no_trj(rhs_fn, dtype, n_iters, hyperparams):
    _, jnp, lax = _ensure_jax_imports()
    outer_step = make_rkf45_outer_step(rhs_fn, dtype, **hyperparams)
    sentinel = jnp.asarray(-1.0, dtype=dtype)

    def integrate(y0, params, dt, dx_inv2):
        def body(i, carry):
            y, dt_cur = carry
            y_new, dt_cur_new = outer_step(y, params, dt, dx_inv2, dt_cur)
            return (y_new, dt_cur_new)

        y_final, _ = lax.fori_loop(0, n_iters, body, (y0, sentinel))
        return y_final

    return integrate


def _build_rkf45_with_trj(rhs_fn, dtype, hyperparams):
    _, jnp, lax = _ensure_jax_imports()
    outer_step = make_rkf45_outer_step(rhs_fn, dtype, **hyperparams)
    sentinel = jnp.asarray(-1.0, dtype=dtype)

    def integrate(y0, params, dt, dx_inv2, rec_mask):
        def body(carry, _flag):
            y, dt_cur = carry
            y_new, dt_cur_new = outer_step(y, params, dt, dx_inv2, dt_cur)
            return (y_new, dt_cur_new), y_new

        (y_final, _), trj = lax.scan(body, (y0, sentinel), rec_mask)
        return y_final, trj

    return integrate


def _build_dopri5_no_trj(rhs_fn, dtype, n_iters, hyperparams):
    _, jnp, lax = _ensure_jax_imports()
    outer_step = make_dopri5_outer_step(rhs_fn, dtype, **hyperparams)
    sentinel = jnp.asarray(-1.0, dtype=dtype)
    flag_zero = jnp.asarray(0.0, dtype=dtype)

    def integrate(y0, params, dt, dx_inv2):
        zeros_like_y = jnp.zeros_like(y0)
        init = (y0, sentinel, zeros_like_y, flag_zero)

        def body(i, carry):
            y, dt_cur, k1_f, hf = carry
            y_new, dt_cur_new, k1_f_new, hf_new = outer_step(
                y, params, dt, dx_inv2, dt_cur, k1_f, hf)
            return (y_new, dt_cur_new, k1_f_new, hf_new)

        y_final, _, _, _ = lax.fori_loop(0, n_iters, body, init)
        return y_final

    return integrate


def _build_dopri5_with_trj(rhs_fn, dtype, hyperparams):
    _, jnp, lax = _ensure_jax_imports()
    outer_step = make_dopri5_outer_step(rhs_fn, dtype, **hyperparams)
    sentinel = jnp.asarray(-1.0, dtype=dtype)
    flag_zero = jnp.asarray(0.0, dtype=dtype)

    def integrate(y0, params, dt, dx_inv2, rec_mask):
        zeros_like_y = jnp.zeros_like(y0)
        init = (y0, sentinel, zeros_like_y, flag_zero)

        def body(carry, _flag):
            y, dt_cur, k1_f, hf = carry
            y_new, dt_cur_new, k1_f_new, hf_new = outer_step(
                y, params, dt, dx_inv2, dt_cur, k1_f, hf)
            return (y_new, dt_cur_new, k1_f_new, hf_new), y_new

        (y_final, _, _, _), trj = lax.scan(body, init, rec_mask)
        return y_final, trj

    return integrate


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class JaxAdaptiveBase(Solver):
    """Base class for adaptive JAX solvers.

    Subclasses set ``self._name`` and ``self._adaptive_kind`` ('rkf45' or
    'dopri5') and pass any tuning parameters via ``self._hyperparams``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Subclass fills these
        self._name = None
        self._adaptive_kind = None
        self._hyperparams = {}

    def _get_compiled(self, rhs_fn, rhs_cache_key, n_iters, want_trj, dtype):
        import jax
        # Hyperparams contribute to the cache key (different tolerance →
        # different compiled body).  dtype too — float32/float64 produce
        # different XLA programs.
        hp_key = tuple(sorted(self._hyperparams.items()))
        key = (self._adaptive_kind, rhs_cache_key, n_iters, want_trj,
               hp_key, np.dtype(dtype).str)
        fn = _ADAPT_COMPILE_CACHE.get(key)
        if fn is not None:
            return fn

        if self._adaptive_kind == "rkf45":
            if want_trj:
                builder = _build_rkf45_with_trj(rhs_fn, dtype, self._hyperparams)
            else:
                builder = _build_rkf45_no_trj(
                    rhs_fn, dtype, n_iters, self._hyperparams)
        elif self._adaptive_kind == "dopri5":
            if want_trj:
                builder = _build_dopri5_with_trj(
                    rhs_fn, dtype, self._hyperparams)
            else:
                builder = _build_dopri5_no_trj(
                    rhs_fn, dtype, n_iters, self._hyperparams)
        else:
            raise ValueError(f"Unknown adaptive kind: {self._adaptive_kind}")

        fn = jax.jit(builder)
        _ADAPT_COMPILE_CACHE[key] = fn
        return fn

    # ------------------------------------------------------------------
    # solve() — same surface as JaxSolverBase but with adaptive carry
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

        if rtol is not None:
            # Adaptive solvers manage their own error control internally.
            # The CPU code also doesn't honour rtol for adaptive solvers
            # in any meaningful way (it would clash with the adaptive
            # tolerance).  We silently ignore.
            pass

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
        is_diploid = is_diploid_model(model)
        if is_diploid:
            if not is_diploid_supported(model):
                pa = model.paternal_model.name
                ma = model.maternal_model.name
                raise ValueError(
                    f"JAX adaptive solver: unsupported diploid sub-models "
                    f"(paternal={pa}, maternal={ma}).")
            pa_name = model.paternal_model.name
            ma_name = model.maternal_model.name
            rhs_fn = build_diploid_rhs(pa_name, ma_name)
            rhs_cache_key = ("diploid", pa_name, ma_name)
        else:
            if not rhs_supported(model.name):
                raise ValueError(
                    f"JAX adaptive solver: unsupported model '{model.name}'. "
                    f"Use device='cpu' for unsupported models.")
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

        rec_mask_np = np.zeros(n_iters, dtype=bool)
        output_iters_sorted = []
        for local_i in range(n_iters):
            absolute_i = iter_begin + local_i
            abs_done = absolute_i + 1
            should_record = False
            if waypoints_set is not None:
                if abs_done in waypoints_set:
                    should_record = True
            elif period_output is not None:
                if local_i == 0 or abs_done % period_output == 0:
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

        # ---- prepare initial state ----
        target_device = model.am._device
        model_dtype = np.dtype(model.dtype)
        dx_inv2 = np.asarray(1.0 / (model.dx ** 2), dtype=model_dtype)
        dt_jax = np.asarray(dt, dtype=model_dtype)

        y0 = model.y_mesh
        if is_diploid:
            params = (model.paternal_model.params,
                      model.maternal_model.params)
        else:
            params = model.params

        # ---- compile (cached) ----
        want_trj = bool(rec_mask_np.any())
        compiled = self._get_compiled(
            rhs_fn, rhs_cache_key, n_iters, want_trj, model_dtype)

        t_total = time.time()

        if want_trj:
            rec_mask_jax = jax.device_put(rec_mask_np, target_device)
            y_final, trj_all = compiled(
                y0, params, dt_jax, dx_inv2, rec_mask_jax)
            y_final.block_until_ready()
            idxs = np.nonzero(rec_mask_np)[0]
            trj_y = trj_all[idxs]
            trj_y.block_until_ready()
        else:
            y_final = compiled(y0, params, dt_jax, dx_inv2)
            y_final.block_until_ready()
            trj_y = None

        # ---- commit y_final back to the model ----
        model._y_mesh = y_final
        if is_diploid:
            alpha = model.alpha
            beta = model.beta
            model._u = alpha * y_final[0] + beta * y_final[2]
            model._v = alpha * y_final[1] + beta * y_final[3]
        else:
            model._u = y_final[0, :, :, :]
            model._v = y_final[1, :, :, :]

        # ---- file I/O ----
        if has_file_io and trj_y is not None and len(output_iters_sorted) > 0:
            from lpf.solvers._jax.base import JaxSolverBase
            JaxSolverBase._write_files(
                model, trj_y, output_iters_sorted, batch_size,
                dpath_morph, dpath_pattern, dpath_states,
                dname_model, fstr_morph, fstr_pattern, fstr_states)

        if verbose >= 1:
            print("- [Duration] : %.5e sec." % (time.time() - t_total))

        # ---- return trajectory if requested ----
        if get_trj:
            if trj_y is None:
                trj_y = jax.numpy.zeros((0, *model.shape_grid),
                                         dtype=model.y_mesh.dtype)
            if trj_waypoints is not None:
                captured_wp = sorted(
                    set(trj_waypoints) & set(output_iters_sorted))
                return {"iters": captured_wp, "trj": trj_y}
            return trj_y

        return None
