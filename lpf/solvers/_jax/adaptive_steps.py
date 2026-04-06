"""JAX adaptive step functions for RKF45 and DOPRI5.

Mirrors the CPU implementations in
``lpf/solvers/adaptiverk45solver.py`` and ``lpf/solvers/dopri5solver.py``.
All scalar constants and tableau coefficients are materialized in the
caller-supplied ``dtype`` (matching ``model.dtype``), so float32/float64
models both run at native precision.

Each ``make_*_outer_step`` returns a function::

    outer_step(y, params, dt, dx_inv2, *adapt_state_in) -> (y_new, *adapt_state_out)

that advances ``y`` by exactly ``dt`` using internal variable-size
sub-step loops (``lax.while_loop``).  The ``adapt_state`` carries
solver-local scalars (e.g. ``dt_current``, DOPRI5 FSAL ``k1``) between
outer iterations so the adaptive step size history is preserved across
the full simulation — matching CPU semantics.
"""

import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# RKF45: Runge-Kutta-Fehlberg 4(5)
# Matches lpf/solvers/adaptiverk45solver.py exactly.
# ---------------------------------------------------------------------------

# Butcher tableau (from the CPU reference)
_RKF45_B4 = (25/216, 0.0, 1408/2565, 2197/4104, -1/5, 0.0)
_RKF45_B5 = (16/135, 0.0, 6656/12825, 28561/56430, -9/50, 2/55)
_RKF45_A = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1/4, 0.0, 0.0, 0.0, 0.0, 0.0),
    (3/32, 9/32, 0.0, 0.0, 0.0, 0.0),
    (1932/2197, -7200/2197, 7296/2197, 0.0, 0.0, 0.0),
    (439/216, -8.0, 3680/513, -845/4104, 0.0, 0.0),
    (-8/27, 2.0, -3544/2565, 1859/4104, -11/40, 0.0),
)


def make_rkf45_outer_step(rhs_fn, dtype, tolerance=1e-6, safety=0.9,
                           min_factor=0.1, max_factor=5.0, max_attempts=10):
    """Build an adaptive-RKF45 outer step for JAX.

    Signature: ``outer_step(y, params, dt, dx_inv2, dt_current) ->
    (y_new, dt_current_new)``.

    All constants are materialized in ``dtype`` (the model's working
    precision) so float64 models retain full precision.

    Matches CPU semantics:
      - Sub-steps until ``t_local >= dt``.
      - Each sub-step tries up to ``max_attempts`` times; on rejection
        the step size is shrunk using the same formula as CPU.
      - ``dt_current`` carries the last accepted step size across outer
        iterations.
    """
    _S = lambda v: jnp.asarray(v, dtype=dtype)  # noqa: E731
    tol = _S(tolerance)
    sf = _S(safety)
    mnf = _S(min_factor)
    mxf = _S(max_factor)
    b4 = tuple(_S(x) for x in _RKF45_B4)
    b5 = tuple(_S(x) for x in _RKF45_B5)
    a = tuple(tuple(_S(x) for x in row) for row in _RKF45_A)
    _TINY = _S(1e-30)
    _ACCEPT_FLOOR = _S(1e-10)
    _ONE_FIFTH = _S(0.2)

    def compute_stages(y_c, params, h, dx_inv2):
        k0 = rhs_fn(y_c, params, dx_inv2)
        y1 = y_c + h * a[1][0] * k0
        k1 = rhs_fn(y1, params, dx_inv2)
        y2 = y_c + h * (a[2][0] * k0 + a[2][1] * k1)
        k2 = rhs_fn(y2, params, dx_inv2)
        y3 = y_c + h * (a[3][0] * k0 + a[3][1] * k1 + a[3][2] * k2)
        k3 = rhs_fn(y3, params, dx_inv2)
        y4s = y_c + h * (a[4][0] * k0 + a[4][1] * k1
                          + a[4][2] * k2 + a[4][3] * k3)
        k4 = rhs_fn(y4s, params, dx_inv2)
        y5s = y_c + h * (a[5][0] * k0 + a[5][1] * k1 + a[5][2] * k2
                          + a[5][3] * k3 + a[5][4] * k4)
        k5 = rhs_fn(y5s, params, dx_inv2)
        return k0, k1, k2, k3, k4, k5

    def attempt_one(y_c, params, h, dx_inv2, dt):
        """Single RKF45 trial at step size h.

        Returns (y4, accept, dt_opt).
            y4     : candidate new state (4th-order solution)
            accept : True if step should be accepted
            dt_opt : clipped optimal dt for next step / next retry
        """
        k0, k1, k2, k3, k4, k5 = compute_stages(y_c, params, h, dx_inv2)

        y4 = y_c + h * (b4[0] * k0 + b4[2] * k2 + b4[3] * k3 + b4[4] * k4)
        y5 = y_c + h * (b5[0] * k0 + b5[2] * k2 + b5[3] * k3
                        + b5[4] * k4 + b5[5] * k5)

        max_error = jnp.max(jnp.abs(y5 - y4))

        safe_err = jnp.where(max_error > 0, max_error, _TINY)
        dt_opt_raw = h * sf * (tol / safe_err) ** _ONE_FIFTH
        dt_opt = jnp.where(
            max_error > 0,
            jnp.maximum(h * mnf, jnp.minimum(h * mxf, dt_opt_raw)),
            h * mxf)

        # CPU: accept if max_error <= tolerance OR h <= dt * 1e-10
        accept = jnp.logical_or(max_error <= tol,
                                 h <= dt * _ACCEPT_FLOOR)
        return y4, accept, dt_opt

    def outer_step(y, params, dt, dx_inv2, dt_current):
        dt = dt.astype(dtype)
        # First call: CPU does `if self._dt_current is None: self._dt_current = dt`
        dt_current = jnp.where(dt_current <= 0, dt, dt_current)

        def sub_cond(carry):
            y_c, t_local, dt_cur = carry
            return t_local < dt

        def sub_body(carry):
            y_c, t_local, dt_cur = carry
            h0 = jnp.minimum(dt_cur, dt - t_local)

            # Inner retry loop (up to max_attempts)
            def att_cond(ac):
                y_a, h, attempt, accepted, dt_opt = ac
                return jnp.logical_and(attempt < max_attempts,
                                       jnp.logical_not(accepted))

            def att_body(ac):
                y_a, h, attempt, _, _ = ac
                y4, accept, dt_opt = attempt_one(y_a, params, h, dx_inv2, dt)
                # On accept: commit y4 and freeze (accepted=True)
                # On reject: stay at y_a, shrink h via min(dt_opt, dt - t_local)
                new_y = jnp.where(accept, y4, y_a)
                # CPU reject branch: dt_current = dt_opt; h = min(dt_opt, dt - t_local)
                new_h = jnp.where(accept, h,
                                   jnp.minimum(dt_opt, dt - t_local))
                return (new_y, new_h, attempt + 1, accept, dt_opt)

            init_att = (y_c, h0, jnp.int32(0), jnp.bool_(False),
                        _S(0.0))
            y_after, h_used, _, accepted, dt_opt_final = lax.while_loop(
                att_cond, att_body, init_att)

            # If max_attempts exhausted without accept, CPU still commits
            # last y4; our inner loop already produced y_after in that case.
            # Advance t_local by h_used either way (CPU behaviour).
            new_t = t_local + h_used
            # dt_current update: on success CPU sets dt_current = dt_opt
            # from the accepted attempt, which is `dt_opt_final`.
            new_dt_cur = dt_opt_final
            return (y_after, new_t, new_dt_cur)

        init_sub = (y, _S(0.0), dt_current)
        y_final, _, dt_cur_final = lax.while_loop(
            sub_cond, sub_body, init_sub)
        return y_final, dt_cur_final

    return outer_step


# ---------------------------------------------------------------------------
# DOPRI5: Dormand-Prince 5th-order with FSAL
# Matches lpf/solvers/dopri5solver.py exactly.
# ---------------------------------------------------------------------------

_DP = {
    "a21": 1/5,
    "a31": 3/40,       "a32": 9/40,
    "a41": 44/45,      "a42": -56/15,      "a43": 32/9,
    "a51": 19372/6561, "a52": -25360/2187, "a53": 64448/6561, "a54": -212/729,
    "a61": 9017/3168,  "a62": -355/33,     "a63": 46732/5247, "a64": 49/176,    "a65": -5103/18656,
    "b1": 35/384, "b3": 500/1113, "b4": 125/192, "b5": -2187/6784, "b6": 11/84,
    "e1": 71/57600, "e3": -71/16695, "e4": 71/1920, "e5": -17253/339200,
    "e6": 22/525, "e7": -1/40,
}


def make_dopri5_outer_step(rhs_fn, dtype, tolerance=1e-6, safety=0.9,
                            min_factor=0.2, max_factor=5.0, max_attempts=8):
    """Build an adaptive DOPRI5 outer step for JAX.

    Signature: ``outer_step(y, params, dt, dx_inv2, dt_current, k1_fsal, has_fsal)
                -> (y_new, dt_current_new, k1_fsal_new, has_fsal_new)``

    All constants are materialized in ``dtype`` (the model's working
    precision) so float64 models retain full precision.

    FSAL (First Same As Last): carries the k7 from a previously accepted
    sub-step as k1 of the next.  ``has_fsal`` is a 0/1 flag (in ``dtype``)
    so the JIT body has no data-dependent control flow.
    """
    _S = lambda v: jnp.asarray(v, dtype=dtype)  # noqa: E731
    tol = _S(tolerance)
    sf = _S(safety)
    mnf = _S(min_factor)
    mxf = _S(max_factor)

    c = {k: _S(v) for k, v in _DP.items()}
    _TINY = _S(1e-30)
    _ACCEPT_FLOOR = _S(1e-12)
    _QUARTER = _S(0.25)
    _ONE_FIFTH = _S(0.2)

    def dopri_attempt(y_c, params, h, dx_inv2, k1):
        """Single DOPRI5 trial.  Returns (y_new, k7, error_norm)."""
        k2 = rhs_fn(y_c + h * c["a21"] * k1, params, dx_inv2)
        k3 = rhs_fn(y_c + h * (c["a31"] * k1 + c["a32"] * k2),
                     params, dx_inv2)
        k4 = rhs_fn(y_c + h * (c["a41"] * k1 + c["a42"] * k2
                                + c["a43"] * k3), params, dx_inv2)
        k5 = rhs_fn(y_c + h * (c["a51"] * k1 + c["a52"] * k2
                                + c["a53"] * k3 + c["a54"] * k4),
                     params, dx_inv2)
        k6 = rhs_fn(y_c + h * (c["a61"] * k1 + c["a62"] * k2
                                + c["a63"] * k3 + c["a64"] * k4
                                + c["a65"] * k5), params, dx_inv2)
        y_new = y_c + h * (c["b1"] * k1 + c["b3"] * k3 + c["b4"] * k4
                            + c["b5"] * k5 + c["b6"] * k6)
        k7 = rhs_fn(y_new, params, dx_inv2)

        error = h * (c["e1"] * k1 + c["e3"] * k3 + c["e4"] * k4
                     + c["e5"] * k5 + c["e6"] * k6 + c["e7"] * k7)
        error_norm = jnp.sqrt(jnp.mean(error ** 2))
        return y_new, k7, error_norm

    def outer_step(y, params, dt, dx_inv2, dt_current, k1_fsal, has_fsal):
        dt = dt.astype(dtype)
        dt_current = jnp.where(dt_current <= 0, dt, dt_current)

        def sub_cond(carry):
            y_c, t_local, dt_cur, k1_f, hf = carry
            return t_local < dt

        def sub_body(carry):
            y_c, t_local, dt_cur, k1_f, hf = carry
            h0 = jnp.minimum(dt_cur, dt - t_local)

            # FSAL: reuse carried k1 if available, else compute fresh
            k1_init = jnp.where(hf > 0, k1_f, rhs_fn(y_c, params, dx_inv2))

            # Retry loop
            def att_cond(ac):
                y_a, h, attempt, accepted, y_new_a, k7_a = ac
                return jnp.logical_and(attempt < max_attempts,
                                       jnp.logical_not(accepted))

            def att_body(ac):
                y_a, h, attempt, _, _, _ = ac
                y_new, k7, err_norm = dopri_attempt(y_a, params, h, dx_inv2,
                                                     k1_init)
                accept = jnp.logical_or(err_norm <= tol,
                                         h <= dt * _ACCEPT_FLOOR)
                # Rejection step-size shrink (CPU formula)
                safe_err = jnp.where(err_norm > 0, err_norm, _TINY)
                reject_factor = jnp.maximum(
                    mnf, sf * (tol / safe_err) ** _QUARTER)
                new_h = jnp.where(accept, h, h * reject_factor)
                new_h = jnp.minimum(new_h, dt - t_local)
                return (y_a, new_h, attempt + 1, accept, y_new, k7)

            init_att = (y_c, h0, jnp.int32(0), jnp.bool_(False),
                        jnp.zeros_like(y_c), jnp.zeros_like(y_c))
            _, h_used, _, accepted, y_new_final, k7_final = lax.while_loop(
                att_cond, att_body, init_att)

            # CPU: accept advances y and updates dt_current using
            #   factor = safety * (tol/err)**0.2, clipped to [min_factor, max_factor].
            # Re-evaluate err_norm on the accepted step to produce dt_cur_new
            # with matching arithmetic.  (Cheap: XLA CSE will share with
            # the last attempt's compute.)
            _, _, err_norm_used = dopri_attempt(y_c, params, h_used, dx_inv2,
                                                  k1_init)
            safe_err_u = jnp.where(err_norm_used > 0, err_norm_used, _TINY)
            factor = jnp.where(
                err_norm_used > 0,
                jnp.maximum(mnf, jnp.minimum(
                    mxf, sf * (tol / safe_err_u) ** _ONE_FIFTH)),
                mxf)
            new_dt_cur = h_used * factor

            # FSAL: only valid if accepted.  CPU invalidates k1_next after
            # rejection within the attempt loop; since our inner loop only
            # commits once (on the final iteration), we just use k7_final
            # when accepted.
            hf_new = jnp.where(accepted, _S(1.0), _S(0.0))
            k1_f_new = k7_final

            new_t = t_local + h_used
            return (y_new_final, new_t, new_dt_cur, k1_f_new, hf_new)

        init_sub = (y, _S(0.0), dt_current, k1_fsal, has_fsal)
        y_final, _, dt_cur_f, k1_f_f, hf_f = lax.while_loop(
            sub_cond, sub_body, init_sub)
        return y_final, dt_cur_f, k1_f_f, hf_f

    return outer_step
