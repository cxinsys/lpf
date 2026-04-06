"""Per-solver single-step functions for the JAX fast path.

Each step_fn has the signature::

    step_fn(rhs, y, params, dt, dx_inv2, state) -> (y_new, new_state)

``state`` is carried between steps and holds solver-local memory
(e.g. previous dydt for Adams-Bashforth).  For stateless solvers it is
just ``None``.

All step functions mirror the Python solver implementations in
``lpf/solvers/*.py`` exactly — same operation order, same constants —
so that results match across backends within ULP of the model dtype.
Scalars and state flags follow ``y.dtype`` so float32/float64 models
both work without precision loss.
"""

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Euler
# ---------------------------------------------------------------------------

def euler_step(rhs, y, params, dt, dx_inv2, state):
    # Matches solver.py: model.y_mesh + delta_y,  delta_y = dt * pdefunc(y)
    delta = dt * rhs(y, params, dx_inv2)
    return y + delta, state


# ---------------------------------------------------------------------------
# Heun (improved Euler)
# Matches heunsolver.py:
#     k1 = dt * pdefunc(y)
#     k2 = dt * pdefunc(y + k1)
#     delta = 0.5 * (k1 + k2)
# ---------------------------------------------------------------------------

def heun_step(rhs, y, params, dt, dx_inv2, state):
    k1 = dt * rhs(y, params, dx_inv2)
    k2 = dt * rhs(y + k1, params, dx_inv2)
    delta = 0.5 * (k1 + k2)
    return y + delta, state


# ---------------------------------------------------------------------------
# Runge-Kutta 4
# Matches rungekuttasolver.py:
#     k1 = dt * pdefunc(y)
#     k2 = dt * pdefunc(y + 0.5*k1)
#     k3 = dt * pdefunc(y + 0.5*k2)
#     k4 = dt * pdefunc(y + k3)
#     delta = (k1 + 2*k2 + 2*k3 + k4) / 6
# ---------------------------------------------------------------------------

def rk4_step(rhs, y, params, dt, dx_inv2, state):
    k1 = dt * rhs(y, params, dx_inv2)
    k2 = dt * rhs(y + 0.5 * k1, params, dx_inv2)
    k3 = dt * rhs(y + 0.5 * k2, params, dx_inv2)
    k4 = dt * rhs(y + k3, params, dx_inv2)
    delta = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y + delta, state


# ---------------------------------------------------------------------------
# Bogacki-Shampine RK23 (fixed step, third-order)
# Matches rk23solver.py:
#     k1 = pdefunc(y)
#     k2 = pdefunc(y + 0.5*dt*k1)
#     k3 = pdefunc(y + 0.75*dt*k2)
#     delta = dt * ((2/9)*k1 + (1/3)*k2 + (4/9)*k3)
# ---------------------------------------------------------------------------

def rk23_step(rhs, y, params, dt, dx_inv2, state):
    k1 = rhs(y, params, dx_inv2)
    k2 = rhs(y + 0.5 * dt * k1, params, dx_inv2)
    k3 = rhs(y + 0.75 * dt * k2, params, dx_inv2)
    delta = dt * ((2.0 / 9.0) * k1 + (1.0 / 3.0) * k2 + (4.0 / 9.0) * k3)
    return y + delta, state


# ---------------------------------------------------------------------------
# Adams-Bashforth 2-step
# Matches adamsbashforth2solver.py:
#     dydt = pdefunc(y)
#     if prev is None: delta = dt * dydt
#     else:            delta = dt * (1.5*dydt - 0.5*prev)
#     prev = dydt
#
# state = (prev_dydt, has_prev_flag)
#   has_prev_flag: 0.0 on first step, 1.0 afterwards (used as a mix factor
#   so the JIT'd body has no data-dependent control flow).
# ---------------------------------------------------------------------------

def ab2_step(rhs, y, params, dt, dx_inv2, state):
    prev, has_prev = state
    dydt = rhs(y, params, dx_inv2)
    # When has_prev == 0 -> delta = dt * dydt
    # When has_prev == 1 -> delta = dt * (1.5*dydt - 0.5*prev)
    delta_first = dt * dydt
    delta_rest = dt * (1.5 * dydt - 0.5 * prev)
    delta = has_prev * delta_rest + (1.0 - has_prev) * delta_first
    new_state = (dydt, jnp.asarray(1.0, dtype=y.dtype))
    return y + delta, new_state


# ---------------------------------------------------------------------------
# Initial state helpers
# ---------------------------------------------------------------------------

def zero_state():
    return None


def ab2_initial_state(y_like):
    return (jnp.zeros_like(y_like), jnp.asarray(0.0, dtype=y_like.dtype))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STEP_FNS = {
    "EulerSolver": (euler_step, zero_state),
    "HeunSolver": (heun_step, zero_state),
    "RungeKuttaSolver": (rk4_step, zero_state),
    "RK23Solver": (rk23_step, zero_state),
    "AdamsBashforth2Solver": (ab2_step, ab2_initial_state),
}


def get_step(solver_name):
    if solver_name not in STEP_FNS:
        raise ValueError(
            f"JAX solver: unsupported solver '{solver_name}'. "
            f"Supported: {sorted(STEP_FNS.keys())}")
    return STEP_FNS[solver_name]


def is_supported(solver_name):
    return solver_name in STEP_FNS
