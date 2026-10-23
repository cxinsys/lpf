"""JAX RHS implementations for all supported reaction-diffusion models.

Each function takes ``(y, params, dx_inv2)`` and returns ``dydt`` with
the same shape as ``y``, with zero boundaries (Neumann).

Operation order is kept identical to the reference Python path
(``TwoComponentModel.pdefunc`` + each model's ``reactions``) so that
float32 results match bit-for-bit (or within ULP) across NumPy / CuPy /
PyTorch / JAX when the same XLA fusion patterns are applied.
"""

import jax.numpy as jnp


def _laplacian2d(a, dx_inv2):
    """Five-point Laplacian stencil identical to TwoComponentModel.laplacian2d.

    Operation order is preserved: (top + left + bottom + right - 4*center) * dx_inv2.
    ``a`` has shape (B, H, W); returns (B, H-2, W-2).
    """
    a_top = a[:, 0:-2, 1:-1]
    a_left = a[:, 1:-1, 0:-2]
    a_bottom = a[:, 2:, 1:-1]
    a_right = a[:, 1:-1, 2:]
    a_center = a[:, 1:-1, 1:-1]
    return (a_top + a_left + a_bottom + a_right - 4.0 * a_center) * dx_inv2


def _reshape_param(p, batch_size):
    return p[:batch_size].reshape(batch_size, 1, 1)


def _apply_boundary(dydt, du, dv):
    """Place du/dv in the interior and zero the Neumann boundary."""
    dydt = dydt.at[0, :, 1:-1, 1:-1].set(du)
    dydt = dydt.at[1, :, 1:-1, 1:-1].set(dv)
    return dydt


# ---------------------------------------------------------------------------
# Per-model RHS
# ---------------------------------------------------------------------------

def liaw_rhs(y, params, dx_inv2):
    """Liaw model: f = ru*u²v/(1+k*u²) + su - mu*u ; g = -rv*u²v/(1+k*u²) + sv."""
    B = y.shape[1]
    u = y[0]
    v = y[1]

    Du = _reshape_param(params[:, 0], B)
    Dv = _reshape_param(params[:, 1], B)
    ru = _reshape_param(params[:, 2], B)
    rv = _reshape_param(params[:, 3], B)
    k  = _reshape_param(params[:, 4], B)
    su = _reshape_param(params[:, 5], B)
    sv = _reshape_param(params[:, 6], B)
    mu = _reshape_param(params[:, 7], B)

    u_c = u[:, 1:-1, 1:-1]
    v_c = v[:, 1:-1, 1:-1]

    # Match LiawModel.reactions exactly:
    #   f = ru * ((u_c ** 2 * v_c) / (1 + k * u_c ** 2)) + su - mu * u_c
    #   g = -rv * ((u_c ** 2 * v_c) / (1 + k * u_c ** 2)) + sv
    u_sq = u_c ** 2
    num = u_sq * v_c
    denom = 1.0 + k * u_sq
    frac = num / denom
    f = ru * frac + su - mu * u_c
    g = -rv * frac + sv

    lap_u = _laplacian2d(u, dx_inv2)
    lap_v = _laplacian2d(v, dx_inv2)

    du = Du * lap_u + f
    dv = Dv * lap_v + g

    dydt = jnp.zeros_like(y)
    return _apply_boundary(dydt, du, dv)


def grayscott_rhs(y, params, dx_inv2):
    B = y.shape[1]
    u = y[0]
    v = y[1]
    Du = _reshape_param(params[:, 0], B)
    Dv = _reshape_param(params[:, 1], B)
    F  = _reshape_param(params[:, 2], B)
    k  = _reshape_param(params[:, 3], B)

    u_c = u[:, 1:-1, 1:-1]
    v_c = v[:, 1:-1, 1:-1]

    u_vsq = u_c * v_c ** 2
    f = -u_vsq + F * (1 - u_c)
    g = u_vsq - (F + k) * v_c

    du = Du * _laplacian2d(u, dx_inv2) + f
    dv = Dv * _laplacian2d(v, dx_inv2) + g
    dydt = jnp.zeros_like(y)
    return _apply_boundary(dydt, du, dv)


def brusselator_rhs(y, params, dx_inv2):
    B = y.shape[1]
    u = y[0]
    v = y[1]
    Du = _reshape_param(params[:, 0], B)
    Dv = _reshape_param(params[:, 1], B)
    A  = _reshape_param(params[:, 2], B)
    Bp = _reshape_param(params[:, 3], B)

    u_c = u[:, 1:-1, 1:-1]
    v_c = v[:, 1:-1, 1:-1]

    usq_v = u_c ** 2 * v_c
    f = A - (Bp + 1) * u_c + usq_v
    g = Bp * u_c - usq_v

    du = Du * _laplacian2d(u, dx_inv2) + f
    dv = Dv * _laplacian2d(v, dx_inv2) + g
    dydt = jnp.zeros_like(y)
    return _apply_boundary(dydt, du, dv)


def fitzhughnagumo_rhs(y, params, dx_inv2):
    B = y.shape[1]
    u = y[0]
    v = y[1]
    Du    = _reshape_param(params[:, 0], B)
    Dv    = _reshape_param(params[:, 1], B)
    eps   = _reshape_param(params[:, 2], B)
    gamma = _reshape_param(params[:, 3], B)
    beta  = _reshape_param(params[:, 4], B)

    u_c = u[:, 1:-1, 1:-1]
    v_c = v[:, 1:-1, 1:-1]

    f = u_c - u_c ** 3 - v_c
    g = eps * (u_c - gamma * v_c + beta)

    du = Du * _laplacian2d(u, dx_inv2) + f
    dv = Dv * _laplacian2d(v, dx_inv2) + g
    dydt = jnp.zeros_like(y)
    return _apply_boundary(dydt, du, dv)


def schnakenberg_rhs(y, params, dx_inv2):
    B = y.shape[1]
    u = y[0]
    v = y[1]
    Du  = _reshape_param(params[:, 0], B)
    Dv  = _reshape_param(params[:, 1], B)
    rho = _reshape_param(params[:, 2], B)
    su  = _reshape_param(params[:, 3], B)
    sv  = _reshape_param(params[:, 4], B)
    mu  = _reshape_param(params[:, 5], B)

    u_c = u[:, 1:-1, 1:-1]
    v_c = v[:, 1:-1, 1:-1]

    usq_v = u_c ** 2 * v_c
    f = su - mu * u_c + rho * usq_v
    g = sv - rho * usq_v

    du = Du * _laplacian2d(u, dx_inv2) + f
    dv = Dv * _laplacian2d(v, dx_inv2) + g
    dydt = jnp.zeros_like(y)
    return _apply_boundary(dydt, du, dv)


def gierermeinhardt_rhs(y, params, dx_inv2):
    B = y.shape[1]
    u = y[0]
    v = y[1]
    Du = _reshape_param(params[:, 0], B)
    Dv = _reshape_param(params[:, 1], B)
    ru = _reshape_param(params[:, 2], B)
    rv = _reshape_param(params[:, 3], B)
    mu = _reshape_param(params[:, 4], B)
    nu = _reshape_param(params[:, 5], B)

    u_c = u[:, 1:-1, 1:-1]
    v_c = v[:, 1:-1, 1:-1]

    usq = u_c ** 2
    f = ru * usq / v_c - mu * u_c
    g = rv * usq - nu * v_c

    du = Du * _laplacian2d(u, dx_inv2) + f
    dv = Dv * _laplacian2d(v, dx_inv2) + g
    dydt = jnp.zeros_like(y)
    return _apply_boundary(dydt, du, dv)


# ---------------------------------------------------------------------------
# Diploid model RHS builders
#
# TwoComponentDiploidModel stores y_mesh as (4, B, H, W):
#     [pa_u, pa_v, ma_u, ma_v]
# and its pdefunc dispatches paternal/maternal halves to two child
# TwoComponentModel.pdefunc calls.  We mirror that here so the entire
# diploid step remains inside one XLA graph.
#
# For JAX, the ``params`` argument is a tuple (pa_params, ma_params) so
# the solver base can pass both into the jit-compiled integrator.
# ---------------------------------------------------------------------------

def build_diploid_rhs(pa_model_name, ma_model_name):
    """Return a RHS function for TwoComponentDiploidModel.

    The returned callable expects ``y`` of shape (4, B, H, W) and
    ``params`` as a tuple ``(pa_params, ma_params)``.  The resulting
    ``dydt`` has shape (4, B, H, W) with zero Neumann boundaries.
    """
    pa_rhs = _RHS_BY_MODEL[pa_model_name]
    ma_rhs = _RHS_BY_MODEL[ma_model_name]

    def diploid_rhs(y, params, dx_inv2):
        pa_params, ma_params = params
        dydt_pa = pa_rhs(y[:2], pa_params, dx_inv2)
        dydt_ma = ma_rhs(y[2:], ma_params, dx_inv2)
        return jnp.concatenate([dydt_pa, dydt_ma], axis=0)

    return diploid_rhs


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_RHS_BY_MODEL = {
    "LiawModel": liaw_rhs,
    "GrayScottModel": grayscott_rhs,
    "BrusselatorModel": brusselator_rhs,
    "FitzHughNagumoModel": fitzhughnagumo_rhs,
    "SchnakenbergModel": schnakenberg_rhs,
    "GiererMeinhardtModel": gierermeinhardt_rhs,
}


def get_rhs(model_name):
    """Return the JAX RHS function for a model name."""
    if model_name not in _RHS_BY_MODEL:
        raise ValueError(
            f"JAX solver: unsupported model '{model_name}'. "
            f"Supported: {sorted(_RHS_BY_MODEL.keys())}")
    return _RHS_BY_MODEL[model_name]


def is_supported(model_name):
    return model_name in _RHS_BY_MODEL


def is_diploid_model(model):
    """Check if a model is a TwoComponentDiploidModel instance."""
    try:
        from lpf.models import TwoComponentDiploidModel
        return isinstance(model, TwoComponentDiploidModel)
    except ImportError:
        return False


def is_diploid_supported(model):
    """Check if both paternal and maternal sub-models have JAX RHS support."""
    if not is_diploid_model(model):
        return False
    pa_name = model.paternal_model.name
    ma_name = model.maternal_model.name
    return pa_name in _RHS_BY_MODEL and ma_name in _RHS_BY_MODEL


# ---------------------------------------------------------------------------
# Geometry: domains on the grid (lpf.domains) and mesh surfaces (lpf.surfaces)
# ---------------------------------------------------------------------------

class _ParamsView:
    """Stands in for a model inside reactions(), which only reads .params."""

    def __init__(self, params):
        self.params = params


def _laplacian_domain(a, st, dx_inv2):
    """Domain-weighted Laplacian; same accumulation order as the CUDA kernel
    and DomainGeometry.laplacian. st: (B or 1, 5, H, W)."""
    lap = jnp.zeros_like(a)
    lap = lap.at[:, 1:, :].set(st[:, 0, 1:, :] * (a[:, :-1, :] - a[:, 1:, :]))
    lap = lap.at[:, :-1, :].set(lap[:, :-1, :] + st[:, 1, :-1, :] * (a[:, 1:, :] - a[:, :-1, :]))
    lap = lap.at[:, :, 1:].set(lap[:, :, 1:] + st[:, 2, :, 1:] * (a[:, :, :-1] - a[:, :, 1:]))
    lap = lap.at[:, :, :-1].set(lap[:, :, :-1] + st[:, 3, :, :-1] * (a[:, :, 1:] - a[:, :, :-1]))
    return lap * dx_inv2


def _laplacian_mesh(a, nbr, wts):
    """Laplace-Beltrami operator in ELL format (K, N); same order as the
    CUDA kernel and MeshGeometry.laplacian."""
    lap = wts[0] * (a[:, nbr[0]] - a)
    for k in range(1, nbr.shape[0]):
        lap = lap + wts[k] * (a[:, nbr[k]] - a)
    return lap


def _kinetics(model):
    """The model class's own reactions(), callable with traced params."""
    reactions = type(model).reactions

    def kinetics(params, u, v):
        return reactions(_ParamsView(params), None, u, v)
    return kinetics


def build_domain_rhs(model):
    """RHS of any two-component model on a domain (lpf.domains). Its params
    pytree is (params, stencils), see DomainGeometry.jax_rhs()."""
    kinetics = _kinetics(model)

    def rhs(y, params, dx_inv2):
        p, st = params
        B = y.shape[1]
        u, v = y[0], y[1]
        f, g = kinetics(p, u, v)
        active = st[:, 4]
        du = (_reshape_param(p[:, 0], B) * _laplacian_domain(u, st, dx_inv2) + f) * active
        dv = (_reshape_param(p[:, 1], B) * _laplacian_domain(v, st, dx_inv2) + g) * active
        return jnp.stack([du, dv])
    return rhs


def build_mesh_rhs(model):
    """RHS of any two-component model on a mesh (lpf.surfaces). Its params
    pytree is (params, nbr, wts), see MeshGeometry.jax_rhs()."""
    kinetics = _kinetics(model)

    def rhs(y, params, dx_inv2):
        p, nbr, wts = params
        B, N = y.shape[1], y.shape[2]
        u, v = y[0], y[1]
        f, g = kinetics(p, u.reshape(B, 1, N), v.reshape(B, 1, N))
        du = p[:, 0].reshape(B, 1) * _laplacian_mesh(u, nbr, wts) + f.reshape(B, N)
        dv = p[:, 1].reshape(B, 1) * _laplacian_mesh(v, nbr, wts) + g.reshape(B, N)
        return jnp.stack([du, dv])
    return rhs


def select_rhs(model):
    """(rhs, cache_key, params) of a model on its geometry (lpf.geometry)."""
    if is_diploid_model(model):
        return model.geometry.jax_diploid_rhs(model)
    return model.geometry.jax_rhs(model)
