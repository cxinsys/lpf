"""
CUDA kernel source templates for fused reaction-diffusion solvers.

Templates use placeholders:
  ${REACTION_CODE} - Model-specific reaction computation (from reactions.py)

Preprocessor macros (set at compile time):
  REAL        - float or double
  REAL_CONST  - suffix macro for literal constants (e.g., 1.0f vs 1.0)
  N_PARAMS    - number of kinetic parameters per batch element
"""


# =============================================================================
# Fused Euler Step Kernel
# Computes: y_out = y_in + dt * (D * laplacian(y_in) + reaction(y_in))
# Uses ping-pong buffers: reads y_in, writes y_out
# Each thread handles one (batch, row, col) point for BOTH u and v species.
# =============================================================================
FUSED_EULER_STEP_TEMPLATE = r"""
extern "C" __global__
void fused_euler_step(
    const REAL* __restrict__ y_in,
    REAL* __restrict__ y_out,
    const REAL* __restrict__ params,
    const int B,
    const int H,
    const int W,
    const REAL dt,
    const REAL dx2_inv
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * H * W;
    if (idx >= total) return;

    const int b  = idx / (H * W);
    const int hw = idx % (H * W);
    const int i  = hw / W;
    const int j  = hw % W;

    const int BHW    = B * H * W;
    const int center = b * (H * W) + i * W + j;

    const REAL u_c = y_in[center];
    const REAL v_c = y_in[BHW + center];

    // Boundary: dydt = 0 (Neumann), so y_out = y_in
    if (i == 0 || i == H - 1 || j == 0 || j == W - 1) {
        y_out[center]       = u_c;
        y_out[BHW + center] = v_c;
        return;
    }

    // 5-point stencil neighbors for u
    const REAL u_top = y_in[center - W];
    const REAL u_bot = y_in[center + W];
    const REAL u_lft = y_in[center - 1];
    const REAL u_rgt = y_in[center + 1];

    // 5-point stencil neighbors for v
    const REAL v_top = y_in[BHW + center - W];
    const REAL v_bot = y_in[BHW + center + W];
    const REAL v_lft = y_in[BHW + center - 1];
    const REAL v_rgt = y_in[BHW + center + 1];

    // Discrete Laplacian
    const REAL lap_u = (u_top + u_lft + u_bot + u_rgt
                        - REAL_CONST(4.0) * u_c) * dx2_inv;
    const REAL lap_v = (v_top + v_lft + v_bot + v_rgt
                        - REAL_CONST(4.0) * v_c) * dx2_inv;

    // Kinetic parameters
    const int param_base = b * N_PARAMS;
    const REAL Du = params[param_base + 0];
    const REAL Dv = params[param_base + 1];

    // --- Model-specific reaction terms ---
    ${REACTION_CODE}

    // Euler update: y_new = y_old + dt * (D * laplacian + reaction)
    y_out[center]       = u_c + dt * (Du * lap_u + f);
    y_out[BHW + center] = v_c + dt * (Dv * lap_v + g);
}
"""


# =============================================================================
# Fused PDE Function Kernel (for multi-stage solvers like RK4)
# Computes: dydt = D * laplacian(y_in) + reaction(y_in)
# Reads y_in, writes dydt. Boundary: dydt = 0.
# =============================================================================
FUSED_PDEFUNC_TEMPLATE = r"""
extern "C" __global__
void fused_pdefunc(
    const REAL* __restrict__ y_in,
    REAL* __restrict__ dydt,
    const REAL* __restrict__ params,
    const int B,
    const int H,
    const int W,
    const REAL dx2_inv
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * H * W;
    if (idx >= total) return;

    const int b  = idx / (H * W);
    const int hw = idx % (H * W);
    const int i  = hw / W;
    const int j  = hw % W;

    const int BHW    = B * H * W;
    const int center = b * (H * W) + i * W + j;

    // Boundary: dydt = 0
    if (i == 0 || i == H - 1 || j == 0 || j == W - 1) {
        dydt[center]       = REAL_CONST(0.0);
        dydt[BHW + center] = REAL_CONST(0.0);
        return;
    }

    const REAL u_c = y_in[center];
    const REAL v_c = y_in[BHW + center];

    // 5-point stencil neighbors for u
    const REAL u_top = y_in[center - W];
    const REAL u_bot = y_in[center + W];
    const REAL u_lft = y_in[center - 1];
    const REAL u_rgt = y_in[center + 1];

    // 5-point stencil neighbors for v
    const REAL v_top = y_in[BHW + center - W];
    const REAL v_bot = y_in[BHW + center + W];
    const REAL v_lft = y_in[BHW + center - 1];
    const REAL v_rgt = y_in[BHW + center + 1];

    // Discrete Laplacian
    const REAL lap_u = (u_top + u_lft + u_bot + u_rgt
                        - REAL_CONST(4.0) * u_c) * dx2_inv;
    const REAL lap_v = (v_top + v_lft + v_bot + v_rgt
                        - REAL_CONST(4.0) * v_c) * dx2_inv;

    // Kinetic parameters
    const int param_base = b * N_PARAMS;
    const REAL Du = params[param_base + 0];
    const REAL Dv = params[param_base + 1];

    // --- Model-specific reaction terms ---
    ${REACTION_CODE}

    // Write derivatives
    dydt[center]       = Du * lap_u + f;
    dydt[BHW + center] = Dv * lap_v + g;
}
"""


# =============================================================================
# RK Stage Update Kernel
# Computes: y_out[i] = y[i] + alpha * k[i]
# Used for intermediate RK stages (e.g., y_temp = y + 0.5*dt*k1)
# =============================================================================
RK_STAGE_UPDATE_SOURCE = r"""
extern "C" __global__
void rk_stage_update(
    const REAL* __restrict__ y,
    const REAL* __restrict__ k,
    REAL* __restrict__ y_out,
    const REAL alpha,
    const int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    y_out[i] = y[i] + alpha * k[i];
}
"""


# =============================================================================
# RK4 Combine Kernel
# Computes: delta[i] = (dt/6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
# Returns delta_y for the solver to apply: y_new = y + delta
# =============================================================================
RK4_COMBINE_SOURCE = r"""
extern "C" __global__
void rk4_combine(
    REAL* __restrict__ delta,
    const REAL* __restrict__ k1,
    const REAL* __restrict__ k2,
    const REAL* __restrict__ k3,
    const REAL* __restrict__ k4,
    const REAL dt_over_6,
    const int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    delta[i] = dt_over_6 * (k1[i]
                             + REAL_CONST(2.0) * k2[i]
                             + REAL_CONST(2.0) * k3[i]
                             + k4[i]);
}
"""


# =============================================================================
# Heun Combine Kernel
# Computes: delta[i] = 0.5 * (k1[i] + k2[i])
# =============================================================================
HEUN_COMBINE_SOURCE = r"""
extern "C" __global__
void heun_combine(
    REAL* __restrict__ delta,
    const REAL* __restrict__ k1,
    const REAL* __restrict__ k2,
    const int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    delta[i] = REAL_CONST(0.5) * (k1[i] + k2[i]);
}
"""


# =============================================================================
# Scale Kernel:  out[i] = alpha * x[i]
# =============================================================================
SCALE_SOURCE = r"""
extern "C" __global__
void scale(
    const REAL* __restrict__ x,
    REAL* __restrict__ out,
    const REAL alpha,
    const int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = alpha * x[i];
}
"""


# =============================================================================
# Linear Combine 2:  out[i] = a*x[i] + b*y[i]
# =============================================================================
LINEAR_COMBINE2_SOURCE = r"""
extern "C" __global__
void linear_combine2(
    const REAL* __restrict__ x,
    const REAL* __restrict__ y,
    REAL* __restrict__ out,
    const REAL a,
    const REAL b,
    const int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = a * x[i] + b * y[i];
}
"""


# =============================================================================
# Linear Combine 3:  out[i] = a*x[i] + b*y[i] + c*z[i]
# =============================================================================
LINEAR_COMBINE3_SOURCE = r"""
extern "C" __global__
void linear_combine3(
    const REAL* __restrict__ x,
    const REAL* __restrict__ y,
    const REAL* __restrict__ z,
    REAL* __restrict__ out,
    const REAL a,
    const REAL b,
    const REAL c,
    const int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = a * x[i] + b * y[i] + c * z[i];
}
"""
