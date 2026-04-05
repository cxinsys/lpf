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
# Tile dimensions for shared-memory stencil (must match compiler.py)
# =============================================================================
TILE_X = 32
TILE_Y = 8


# =============================================================================
# Fused Euler Step Kernel (shared-memory tiled, 2-D grid)
# Computes: y_out = y_in + dt * (D * laplacian(y_in) + reaction(y_in))
# Uses ping-pong buffers: reads y_in, writes y_out
# Grid: (ceil(W/TILE_X), ceil(H/TILE_Y), B)  Block: (TILE_X, TILE_Y)
# =============================================================================
FUSED_EULER_STEP_TEMPLATE = r"""
#define TILE_X 32
#define TILE_Y 8
#define SW (TILE_X + 2)

extern "C" __global__
void fused_euler_step(
    const REAL* __restrict__ y_in,
    REAL* __restrict__ y_out,
    const REAL* __restrict__ params,
    const int B,
    const int H,
    const int W,
    const double dt_d,
    const double dx2_inv_d
) {
    const REAL dt = (REAL)dt_d;
    const REAL dx2_inv = (REAL)dx2_inv_d;
    __shared__ REAL s_u[(TILE_Y + 2) * SW];
    __shared__ REAL s_v[(TILE_Y + 2) * SW];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gx = blockIdx.x * TILE_X + tx;
    const int gy = blockIdx.y * TILE_Y + ty;
    const int b  = blockIdx.z;
    const int HW = H * W, BHW = B * HW;
    const int gidx = b * HW + gy * W + gx;
    const int si = (ty + 1) * SW + (tx + 1);
    const int valid = (gx < W && gy < H);

    // Load center
    const REAL u_val = valid ? y_in[gidx]       : REAL_CONST(0.0);
    const REAL v_val = valid ? y_in[BHW + gidx] : REAL_CONST(0.0);
    s_u[si] = u_val;  s_v[si] = v_val;

    // Load halos
    if (ty == 0) {
        const int h = tx + 1, h_gy = gy - 1;
        if (h_gy >= 0 && gx < W) { int hi = b*HW + h_gy*W + gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    if (ty == TILE_Y - 1) {
        const int h = (TILE_Y+1)*SW + (tx+1), h_gy = gy + 1;
        if (h_gy < H && gx < W) { int hi = b*HW + h_gy*W + gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    if (tx == 0) {
        const int h = (ty+1)*SW, h_gx = gx - 1;
        if (h_gx >= 0 && gy < H) { int hi = b*HW + gy*W + h_gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    if (tx == TILE_X - 1) {
        const int h = (ty+1)*SW + (TILE_X+1), h_gx = gx + 1;
        if (h_gx < W && gy < H) { int hi = b*HW + gy*W + h_gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    __syncthreads();

    if (!valid) return;

    // Boundary: dydt = 0 (Neumann), so y_out = y_in
    if (gx == 0 || gx == W-1 || gy == 0 || gy == H-1) {
        y_out[gidx]       = u_val;
        y_out[BHW + gidx] = v_val;
        return;
    }

    // 5-point stencil from shared memory
    const REAL u_c = s_u[si], v_c = s_v[si];
    const REAL lap_u = (s_u[si-SW] + s_u[si-1] + s_u[si+SW] + s_u[si+1]
                        - REAL_CONST(4.0) * u_c) * dx2_inv;
    const REAL lap_v = (s_v[si-SW] + s_v[si-1] + s_v[si+SW] + s_v[si+1]
                        - REAL_CONST(4.0) * v_c) * dx2_inv;

    // Kinetic parameters
    const int param_base = b * N_PARAMS;
    const REAL Du = params[param_base + 0];
    const REAL Dv = params[param_base + 1];

    // --- Model-specific reaction terms ---
    ${REACTION_CODE}

    // Euler update: y_new = y_old + dt * (D * laplacian + reaction)
    y_out[gidx]       = u_c + dt * (Du * lap_u + f);
    y_out[BHW + gidx] = v_c + dt * (Dv * lap_v + g);
}
"""


# =============================================================================
# Fused PDE Function Kernel (shared-memory tiled, 2-D grid)
# Computes: dydt = D * laplacian(y_in) + reaction(y_in)
# Reads y_in, writes dydt. Boundary: dydt = 0.
# Grid: (ceil(W/TILE_X), ceil(H/TILE_Y), B)  Block: (TILE_X, TILE_Y)
# =============================================================================
FUSED_PDEFUNC_TEMPLATE = r"""
#define TILE_X 32
#define TILE_Y 8
#define SW (TILE_X + 2)

extern "C" __global__
void fused_pdefunc(
    const REAL* __restrict__ y_in,
    REAL* __restrict__ dydt,
    const REAL* __restrict__ params,
    const int B,
    const int H,
    const int W,
    const double dx2_inv_d
) {
    const REAL dx2_inv = (REAL)dx2_inv_d;
    __shared__ REAL s_u[(TILE_Y + 2) * SW];
    __shared__ REAL s_v[(TILE_Y + 2) * SW];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gx = blockIdx.x * TILE_X + tx;
    const int gy = blockIdx.y * TILE_Y + ty;
    const int b  = blockIdx.z;
    const int HW = H * W, BHW = B * HW;
    const int gidx = b * HW + gy * W + gx;
    const int si = (ty + 1) * SW + (tx + 1);
    const int valid = (gx < W && gy < H);

    // Load center
    const REAL u_val = valid ? y_in[gidx]       : REAL_CONST(0.0);
    const REAL v_val = valid ? y_in[BHW + gidx] : REAL_CONST(0.0);
    s_u[si] = u_val;  s_v[si] = v_val;

    // Load halos
    if (ty == 0) {
        const int h = tx + 1, h_gy = gy - 1;
        if (h_gy >= 0 && gx < W) { int hi = b*HW + h_gy*W + gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    if (ty == TILE_Y - 1) {
        const int h = (TILE_Y+1)*SW + (tx+1), h_gy = gy + 1;
        if (h_gy < H && gx < W) { int hi = b*HW + h_gy*W + gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    if (tx == 0) {
        const int h = (ty+1)*SW, h_gx = gx - 1;
        if (h_gx >= 0 && gy < H) { int hi = b*HW + gy*W + h_gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    if (tx == TILE_X - 1) {
        const int h = (ty+1)*SW + (TILE_X+1), h_gx = gx + 1;
        if (h_gx < W && gy < H) { int hi = b*HW + gy*W + h_gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    __syncthreads();

    if (!valid) return;

    // Boundary: dydt = 0
    if (gx == 0 || gx == W-1 || gy == 0 || gy == H-1) {
        dydt[gidx]       = REAL_CONST(0.0);
        dydt[BHW + gidx] = REAL_CONST(0.0);
        return;
    }

    // 5-point stencil from shared memory
    const REAL u_c = s_u[si], v_c = s_v[si];
    const REAL lap_u = (s_u[si-SW] + s_u[si-1] + s_u[si+SW] + s_u[si+1]
                        - REAL_CONST(4.0) * u_c) * dx2_inv;
    const REAL lap_v = (s_v[si-SW] + s_v[si-1] + s_v[si+SW] + s_v[si+1]
                        - REAL_CONST(4.0) * v_c) * dx2_inv;

    // Kinetic parameters
    const int param_base = b * N_PARAMS;
    const REAL Du = params[param_base + 0];
    const REAL Dv = params[param_base + 1];

    // --- Model-specific reaction terms ---
    ${REACTION_CODE}

    // Write derivatives
    dydt[gidx]       = Du * lap_u + f;
    dydt[BHW + gidx] = Dv * lap_v + g;
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
    const double alpha_d,
    const int N
) {
    const REAL alpha = (REAL)alpha_d;
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
    const double dt_over_6_d,
    const int N
) {
    const REAL dt_over_6 = (REAL)dt_over_6_d;
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
    const double alpha_d,
    const int N
) {
    const REAL alpha = (REAL)alpha_d;
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
    const double a_d,
    const double b_d,
    const int N
) {
    const REAL a = (REAL)a_d;
    const REAL b = (REAL)b_d;
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
    const double a_d,
    const double b_d,
    const double c_d,
    const int N
) {
    const REAL a = (REAL)a_d;
    const REAL b = (REAL)b_d;
    const REAL c = (REAL)c_d;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = a * x[i] + b * y[i] + c * z[i];
}
"""
