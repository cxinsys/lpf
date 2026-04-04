"""
CUDA reaction code registry for each reaction-diffusion model type.

Each entry maps model_name -> dict with:
  - n_params: number of kinetic parameters per batch element
  - code: CUDA code snippet computing reaction terms f and g
          Expects: u_c, v_c (center values), params (parameter array),
                   param_base (offset into params for this batch element)
          Must define: REAL f, g
"""

REACTION_REGISTRY = {

    "GrayScottModel": {
        "n_params": 4,
        "code": """
            // Gray-Scott: f = -u*v^2 + F*(1-u), g = u*v^2 - (F+k)*v
            const REAL F_p = params[param_base + 2];
            const REAL k_p = params[param_base + 3];
            const REAL u_vsq = u_c * v_c * v_c;
            const REAL f = -u_vsq + F_p * (REAL_CONST(1.0) - u_c);
            const REAL g = u_vsq - (F_p + k_p) * v_c;
        """
    },

    "LiawModel": {
        "n_params": 8,
        "code": """
            // Liaw: f = ru*(u^2*v)/(1+k*u^2) + su - mu*u
            //        g = -rv*(u^2*v)/(1+k*u^2) + sv
            const REAL ru_p = params[param_base + 2];
            const REAL rv_p = params[param_base + 3];
            const REAL k_p  = params[param_base + 4];
            const REAL su_p = params[param_base + 5];
            const REAL sv_p = params[param_base + 6];
            const REAL mu_p = params[param_base + 7];
            const REAL u_sq = u_c * u_c;
            const REAL u2v_term = u_sq * v_c / (REAL_CONST(1.0) + k_p * u_sq);
            const REAL f = ru_p * u2v_term + su_p - mu_p * u_c;
            const REAL g = -rv_p * u2v_term + sv_p;
        """
    },

    "SchnakenbergModel": {
        "n_params": 6,
        "code": """
            // Schnakenberg: f = su - mu*u + rho*u^2*v, g = sv - rho*u^2*v
            const REAL rho_p = params[param_base + 2];
            const REAL su_p  = params[param_base + 3];
            const REAL sv_p  = params[param_base + 4];
            const REAL mu_p  = params[param_base + 5];
            const REAL usq_v = u_c * u_c * v_c;
            const REAL f = su_p - mu_p * u_c + rho_p * usq_v;
            const REAL g = sv_p - rho_p * usq_v;
        """
    },

    "GiererMeinhardtModel": {
        "n_params": 6,
        "code": """
            // Gierer-Meinhardt: f = ru*u^2/v - mu*u, g = rv*u^2 - nu*v
            const REAL ru_p = params[param_base + 2];
            const REAL rv_p = params[param_base + 3];
            const REAL mu_p = params[param_base + 4];
            const REAL nu_p = params[param_base + 5];
            const REAL usq = u_c * u_c;
            const REAL f = ru_p * usq / v_c - mu_p * u_c;
            const REAL g = rv_p * usq - nu_p * v_c;
        """
    },

    "BrusselatorModel": {
        "n_params": 4,
        "code": """
            // Brusselator: f = A-(B+1)*u+u^2*v, g = B*u-u^2*v
            const REAL A_p = params[param_base + 2];
            const REAL B_p = params[param_base + 3];
            const REAL usq_v = u_c * u_c * v_c;
            const REAL f = A_p - (B_p + REAL_CONST(1.0)) * u_c + usq_v;
            const REAL g = B_p * u_c - usq_v;
        """
    },

    "FitzHughNagumoModel": {
        "n_params": 5,
        "code": """
            // FitzHugh-Nagumo: f = u-u^3-v, g = eps*(u-gamma*v+beta)
            const REAL eps_p   = params[param_base + 2];
            const REAL gamma_p = params[param_base + 3];
            const REAL beta_p  = params[param_base + 4];
            const REAL f = u_c - u_c * u_c * u_c - v_c;
            const REAL g = eps_p * (u_c - gamma_p * v_c + beta_p);
        """
    },

    "LengyelEpsteinModel": {
        "n_params": 4,
        "code": """
            // Lengyel-Epstein: f = a-u-4uv/(1+u^2), g = b*(u-uv/(1+u^2))
            const REAL a_p = params[param_base + 2];
            const REAL b_p = params[param_base + 3];
            const REAL uv_sat = u_c * v_c / (REAL_CONST(1.0) + u_c * u_c);
            const REAL f = a_p - u_c - REAL_CONST(4.0) * uv_sat;
            const REAL g = b_p * (u_c - uv_sat);
        """
    },

    "ThomasModel": {
        "n_params": 7,
        "code": """
            // Thomas: f = a-u-rho*uv/(1+u+K*u^2), g = alpha*(b-v)-rho*uv/(1+u+K*u^2)
            const REAL a_p     = params[param_base + 2];
            const REAL b_p     = params[param_base + 3];
            const REAL rho_p   = params[param_base + 4];
            const REAL alpha_p = params[param_base + 5];
            const REAL K_p     = params[param_base + 6];
            const REAL uv_term = rho_p * u_c * v_c / (REAL_CONST(1.0) + u_c + K_p * u_c * u_c);
            const REAL f = a_p - u_c - uv_term;
            const REAL g = alpha_p * (b_p - v_c) - uv_term;
        """
    },

    "BarkleyModel": {
        "n_params": 5,
        "code": """
            // Barkley: f = (1/eps)*u*(1-u)*(u-(v+b)/a), g = u-v
            const REAL eps_p = params[param_base + 2];
            const REAL a_p   = params[param_base + 3];
            const REAL b_p   = params[param_base + 4];
            const REAL f = (REAL_CONST(1.0) / eps_p) * u_c * (REAL_CONST(1.0) - u_c) * (u_c - (v_c + b_p) / a_p);
            const REAL g = u_c - v_c;
        """
    },

}
