"""
ctypes wrapper for the native libsolver.so shared library.

Usage:
    from lpf.kernels.aot.native import NativeSolver
    ns = NativeSolver()
    if ns.is_available():
        result_buf = ns.solve(solver_type, model, y_a, y_b, params, ...)
"""

import ctypes
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_THIS_DIR, "libsolver.so")

# Solver type codes
SOLVER_EULER = 0
SOLVER_HEUN = 1
SOLVER_RK4 = 2
SOLVER_RK23 = 3
SOLVER_AB2 = 4

# Model type codes
_MODEL_CODES = {
    "GrayScottModel": 0,
    "LiawModel": 1,
    "SchnakenbergModel": 2,
    "GiererMeinhardtModel": 3,
    "BrusselatorModel": 4,
    "FitzHughNagumoModel": 5,
    "LengyelEpsteinModel": 6,
    "ThomasModel": 7,
    "BarkleyModel": 8,
}

# Dtype codes
_DTYPE_CODES = {
    np.dtype(np.float16): 0,
    np.dtype(np.float32): 1,
    np.dtype(np.float64): 2,
}

# Number of work buffers per solver
_WORK_COUNT = {
    SOLVER_EULER: 0,
    SOLVER_HEUN: 3,   # k1, k2, y_temp
    SOLVER_RK4: 6,    # k1, k2, k3, k4, y_temp, delta
    SOLVER_RK23: 5,   # k1, k2, k3, y_temp, delta
    SOLVER_AB2: 3,     # dydt_a, dydt_b, delta
}


class NativeSolver:
    """Thin ctypes wrapper around the lpf_native_solve C function."""

    def __init__(self):
        self._lib = None
        self._available = False
        self._try_load()

    def _try_load(self):
        if not os.path.isfile(_LIB_PATH):
            return
        try:
            self._lib = ctypes.CDLL(_LIB_PATH)
            fn = self._lib.lpf_native_solve
            fn.restype = ctypes.c_int
            # argtypes set loosely — we marshal manually
            self._available = True
        except Exception:
            self._available = False

    def is_available(self):
        return self._available

    def solve(self, solver_type, model_name, dtype,
              y_a, y_b, params,
              B, H, W, dt, dx2_inv, n_iters,
              work_bufs=None, trj_buf=None, trj_iters=None,
              rtol=0.0, rtol_period=1000):
        """Run the full solve loop in native C/CUDA.

        Parameters
        ----------
        solver_type : int
            SOLVER_EULER, SOLVER_HEUN, etc.
        model_name : str
            e.g. "LiawModel"
        dtype : numpy dtype
        y_a, y_b : cupy arrays
            Ping-pong state buffers.
        params : cupy array
        B, H, W : int
        dt, dx2_inv : float
        n_iters : int
        work_bufs : list of cupy arrays or None
            Solver-specific working buffers.
        trj_buf : cupy array or None
            Pre-allocated trajectory buffer.
        trj_iters : numpy int32 array or None
            1-based iteration numbers for trajectory capture (host).
        rtol : float
            Early stopping tolerance (0 = disabled).
        rtol_period : int
            Check convergence every N iterations.

        Returns
        -------
        dict with keys:
            'result_buf': 0 if final state in y_a, 1 if in y_b
            'stopped_at': iteration where stopped, -1 if completed
        """
        model_code = _MODEL_CODES[model_name]
        dtype_code = _DTYPE_CODES[np.dtype(dtype)]

        # Work buffer pointers
        n_work = _WORK_COUNT[solver_type]
        WorkArray = ctypes.c_void_p * max(n_work, 1)
        c_work = WorkArray()
        if work_bufs:
            for i, buf in enumerate(work_bufs[:n_work]):
                c_work[i] = ctypes.c_void_p(buf.data.ptr)

        # Trajectory
        c_trj_buf = ctypes.c_void_p(trj_buf.data.ptr if trj_buf is not None else 0)
        if trj_iters is not None:
            trj_iters = np.ascontiguousarray(trj_iters, dtype=np.int32)
            c_trj_iters = trj_iters.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            c_n_trj = len(trj_iters)
        else:
            c_trj_iters = ctypes.POINTER(ctypes.c_int)()
            c_n_trj = 0

        # Output
        stopped_at = ctypes.c_int(-1)

        result_buf = self._lib.lpf_native_solve(
            ctypes.c_void_p(y_a.data.ptr),
            ctypes.c_void_p(y_b.data.ptr),
            ctypes.c_void_p(params.data.ptr),
            ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(W),
            ctypes.c_double(dt), ctypes.c_double(dx2_inv),
            ctypes.c_int(n_iters),
            ctypes.c_int(solver_type), ctypes.c_int(model_code),
            ctypes.c_int(dtype_code),
            ctypes.cast(c_work, ctypes.POINTER(ctypes.c_void_p)),
            c_trj_buf,
            c_trj_iters,
            ctypes.c_int(c_n_trj),
            ctypes.c_double(rtol),
            ctypes.c_int(rtol_period),
            ctypes.byref(stopped_at),
        )

        return {
            'result_buf': result_buf,
            'stopped_at': stopped_at.value,
        }
