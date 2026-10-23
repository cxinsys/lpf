"""Fused CUDA kernels of one model on one spatial discretization.

A KernelOp is the Python counterpart of the operator of the native solve
loop (Operator<T, R, Space> in csrc/solver_loop.cu). It is created by the
model's geometry (lpf.geometry) for a CuKernelManager, resolves the kernels
and their launch configuration once, and then only launches. Each method is
one fused kernel (csrc/include/rd_kernels.cuh) that evaluates f(y) = dy/dt
and applies an update in the same pass:

    euler(y, y_next, params, dt)                y_next = y + dt f(y)
    pdefunc(y, dydt, params)                    dydt = f(y)
    stage(y, base, k, y_next, params, alpha)    k = f(y), y_next = base + alpha k
    heun(y, base, k1, y_next, params, a, b)     y_next = base + (a k1 + b f(y))
    rk23(y, base, k1, k2, y_next, params, a, b, c)
    rk4(y, base, k1, k2, k3, y_next, params, d) y_next = base + d (k1 + 2 k2 + 2 k3 + f(y))
    ab2(y, prev, cur, y_next, params, a, b)     cur = f(y), y_next = y + (a cur + b prev)
    native_solve(native, **loop)                the whole loop in libsolver

Arrays are CuPy arrays of the state layout (2, B, *spatial_shape); y is
never written, and y_next must not alias y (neighbors are read from it).
"""

import numpy as np

from lpf.kernels.aot.loader import TILE_X, TILE_Y


class KernelOp:
    """Base class; subclasses set the space, launch shape and arguments."""

    space = None  # kernel name part: "grid", "domain", "mesh" or "meshb"
    native_kind = None  # libsolver entry point: "grid", "domain", "surface"

    def __init__(self, km, grid, block, space_args):
        self._km = km
        self._grid = grid
        self._block = block
        self._space_args = space_args  # arguments before (y, params, ...)
        self._kernels = {}

    def _launch(self, epilogue, args):
        kernel = self._kernels.get(epilogue)
        if kernel is None:
            kernel = self._kernels[epilogue] = self._km.kernel(epilogue,
                                                               self.space)
        kernel(self._grid, self._block, self._space_args + args)

    def euler(self, y, y_next, params, dt):
        self._launch("euler", (y, params, y_next, np.float64(dt)))

    def pdefunc(self, y, dydt, params):
        self._launch("rhs", (y, params, dydt))

    def stage(self, y, base, k, y_next, params, alpha):
        self._launch("stage", (y, params, k, y_next, base, np.float64(alpha)))

    def heun(self, y, base, k1, y_next, params, a, b):
        self._launch("heun", (y, params, y_next, base, k1,
                              np.float64(a), np.float64(b)))

    def rk23(self, y, base, k1, k2, y_next, params, a, b, c):
        self._launch("rk23", (y, params, y_next, base, k1, k2,
                              np.float64(a), np.float64(b), np.float64(c)))

    def rk4(self, y, base, k1, k2, k3, y_next, params, d):
        self._launch("rk4", (y, params, y_next, base, k1, k2, k3,
                             np.float64(d)))

    def ab2(self, y, prev, cur, y_next, params, a, b):
        self._launch("ab2", (y, params, cur, y_next, prev,
                             np.float64(a), np.float64(b)))

    def native_solve(self, native, **loop):
        raise NotImplementedError()


class GridKernelOp(KernelOp):
    """The full grid (lpf.geometry.GridGeometry)."""

    space = "grid"
    native_kind = "grid"

    def __init__(self, km, B, H, W, dx):
        self._B, self._H, self._W = B, H, W
        self._dx2_inv = 1.0 / (dx ** 2)
        super().__init__(km, _tiled_grid(B, H, W), (TILE_X, TILE_Y),
                         (np.int32(B), np.int32(H), np.int32(W),
                          np.float64(self._dx2_inv)))

    def native_solve(self, native, **loop):
        return native.solve(B=self._B, H=self._H, W=self._W,
                            dx2_inv=self._dx2_inv, **loop)


class DomainKernelOp(GridKernelOp):
    """A domain on the grid (lpf.geometry.DomainGeometry). stencil is a CuPy
    array (B or 1, 5, H, W); model b uses stencil + b * stride. flags
    (B or 1, H, W) uint8 classify the cells: 0 inactive, 1 all weights 1,
    2 other active cells."""

    space = "domain"
    native_kind = "domain"

    def __init__(self, km, B, H, W, dx, stencil, stride, flags):
        super().__init__(km, B, H, W, dx)
        self._space_args += (stencil, np.int64(stride), flags)
        self._stencil, self._stride, self._flags = stencil, stride, flags

    def native_solve(self, native, **loop):
        return native.solve_domain(stencil=self._stencil, flags=self._flags,
                                   stencil_stride=self._stride,
                                   B=self._B, H=self._H, W=self._W,
                                   dx2_inv=self._dx2_inv, **loop)


class SurfaceKernelOp(KernelOp):
    """A triangle-mesh surface (lpf.geometry.MeshGeometry). nbr/wts are the
    operator in transposed ELL format, CuPy arrays (K, N). A batch of
    several models runs four per thread ("meshb"), which reads the operator
    once for all four."""

    native_kind = "surface"
    BLOCK = 256

    def __init__(self, km, B, N, nbr, wts):
        K = nbr.shape[0]
        self.space, chunk = ("mesh", 1) if B == 1 else ("meshb", 4)
        super().__init__(km, ((N + self.BLOCK - 1) // self.BLOCK,
                              (B + chunk - 1) // chunk),
                         (self.BLOCK,),
                         (np.int32(B), np.int32(N), np.int32(K), nbr, wts))
        self._B, self._N, self._K = B, N, K
        self._nbr, self._wts = nbr, wts

    def native_solve(self, native, **loop):
        return native.solve_surface(nbr=self._nbr, wts=self._wts, K=self._K,
                                    B=self._B, N=self._N, **loop)


def _tiled_grid(B, H, W):
    return ((W + TILE_X - 1) // TILE_X, (H + TILE_Y - 1) // TILE_Y, B)
