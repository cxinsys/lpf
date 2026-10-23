import numpy as np

from lpf.geometry.base import Geometry


def grid_laplacian(a, dx):
    """5-point Laplacian of a (B, H, W) array on its interior (B, H-2, W-2)."""
    a_top = a[:, 0:-2, 1:-1]
    a_left = a[:, 1:-1, 0:-2]
    a_bottom = a[:, 2:, 1:-1]
    a_right = a[:, 1:-1, 2:]
    a_center = a[:, 1:-1, 1:-1]
    return (a_top + a_left + a_bottom + a_right - 4 * a_center) / dx ** 2


_INTERIOR = (slice(None), slice(1, -1), slice(1, -1))
_BORDERS = ((slice(None), slice(None), 0, slice(None)),
            (slice(None), slice(None), -1, slice(None)),
            (slice(None), slice(None), slice(None), 0),
            (slice(None), slice(None), slice(None), -1))


class GridGeometry(Geometry):
    """The height x width grid with spacing dx (LPF's default space).

    The outermost cells are held fixed (dy/dt = 0); the interior diffuses
    with the 5-point stencil.
    """

    def __init__(self, height, width, dx):
        super().__init__()
        self._height = height
        self._width = width
        self._dx = dx

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def dx(self):
        return self._dx

    @property
    def spatial_shape(self):
        return (self._height, self._width)

    def is_compatible(self, other):
        return (super().is_compatible(other)
                and other.spatial_shape == self.spatial_shape
                and other.dx == self.dx)

    # ---- array-module path ----

    def reaction_view(self, a):
        return a[_INTERIOR]

    def laplacian(self, a):
        return grid_laplacian(a, self._dx)

    def assemble(self, dydt, du, dv):
        am = self._am
        dydt = am.set(dydt, (0, *_INTERIOR), du)
        dydt = am.set(dydt, (1, *_INTERIOR), dv)
        for border in _BORDERS:  # boundary condition: dy/dt = 0 at the edges
            dydt = am.set(dydt, border, 0.0)
        return dydt

    # ---- solvers ----

    def lambda_max(self):
        # Eigenvalues of the 1D Laplacian with n interior points between two
        # fixed ends: (4 / dx^2) sin^2(k pi / (2 (n + 1))), k = 1..n.
        lam = 0.0
        for n in (self._height - 2, self._width - 2):
            if n > 0:
                lam += 4.0 * np.sin(n * np.pi / (2 * (n + 1))) ** 2
        return lam / self._dx ** 2

    def _make_kernel_op(self, km):
        from lpf.kernels.operators import GridKernelOp
        return GridKernelOp(km, self._batch_size, self._height, self._width,
                            self._dx)

    def jax_rhs(self, model):
        from lpf.solvers._jax.rhs import get_rhs
        return get_rhs(model.name), model.name, model.params

    def jax_diploid_rhs(self, model):
        from lpf.solvers._jax.rhs import build_diploid_rhs, is_diploid_supported
        pa, ma = model.paternal_model, model.maternal_model
        if not is_diploid_supported(model):
            raise ValueError(
                f"JAX solver: unsupported diploid sub-models "
                f"(paternal={pa.name}, maternal={ma.name}).")
        return (build_diploid_rhs(pa.name, ma.name),
                ("diploid", pa.name, ma.name),
                (pa.params, ma.params))

    # ---- output ----

    def save_color(self, model, index, arr_color, fpath_morph, fpath_pattern):
        morph, pattern = model.create_image(index, arr_color)
        if fpath_morph:
            morph.save(fpath_morph)
        if fpath_pattern:
            pattern.save(fpath_pattern)
