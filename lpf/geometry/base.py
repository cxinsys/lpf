"""Spatial discretization of a reaction-diffusion model.

A geometry decides where a model's state lives and how it diffuses:

    GridGeometry    the rectangular grid, LPF's default. The outermost cells
                    are held fixed and the 5-point stencil acts inside.
    DomainGeometry  part of the grid given by an lpf.domains.Domain, with a
                    zero-flux boundary (cut-cell finite volumes).
    MeshGeometry    the vertices of an lpf.surfaces.SurfaceMesh (cotangent
                    Laplace-Beltrami operator).

Models delegate everything that depends on space to their geometry: the
state shape, the Laplacian of the array-module path (NumPy, CuPy, PyTorch),
the CUDA kernel launches, the JAX right-hand side, the explicit stability
limit, picture output and (de)serialization. Model code is therefore the
same for every geometry, and a new geometry is one new subclass.

Arrays of the array-module path come in two layouts:
    state layout     (B, *spatial_shape), a component of the model state.
    reaction layout  (B, X, Y), what a model's reactions() receive; kinetic
                     parameters broadcast against it as (B, 1, 1).
"""

from abc import ABC, abstractmethod


class Geometry(ABC):
    """Base class of spatial discretizations (see the module docstring).

    A geometry is created with its model (make_geometry) and bound to the
    model's array module, dtype and batch size in model.initialize().
    """

    def __init__(self):
        self._am = None
        self._dtype = None
        self._batch_size = None
        self._kernel_ops = {}

    # ---- description ----

    @property
    @abstractmethod
    def spatial_shape(self):
        """Shape of one state component of one model, e.g. (H, W) or (N,)."""

    @property
    def domain(self):
        """The lpf.domains.Domain (or one per model), or None."""
        return None

    @property
    def mesh(self):
        """The lpf.surfaces.SurfaceMesh, or None."""
        return None

    def is_compatible(self, other):
        """Whether other discretizes the same space (e.g. for the paternal
        and maternal models of a diploid model)."""
        return type(other) is type(self)

    # ---- binding to a model ----

    def bind(self, am, dtype, batch_size):
        """Prepare the operator for a model's array module, dtype and batch
        size. Called (inside ``with am:``) by model.initialize()."""
        self._am = am
        self._dtype = dtype
        self._batch_size = batch_size

    # ---- array-module path ----

    @abstractmethod
    def reaction_view(self, a):
        """State layout -> reaction layout (the cells that are solved)."""

    @abstractmethod
    def laplacian(self, a):
        """Laplacian of a state-layout array, in reaction layout."""

    def restrict(self, f):
        """Zero a reaction-layout array where nothing is solved."""
        return f

    @abstractmethod
    def assemble(self, dydt, du, dv):
        """Write du/dv (reaction layout) into dydt (2, B, *spatial_shape)
        and return the result (a new array on functional backends)."""

    # ---- solvers ----

    # Whether solvers warn when dt exceeds the explicit stability limit. On
    # the grid (and any domain) the limit is the familiar dx^2 / (4 D); on a
    # mesh it depends on the smallest elements, so it is worth a warning.
    warns_unstable_dt = False

    @abstractmethod
    def lambda_max(self):
        """Largest eigenvalue of -laplacian for unit diffusivity. An explicit
        method with stability interval c is stable for dt * D * lambda <= c."""

    def kernel_op(self, km):
        """CUDA kernels of this geometry for kernel manager km (lpf.kernels),
        as an lpf.kernels.operators.KernelOp. Cached, so repeated solves do
        not upload the operator again."""
        key = (km, km.using_aot, self._batch_size)
        if key not in self._kernel_ops:
            self._kernel_ops[key] = self._make_kernel_op(km)
        return self._kernel_ops[key]

    @abstractmethod
    def _make_kernel_op(self, km):
        pass

    @abstractmethod
    def jax_rhs(self, model):
        """(rhs, cache_key, params) of a haploid model for the JAX solvers:
        rhs(y, params, dx_inv2) -> dy/dt, params a pytree of arrays."""

    def jax_diploid_rhs(self, model):
        """Like jax_rhs() for a TwoComponentDiploidModel."""
        raise ValueError("JAX solver: diploid models are not supported on a "
                         "%s." % type(self).__name__)

    # ---- output and serialization ----

    def outside_mask(self):
        """NumPy bool array (B or 1, *spatial_shape) of the points outside
        the medium (drawn as background), or None."""
        return None

    @abstractmethod
    def save_color(self, model, index, arr_color, fpath_morph, fpath_pattern):
        """Write the colors of model index (arr_color from model.colorize())."""

    def to_dict(self, index=0):
        """Entries describing the geometry in a model dictionary."""
        return {}

    def save_assets(self, model_dict, dpath):
        """Write files the model dictionary refers to into directory dpath."""

    @classmethod
    def kwargs_from_dicts(cls, model_dicts, dpath="."):
        """Model constructor keyword arguments that rebuild this kind of
        geometry from model dictionaries, or None if they describe another
        kind."""
        return None
