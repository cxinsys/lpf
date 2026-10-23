import numpy as np

from lpf.geometry.grid import GridGeometry


class DomainGeometry(GridGeometry):
    """The part of the grid inside an lpf.domains.Domain.

    The domain boundary has zero flux and cells outside keep their values;
    see lpf.domains for the cut-cell weights. ``domain`` is one Domain for
    the whole batch or a sequence with one Domain per model.
    """

    def __init__(self, domain, height, width, dx):
        super().__init__(height, width, dx)
        self._domain = domain
        self._stencils = None     # NumPy (B or 1, 5, H, W), see stencils()
        self._stencils_am = None  # the same on the model's array module

    @property
    def domain(self):
        return self._domain

    @property
    def domains(self):
        """The domains as a list (one entry if shared by the batch)."""
        if isinstance(self._domain, (list, tuple)):
            return list(self._domain)
        return [self._domain]

    def is_compatible(self, other):
        return (super().is_compatible(other)
                and [d.to_dict() for d in other.domains]
                == [d.to_dict() for d in self.domains])

    def stencils(self):
        """Stencils as a NumPy array (B or 1, 5, H, W); see Domain.stencil."""
        if self._stencils is None:
            stencils = np.stack([d.stencil(self._height, self._width)
                                 for d in self.domains])
            stencils.flags.writeable = False
            self._stencils = stencils
        return self._stencils

    def bind(self, am, dtype, batch_size):
        super().bind(am, dtype, batch_size)
        n = len(self.domains)
        if isinstance(self._domain, (list, tuple)) and n != batch_size:
            raise ValueError("Got %d domains for a batch of %d."
                             % (n, batch_size))
        self._stencils_am = am.array(self.stencils(), dtype=dtype)

    # ---- array-module path ----

    def reaction_view(self, a):
        return a

    def laplacian(self, a):
        """Domain-weighted Laplacian of (B, H, W), zero-flux at the domain
        boundary, in the accumulation order of the CUDA kernel:
        wN*(uN - u) + wS*(uS - u) + wW*(uW - u) + wE*(uE - u)."""
        am, st = self._am, self._stencils_am  # st: (B or 1, 5, H, W)
        lap = am.zeros(a.shape, dtype=self._dtype)
        lap = am.set(
            lap, (slice(None), slice(1, None), slice(None)),
            st[:, 0, 1:, :] * (a[:, :-1, :] - a[:, 1:, :]))
        lap = am.set(
            lap, (slice(None), slice(None, -1), slice(None)),
            lap[:, :-1, :] + st[:, 1, :-1, :] * (a[:, 1:, :] - a[:, :-1, :]))
        lap = am.set(
            lap, (slice(None), slice(None), slice(1, None)),
            lap[:, :, 1:] + st[:, 2, :, 1:] * (a[:, :, :-1] - a[:, :, 1:]))
        lap = am.set(
            lap, (slice(None), slice(None), slice(None, -1)),
            lap[:, :, :-1] + st[:, 3, :, :-1] * (a[:, :, 1:] - a[:, :, :-1]))
        return lap / self._dx ** 2

    def restrict(self, f):
        return f * self._stencils_am[:, 4]

    def assemble(self, dydt, du, dv):
        # Inactive cells keep their values: dy/dt = 0 there.
        am, active = self._am, self._stencils_am[:, 4]
        dydt = am.set(dydt, (0, ...), du * active)
        dydt = am.set(dydt, (1, ...), dv * active)
        return dydt

    # ---- solvers ----

    def lambda_max(self):
        # Gershgorin bound 2 max_i sum_f w_f / dx^2. The cut-cell weights are
        # limited to sum_f w_f <= 4 (lpf.domains), so this never exceeds the
        # grid's 8 / dx^2.
        weights = self.stencils()[:, :4].sum(axis=1)
        return 2.0 * float(weights.max()) / self._dx ** 2

    def _make_kernel_op(self, km):
        import cupy as cp
        from lpf.kernels.operators import DomainKernelOp
        # Weights as the kernels see them (in the model's dtype).
        stencils = self.stencils().astype(self._dtype)
        stride = 0 if stencils.shape[0] == 1 else stencils[0].size
        active = stencils[:, 4] > 0
        full = (stencils[:, :4] == 1).all(axis=1)
        flags = np.where(active, np.where(full, 1, 2), 0).astype(np.uint8)
        return DomainKernelOp(km, self._batch_size, self._height, self._width,
                              self._dx, cp.asarray(stencils), stride,
                              cp.asarray(flags))

    def jax_rhs(self, model):
        from lpf.solvers._jax.rhs import build_domain_rhs
        return (build_domain_rhs(model), ("domain", type(model).__name__),
                (model.params, self._stencils_am))

    def jax_diploid_rhs(self, model):
        raise ValueError("JAX solver: diploid models with a domain are not "
                         "supported.")

    # ---- output and serialization ----

    def outside_mask(self):
        return self.stencils()[:, 4] == 0

    def to_dict(self, index=0):
        domains = self.domains
        domain = domains[index or 0] if len(domains) > 1 else domains[0]
        return {"domain": domain.to_dict()}

    @classmethod
    def kwargs_from_dicts(cls, model_dicts, dpath="."):
        domains = [n2v.get("domain") for n2v in model_dicts]
        if all(d is None for d in domains):
            return None
        if any(d is None for d in domains):
            raise ValueError("Either all or none of the models of a batch "
                             "must have a domain.")
        from lpf.domains import domain_from_dict
        if all(d == domains[0] for d in domains):
            return {"domain": domain_from_dict(domains[0])}
        return {"domain": [domain_from_dict(d) for d in domains]}
