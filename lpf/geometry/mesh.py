import os.path as osp

from lpf.geometry.base import Geometry


class MeshGeometry(Geometry):
    """The vertices of an lpf.surfaces.SurfaceMesh.

    The state is (2, B, n_vertices) and diffuses with the cotangent
    Laplace-Beltrami operator; open mesh boundaries have zero flux.
    """

    def __init__(self, mesh):
        super().__init__()
        self._mesh = mesh
        self._nbr = None
        self._wts = None

    @property
    def mesh(self):
        return self._mesh

    @property
    def spatial_shape(self):
        return (self._mesh.n_vertices,)

    def is_compatible(self, other):
        return (super().is_compatible(other)
                and (other.mesh is self._mesh
                     or other.mesh.hash == self._mesh.hash))

    def bind(self, am, dtype, batch_size):
        super().bind(am, dtype, batch_size)
        nbr, wts = self._mesh.ell()
        self._nbr = am.array(nbr, dtype=am.index_dtype)
        self._wts = am.array(wts, dtype=dtype)

    # ---- array-module path ----

    def reaction_view(self, a):
        # Reactions broadcast their parameters as (B, 1, 1).
        return a.reshape(a.shape[0], 1, a.shape[1])

    def laplacian(self, a):
        """Laplace-Beltrami operator of (B, N), summed over the ELL slots in
        the order of the CUDA kernel."""
        nbr, wts = self._nbr, self._wts  # (K, N)
        lap = wts[0] * (a[:, nbr[0]] - a)
        for k in range(1, nbr.shape[0]):
            lap = lap + wts[k] * (a[:, nbr[k]] - a)
        return self.reaction_view(lap)

    def assemble(self, dydt, du, dv):
        am = self._am
        dydt = am.set(dydt, (0, ...), du.reshape(dydt.shape[1:]))
        dydt = am.set(dydt, (1, ...), dv.reshape(dydt.shape[1:]))
        return dydt

    # ---- solvers ----

    warns_unstable_dt = True

    def lambda_max(self):
        return self._mesh.lambda_max()

    def _make_kernel_op(self, km):
        import cupy as cp
        from lpf.kernels.operators import SurfaceKernelOp
        nbr, wts = self._mesh.ell()
        return SurfaceKernelOp(km, self._batch_size, self._mesh.n_vertices,
                               cp.asarray(nbr, dtype=cp.int32),
                               cp.asarray(wts, dtype=self._dtype))

    def jax_rhs(self, model):
        from lpf.solvers._jax.rhs import build_mesh_rhs
        return (build_mesh_rhs(model), ("mesh", type(model).__name__),
                (model.params, self._nbr, self._wts))

    # ---- output and serialization ----

    def save_color(self, model, index, arr_color, fpath_morph, fpath_pattern):
        # A surface has no 2D picture: both outputs are written as the
        # colored mesh (.ply, replacing the given extension).
        if arr_color is None:
            arr_color = model.colorize()
        for fpath in (fpath_morph, fpath_pattern):
            if fpath:
                self._mesh.save_ply(osp.splitext(fpath)[0] + ".ply",
                                    colors=arr_color[index])

    def to_dict(self, index=0):
        return {"mesh": self._mesh.to_dict()}

    def save_assets(self, model_dict, dpath):
        # The mesh is stored once next to the model files and referred to.
        fname = "mesh_%s.npz" % model_dict["mesh"]["hash"][:16]
        fpath = osp.join(dpath, fname)
        if not osp.isfile(fpath):
            self._mesh.save(fpath)
        model_dict["mesh"]["file"] = fname

    @classmethod
    def kwargs_from_dicts(cls, model_dicts, dpath="."):
        meshes = [n2v.get("mesh") for n2v in model_dicts]
        if all(m is None for m in meshes):
            return None
        if any(m is None or m.get("hash") != meshes[0].get("hash")
               for m in meshes):
            raise ValueError("All models of a batch must share one mesh.")
        from lpf.surfaces import SurfaceMesh
        return {"mesh": SurfaceMesh.from_dict(meshes[0], dpath)}
