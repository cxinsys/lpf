"""Tests for the spatial discretizations (lpf.geometry) and the fused CUDA
kernels they launch (lpf.kernels.operators, csrc/include/rd_kernels.cuh).

Covers:
  - make_geometry / geometry_kwargs and the geometry of each model
  - lambda_max: exact on the grid, a bound on domains; max_stable_dt and
    which geometries warn about an unstable dt
  - diploid models share the paternal geometry (domain colorize)
  - init_pts must match the dimensionality of the space
  - CUDA: the native loop and the Python loop with AOT and JIT kernels give
    bit-identical results for every solver on the grid, a domain and a mesh
    (fused stages, domain cell flags, four batch elements per thread)
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

import lpf.solvers as S
from lpf.domains import PolygonDomain
from lpf.geometry import (DomainGeometry, GridGeometry, MeshGeometry,
                          geometry_kwargs, make_geometry)
from lpf.initializers import LiawInitializer
from lpf.models import LiawModel, TwoComponentDiploidModel
from lpf.surfaces import SurfaceMesh

from tests.test_surfaces import icosphere


LIAW = {"u0": 2.0, "v0": 1.0, "Du": 0.0005, "Dv": 0.075, "ru": 0.18,
        "rv": 0.0287, "k": 0.2, "su": 0.001, "sv": 0.025, "mu": 0.08}
STAR = np.array([(16 + 14 * np.sin(a) * (1 if i % 2 == 0 else 0.45),
                  16 - 14 * np.cos(a) * (1 if i % 2 == 0 else 0.45))
                 for i, a in enumerate(np.arange(10) * np.pi / 5)])
DISK = np.stack([16 + 12 * np.cos(np.linspace(0, 2 * np.pi, 80, endpoint=False)),
                 15 + 10 * np.sin(np.linspace(0, 2 * np.pi, 80, endpoint=False))], 1)
SOLVERS = ["EulerSolver", "HeunSolver", "RungeKuttaSolver", "RK23Solver",
           "AdamsBashforth2Solver"]


def _mesh():
    v, f = icosphere(3)
    return SurfaceMesh(v, f, scale=1.0, reorder=True)


def _points(geometry, i, n=6):
    rng = np.random.default_rng(i)
    if geometry == "mesh":
        return [[int(x)] for x in rng.choice(642, n, replace=False)]
    if geometry in ("domain", "domains"):
        dom = PolygonDomain(STAR if geometry == "domain" or i % 2 == 0 else DISK)
        return [[int(r), int(c)] for r, c in dom.random_points(n, 32, 32, rng=rng)]
    return [[int(x) for x in p] for p in rng.integers(1, 31, size=(n, 2))]


def _model(geometry, device="cpu", batch=3, dtype=np.float32, mesh=None, **over):
    dicts = []
    for i in range(batch):
        d = dict(LIAW, **over)
        d["Du"] *= 1 + 0.1 * i
        if geometry == "mesh":
            d["Du"], d["Dv"] = 0.002, 0.05
        for j, p in enumerate(_points(geometry, i)):
            d["init_pts_%d" % j] = p
        dicts.append(d)
    kwargs = {"grid": {},
              "domain": {"domain": PolygonDomain(STAR)},
              "domains": {"domain": [PolygonDomain(STAR if i % 2 == 0 else DISK)
                                     for i in range(batch)]},
              "mesh": {"mesh": mesh if mesh is not None else _mesh()}}[geometry]
    init = LiawInitializer(dtype=dtype)
    init.update(dicts)
    return LiawModel(initializer=init, params=LiawModel.parse_params(dicts, dtype=dtype),
                     width=32, height=32, dx=0.1, device=device, dtype=dtype,
                     **kwargs)


def _numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


# ---------- geometry objects ----------

class TestGeometry:

    def test_make_geometry(self):
        assert type(make_geometry(8, 9, 0.1)) is GridGeometry
        dom = PolygonDomain(STAR)
        g = make_geometry(32, 32, 0.1, domain=dom)
        assert type(g) is DomainGeometry and g.domain is dom
        assert g.spatial_shape == (32, 32)
        m = _mesh()
        g = make_geometry(32, 32, 0.1, mesh=m)
        assert type(g) is MeshGeometry and g.mesh is m
        assert g.spatial_shape == (642,)
        with pytest.raises(ValueError):
            make_geometry(32, 32, 0.1, domain=dom, mesh=m)

    def test_model_delegates_to_geometry(self):
        for geometry, cls in (("grid", GridGeometry), ("domain", DomainGeometry),
                              ("mesh", MeshGeometry)):
            model = _model(geometry)
            assert type(model.geometry) is cls
            assert model.spatial_shape == model.geometry.spatial_shape
            assert (model.domain is None) == (geometry != "domain")
            assert (model.mesh is None) == (geometry != "mesh")

    def test_geometry_kwargs(self):
        assert geometry_kwargs([{}, {}]) == {}
        dom = PolygonDomain(STAR)
        kw = geometry_kwargs([{"domain": dom.to_dict()}] * 2)
        assert type(kw["domain"]) is PolygonDomain
        with pytest.raises(ValueError):
            geometry_kwargs([{"domain": dom.to_dict()}, {}])

    def test_grid_lambda_max_is_exact(self):
        H, W, dx = 7, 9, 0.1
        lap1 = [sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n - 2, n - 2))
                for n in (H, W)]
        lap = (sp.kron(lap1[0], sp.eye(W - 2)) + sp.kron(sp.eye(H - 2), lap1[1])) / dx ** 2
        exact = np.max(np.linalg.eigvalsh(-lap.toarray()))
        assert GridGeometry(H, W, dx).lambda_max() == pytest.approx(exact, rel=1e-12)

    def test_domain_lambda_max_bounds(self):
        g = DomainGeometry([PolygonDomain(STAR), PolygonDomain(DISK)], 32, 32, 0.1)
        assert g.lambda_max() <= 8 / 0.1 ** 2 + 1e-9
        assert g.lambda_max() >= GridGeometry(32, 32, 0.1).lambda_max() * 0.9

    def test_is_compatible(self):
        dom = PolygonDomain(STAR)
        a = DomainGeometry(dom, 32, 32, 0.1)
        assert a.is_compatible(DomainGeometry(PolygonDomain(STAR), 32, 32, 0.1))
        assert not a.is_compatible(DomainGeometry(PolygonDomain(DISK), 32, 32, 0.1))
        assert not a.is_compatible(GridGeometry(32, 32, 0.1))
        assert not GridGeometry(32, 32, 0.1).is_compatible(GridGeometry(16, 32, 0.1))


# ---------- solvers ----------

class TestStability:

    def test_max_stable_dt_on_every_geometry(self):
        for geometry in ("grid", "domain", "mesh"):
            model = _model(geometry)
            d_max = model.max_diffusivity()
            lam = model.geometry.lambda_max()
            assert S.EulerSolver().max_stable_dt(model) == pytest.approx(2.0 / (d_max * lam))
            assert S.AdaptiveRKF45Solver().max_stable_dt(model) is None

    def test_only_meshes_warn(self):
        grid = _model("grid", Dv=0.5)
        limit = S.EulerSolver().max_stable_dt(grid)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            S.EulerSolver().solve(model=grid, dt=1.5 * limit, n_iters=1)
        mesh = _model("mesh")
        limit = S.EulerSolver().max_stable_dt(mesh)
        with pytest.warns(UserWarning, match="stability limit"):
            S.EulerSolver().solve(model=mesh, dt=1.5 * limit, n_iters=1)


class TestModels:

    def test_diploid_shares_the_paternal_domain(self):
        pa = _model("domain", batch=2)
        ma = _model("domain", batch=2, ru=0.2)
        dip = TwoComponentDiploidModel(paternal_model=pa, maternal_model=ma)
        assert dip.geometry is pa.geometry
        S.EulerSolver().solve(model=dip, dt=0.01, n_iters=5)
        color = dip.colorize()
        assert color.shape == (2, 32, 32, 3)
        outside = ~PolygonDomain(STAR).mask(32, 32)
        assert (color[:, outside] == 255).all()

    def test_diploid_rejects_different_spaces(self):
        with pytest.raises(ValueError):
            TwoComponentDiploidModel(paternal_model=_model("domain", batch=2),
                                     maternal_model=_model("grid", batch=2))

    def test_init_pts_must_match_the_space(self):
        model = _model("grid")
        dicts = [dict(LIAW, init_pts_0=[3]) for _ in range(3)]
        model.initializer.update(dicts)
        with pytest.raises(ValueError, match="coordinates"):
            model.initialize()


# ---------- fused CUDA kernels ----------

def _need_cuda():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device")


@pytest.mark.parametrize("geometry", ["grid", "domains", "mesh"])
@pytest.mark.parametrize("solver_name", SOLVERS)
def test_native_aot_jit_bit_identical(geometry, solver_name, monkeypatch):
    """The native loop and the Python loop (AOT and JIT kernels) run the
    same fused kernels, so every solver agrees bit for bit."""
    _need_cuda()
    import lpf.kernels.compiler as compiler
    import lpf.solvers._cuda.base as cuda_base
    from lpf.kernels.aot.loader import AOTKernelLoader

    mesh = _mesh()
    out = {}

    def run(label):
        compiler._KERNEL_CACHE.clear()
        model = _model(geometry, device="cuda:0", batch=5, mesh=mesh)
        trj = getattr(S, solver_name)().solve(model=model, dt=0.01, n_iters=60,
                                              trj_waypoints=[1, 30, 60])
        out[label] = (_numpy(trj["trj"]), _numpy(model.y_mesh))

    native = cuda_base._get_native()
    if native.supports(model_kind := {"grid": "grid", "domains": "domain",
                                      "mesh": "surface"}[geometry]):
        run("native")
    monkeypatch.setattr(cuda_base, "_native_solver",
                        SimpleNamespace(is_available=lambda: False))
    if AOTKernelLoader().is_available():
        run("aot")
    monkeypatch.setattr(AOTKernelLoader, "is_available", lambda self: False)
    run("jit")
    compiler._KERNEL_CACHE.clear()
    assert len(out) >= 2 or model_kind is None
    for label, (trj, y) in out.items():
        np.testing.assert_array_equal(trj, out["jit"][0], err_msg=label)
        np.testing.assert_array_equal(y, out["jit"][1], err_msg=label)


@pytest.mark.parametrize("geometry", ["grid", "domain", "mesh"])
@pytest.mark.parametrize("solver_name", SOLVERS)
def test_gpu_matches_cpu(geometry, solver_name):
    """The fused kernels reproduce the NumPy path to a few ULP (NumPy divides
    by dx^2 where the kernels multiply by 1/dx^2, and the CPU RK4/RK23/AB2
    group their stage sums differently), and exactly for Euler and Heun on a
    mesh."""
    _need_cuda()
    mesh = _mesh()
    ys = []
    for device in ("cpu", "cuda:0"):
        model = _model(geometry, device=device, batch=5, mesh=mesh)
        getattr(S, solver_name)().solve(model=model, dt=0.01, n_iters=40)
        ys.append(_numpy(model.y_mesh))
    if geometry == "mesh" and solver_name in ("EulerSolver", "HeunSolver"):
        np.testing.assert_array_equal(ys[0], ys[1])
    else:
        np.testing.assert_allclose(ys[0], ys[1], rtol=1e-5, atol=1e-7)
