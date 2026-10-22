"""Tests for reaction-diffusion on triangle-mesh surfaces (lpf.surfaces).

Covers:
  - SurfaceMesh: Laplace-Beltrami eigenvalues of the sphere, area, quad
    splitting, polygon lists, open / non-manifold / mixed-polygon meshes,
    flat grid = 5-point stencil, stability bounds, ELL layout, vertex
    reordering, save/load, hash, PLY output, errors
  - Mesh models through the LPF API (LiawModel(mesh=...), LiawInitializer
    with vertex points, lpf.solvers): conservation, every solver, vertex
    relabeling, early stopping, file output and parse_geometry, diploid
  - CUDA: CPU parity for every solver, native loop vs Python loop, JIT vs AOT
  - Other backends (PyTorch, JAX) when installed
"""

import json
import os

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import lpf.solvers as S
from lpf.initializers import LiawInitializer
from lpf.models import LiawModel, GrayScottModel, TwoComponentDiploidModel
from lpf.surfaces import SurfaceMesh


# ---------- helpers ----------

def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t),
             (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1),
             (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v, dtype=float) / np.linalg.norm(v) for v in verts]
    for _ in range(level):
        cache, new_faces = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.array(verts), np.array(faces)


def grid_quads(n, size=1.0):
    """Open n x n quad grid on [0, size]^2."""
    xs = np.linspace(0, size, n + 1)
    xx, yy = np.meshgrid(xs, xs)
    verts = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]],
                     axis=-1).reshape(-1, 4)
    return verts, quads


LIAW_KEYS = ["Du", "Dv", "ru", "rv", "k", "su", "sv", "mu"]
LIAW_BASE = dict(zip(LIAW_KEYS, [0.0005, 0.075, 0.18, 0.0287, 0.2, 0.001,
                                 0.025, 0.08]), u0=2.0, v0=1.0)


def _liaw_dicts(mesh, batch_size=2, n_pts=20, **overrides):
    dicts = []
    for b in range(batch_size):
        n2v = dict(LIAW_BASE, k=0.2 - 0.06 * b)
        n2v.update(overrides)
        for i, vid in enumerate(mesh.farthest_points(n_pts, start=b)):
            n2v["init_pts_%d" % i] = [int(vid)]
        dicts.append(n2v)
    return dicts


def _make_model(mesh, device="cpu", dtype=np.float32, model_dicts=None):
    if model_dicts is None:
        model_dicts = _liaw_dicts(mesh)
    initializer = LiawInitializer()
    initializer.update(model_dicts)
    return LiawModel(initializer=initializer,
                     params=LiawModel.parse_params(model_dicts),
                     mesh=mesh, device=device, dtype=dtype)


def _get_numpy(arr):
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def _solve(model, solver_name, n_iters=300, dt=0.01, **kwargs):
    return getattr(S, solver_name)().solve(model=model, dt=dt,
                                           n_iters=n_iters, **kwargs)


FIXED_SOLVERS = ["EulerSolver", "HeunSolver", "RungeKuttaSolver",
                 "RK23Solver", "AdamsBashforth2Solver"]
ALL_SOLVERS = FIXED_SOLVERS + ["DOPRI5Solver", "AdaptiveRKF45Solver"]


@pytest.fixture(scope="module")
def sphere_mesh():
    v, f = icosphere(3)
    return SurfaceMesh(v, f, scale=4.0)


def _need_cuda():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device")


class _NativeOff:
    """Stands in for the native library to force the Python loop."""

    def is_available(self):
        return False


# ---------- SurfaceMesh ----------

class TestMesh:

    def test_sphere_eigenvalues(self):
        """Laplace-Beltrami on the unit sphere: l(l+1), multiplicity 2l+1."""
        v, f = icosphere(4)
        mesh = SurfaceMesh(v, f)
        op = mesh.operator
        neg_lap = sp.diags(np.asarray(op.sum(axis=1)).ravel()) - op
        lam = sla.eigs(neg_lap, k=16, sigma=-1e-3, which="LM",
                       return_eigenvectors=False)
        lam = np.sort(lam.real)
        exact = np.array([0] + [2] * 3 + [6] * 5 + [12] * 7, dtype=float)
        assert abs(lam[0]) < 1e-8
        np.testing.assert_allclose(lam[1:], exact[1:], rtol=0.01)

    def test_area(self):
        v, f = icosphere(4)
        assert SurfaceMesh(v, f).area == pytest.approx(4 * np.pi, rel=2e-3)
        assert SurfaceMesh(v, f, scale=3.0).area == pytest.approx(
            36 * np.pi, rel=2e-3)

    def test_quads_split_along_shorter_diagonal(self):
        # Rhombus: diagonal 1-3 is much shorter than 0-2.
        v = np.array([(0, 0, 0), (2, 0.5, 0), (4, 0, 0), (2, -0.5, 0)], float)
        mesh = SurfaceMesh(v, [(0, 1, 2, 3)])
        edges = {tuple(sorted(e)) for t in mesh.triangles
                 for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
        assert (1, 3) in edges and (0, 2) not in edges
        assert mesh.n_negative_weights == 0

    def test_polygon_lists_and_quads_agree(self):
        v, quads = grid_quads(4)
        a = SurfaceMesh(v, quads)
        b = SurfaceMesh(v, [list(q) for q in quads])
        assert (a.operator != b.operator).nnz == 0

    def test_open_mesh_keeps_constants(self):
        v, quads = grid_quads(8)
        mesh = SurfaceMesh(v, quads)
        np.testing.assert_allclose(mesh.laplacian(np.full((2, len(v)), 3.0)),
                                   0.0, atol=1e-12)

    def test_mixed_nonmanifold_mesh(self):
        """Triangles, a quad and a pentagon; an edge shared by three faces;
        an open boundary. The operator stays conservative and consistent."""
        v = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                      (2, 0.5, 0), (0.5, -1, 0), (0.5, 0.5, 1),
                      (1.6, 1.4, 0.2), (2.4, 1.2, 0.1)], float)
        faces = [(0, 1, 2, 3),          # quad
                 (1, 4, 2),             # triangle
                 (0, 5, 1),             # triangle
                 (0, 1, 6),             # third face on edge 0-1
                 (2, 4, 8, 7, 3)]       # pentagon
        mesh = SurfaceMesh(v, faces)
        weights = sp.diags(mesh.mass) @ mesh.operator   # w_ij, symmetric
        np.testing.assert_allclose(weights.toarray(), weights.T.toarray(),
                                   atol=1e-12)
        np.testing.assert_allclose(mesh.laplacian(np.ones(len(v))), 0.0,
                                   atol=1e-12)

    def test_flat_grid_matches_five_point_stencil(self):
        """Interior of a flat grid split into right triangles: the cotangent
        Laplacian is the 5-point stencil."""
        n, h = 10, 0.1
        v, quads = grid_quads(n, size=n * h)
        mesh = SurfaceMesh(v, quads)
        u = np.sin(v[:, 0] * 3) * np.cos(v[:, 1] * 2)
        lap = mesh.laplacian(u)
        idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
        ui = u[idx]
        five = (ui[:-2, 1:-1] + ui[2:, 1:-1] + ui[1:-1, :-2] + ui[1:-1, 2:]
                - 4 * ui[1:-1, 1:-1]) / h ** 2
        np.testing.assert_allclose(lap[idx[1:-1, 1:-1]], five, rtol=1e-9)

    def test_exact_stability_bound_is_sharper(self, sphere_mesh):
        gershgorin = sphere_mesh.max_stable_dt(0.075)
        exact = sphere_mesh.max_stable_dt(0.075, exact=True)
        assert exact >= gershgorin * (1 - 1e-9)

    def test_ell_matches_csr(self, sphere_mesh):
        nbr, wts = sphere_mesh.ell()
        u = np.random.default_rng(0).random(sphere_mesh.n_vertices)
        ell = (wts * (u[nbr] - u)).sum(axis=0)
        np.testing.assert_allclose(ell, sphere_mesh.laplacian(u), atol=1e-10)

    def test_reorder_is_a_relabeling(self):
        v, f = icosphere(3)
        perm = np.random.default_rng(5).permutation(len(v))
        inv = np.argsort(perm)
        shuffled = SurfaceMesh(v[perm], inv[f], scale=2.0)
        ordered = SurfaceMesh(v[perm], inv[f], scale=2.0, reorder=True)
        u = np.random.default_rng(1).random(len(v))  # in shuffled order
        lap_ordered = ordered.to_original(
            ordered.laplacian(ordered.from_original(u)))
        np.testing.assert_allclose(lap_ordered, shuffled.laplacian(u),
                                   atol=1e-10)

        def bandwidth(mesh):
            t = mesh.triangles
            return np.abs(t[:, :, None] - t[:, None, :]).max()

        assert bandwidth(ordered) < bandwidth(shuffled)

    def test_save_load_roundtrip(self, sphere_mesh, tmp_path):
        fpath = tmp_path / "mesh.npz"
        sphere_mesh.save(fpath)
        loaded = SurfaceMesh.load(fpath)
        assert (loaded.operator != sphere_mesh.operator).nnz == 0
        assert loaded.hash == sphere_mesh.hash

        v, f = icosphere(2)
        reordered = SurfaceMesh(v, f, reorder=True)
        reordered.save(tmp_path / "r.npz")
        np.testing.assert_array_equal(
            SurfaceMesh.load(tmp_path / "r.npz").order, reordered.order)

    def test_ply_output(self, sphere_mesh, tmp_path):
        colors = np.random.default_rng(0).integers(
            0, 255, (sphere_mesh.n_vertices, 3)).astype(np.uint8)
        fpath = tmp_path / "m.ply"
        sphere_mesh.save_ply(fpath, colors=colors)
        header, body = fpath.read_bytes().split(b"end_header\n", 1)
        n_v, n_f = sphere_mesh.n_vertices, len(sphere_mesh.triangles)
        assert b"element vertex %d" % n_v in header
        assert b"element face %d" % n_f in header
        assert len(body) == n_v * (12 + 3) + n_f * (1 + 12)
        vert = np.frombuffer(body[:n_v * 15],
                             dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        np.testing.assert_array_equal(vert["c"], colors)

    def test_invalid_input(self):
        v, f = icosphere(1)
        with pytest.raises(ValueError):  # unreferenced vertex
            SurfaceMesh(np.vstack([v, [[5, 5, 5]]]), f)
        with pytest.raises(ValueError):
            SurfaceMesh(v, f + len(v))
        with pytest.raises(ValueError):
            SurfaceMesh(v[:, :2], f)
        with pytest.raises(ValueError):
            LiawModel(params=np.zeros((1, 8)), mesh=SurfaceMesh(v, f),
                      domain=object())


# ---------- mesh models through the LPF API (CPU) ----------

class TestModelCPU:

    def test_state_shape_and_initializer(self, sphere_mesh):
        model = _make_model(sphere_mesh)
        model.initialize()
        y = _get_numpy(model.y_mesh)
        assert y.shape == (2, 2, sphere_mesh.n_vertices)
        seeds = sphere_mesh.farthest_points(20, start=0)
        assert (y[0, 0, seeds] == 2.0).all()
        assert y[0, 0].sum() == pytest.approx(2.0 * 20)
        assert (y[1] == 1.0).all()
        n2v = model.to_dict(index=0, initializer=model.initializer)
        assert n2v["init_pts_0"] == [int(seeds[0])]
        assert n2v["mesh"]["hash"] == sphere_mesh.hash

    def test_pure_diffusion_conserves_mass(self, sphere_mesh):
        dicts = _liaw_dicts(sphere_mesh, batch_size=1, Du=0.05, ru=0.0,
                            rv=0.0, su=0.0, sv=0.0, mu=0.0)
        model = _make_model(sphere_mesh, dtype=np.float64, model_dicts=dicts)
        model.initialize()
        u0 = _get_numpy(model.u)[0]
        S.EulerSolver().solve(model=model, dt=0.01, n_iters=2000,
                              init_model=False)
        u = _get_numpy(model.u)[0]
        assert sphere_mesh.mass @ u == pytest.approx(sphere_mesh.mass @ u0,
                                                     rel=1e-12)
        assert u.std() < 0.1 * u0.std()  # it did spread

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_every_solver(self, sphere_mesh, solver_name):
        model = _make_model(sphere_mesh, dtype=np.float64)
        out = _solve(model, solver_name, n_iters=200,
                     trj_waypoints=[1, 100, 200])
        trj = _get_numpy(out["trj"])
        assert trj.shape == (3, 2, 2, sphere_mesh.n_vertices)
        assert np.isfinite(trj).all()
        np.testing.assert_array_equal(trj[-1], _get_numpy(model.y_mesh))

    def test_relabeled_mesh_gives_same_field(self):
        """Renumbering the mesh's vertices only renumbers the solution, with
        or without reordering."""
        v, f = icosphere(3)
        perm = np.random.default_rng(3).permutation(len(v))  # new k = old perm[k]
        inv = np.argsort(perm)                               # old i -> new inv[i]
        seeds = np.arange(0, len(v), 37)

        def run(mesh, seed_ids):
            n2v = dict(LIAW_BASE)
            for i, vid in enumerate(seed_ids):
                n2v["init_pts_%d" % i] = [int(vid)]
            model = _make_model(mesh, dtype=np.float64, model_dicts=[n2v])
            S.EulerSolver().solve(model=model, dt=0.01, n_iters=300)
            return _get_numpy(model.y_mesh)

        y = run(SurfaceMesh(v, f, scale=4.0), seeds)
        y_shuffled = run(SurfaceMesh(v[perm], inv[f], scale=4.0), inv[seeds])
        np.testing.assert_allclose(y_shuffled[..., inv], y,
                                   rtol=1e-10, atol=1e-12)

        rcm = SurfaceMesh(v[perm], inv[f], scale=4.0, reorder=True)
        rcm_index = np.argsort(rcm.order)  # shuffled index -> rcm index
        y_rcm = rcm.to_original(run(rcm, rcm_index[inv[seeds]]))
        np.testing.assert_allclose(y_rcm[..., inv], y, rtol=1e-10, atol=1e-12)

    def test_early_stopping(self, sphere_mesh):
        dicts = _liaw_dicts(sphere_mesh, batch_size=1, Du=0.2, Dv=0.2, ru=0.0,
                            rv=0.0, su=0.0, sv=0.0, mu=0.0)
        model = _make_model(sphere_mesh, dtype=np.float64, model_dicts=dicts)
        trj = S.EulerSolver().solve(model=model, dt=0.01, n_iters=100000,
                                    rtol=1e-6, period_output=1000, get_trj=True)
        assert 1 < _get_numpy(trj).shape[0] < 101  # 101 if it ran to the end
        u = _get_numpy(model.u)[0]
        assert u.std() < 0.05 * u.mean()  # near the uniform steady state

    def test_file_output_and_reload(self, sphere_mesh, tmp_path):
        model = _make_model(sphere_mesh)
        S.EulerSolver().solve(model=model, dt=0.01, n_iters=200,
                              period_output=100, dpath_model=str(tmp_path),
                              dpath_morph=str(tmp_path),
                              dpath_states=str(tmp_path))
        files = sorted(os.listdir(tmp_path / "model_1"))
        assert files == ["morph_001.ply", "morph_100.ply", "morph_200.ply",
                         "states_001.npz", "states_100.npz", "states_200.npz"]
        states = np.load(tmp_path / "model_1" / "states_200.npz")
        np.testing.assert_array_equal(states["u"], _get_numpy(model.u)[0])

        dpath_models = tmp_path / "models"
        dicts = [json.load(open(dpath_models / f))
                 for f in sorted(os.listdir(dpath_models)) if f.endswith(".json")]
        geometry = LiawModel.parse_geometry(dicts, dpath_models)
        assert geometry["mesh"].hash == sphere_mesh.hash
        initializer = LiawInitializer()
        initializer.update(dicts)
        reloaded = LiawModel(initializer=initializer,
                             params=LiawModel.parse_params(dicts), **geometry)
        S.EulerSolver().solve(model=reloaded, dt=0.01, n_iters=200)
        np.testing.assert_array_equal(_get_numpy(reloaded.y_mesh),
                                      _get_numpy(model.y_mesh))

    def test_colorize(self, sphere_mesh):
        model = _make_model(sphere_mesh)
        S.EulerSolver().solve(model=model, dt=0.01, n_iters=10)
        color = model.colorize()
        assert color.shape == (2, sphere_mesh.n_vertices, 3)
        high = _get_numpy(model.u) > 0.5
        assert (color[high] == (5, 5, 5)).all()
        assert (color[~high] == (231, 79, 3)).all()

    def test_diploid(self, sphere_mesh):
        dicts = _liaw_dicts(sphere_mesh, batch_size=1)
        pa = _make_model(sphere_mesh, model_dicts=dicts)
        ma = _make_model(sphere_mesh, model_dicts=[dict(dicts[0], k=0.1)])
        d = TwoComponentDiploidModel(
            paternal_model=pa, maternal_model=ma, alpha=0.5, beta=0.5,
            params=pa.params, initializer=pa.initializer)
        S.EulerSolver().solve(model=d, dt=0.01, n_iters=100)
        y = _get_numpy(d.y_mesh)
        assert y.shape == (4, 1, sphere_mesh.n_vertices)
        assert np.isfinite(y).all()
        assert d.colorize().shape == (1, sphere_mesh.n_vertices, 3)

    def test_other_model(self, sphere_mesh):
        params = np.array([[0.02, 0.01, 0.04, 0.06]], dtype=np.float32)
        model = GrayScottModel(params=params, mesh=sphere_mesh,
                               initializer=_GrayScottSeed(sphere_mesh))
        S.EulerSolver().solve(model=model, dt=0.1, n_iters=200)
        assert np.isfinite(_get_numpy(model.y_mesh)).all()


class _GrayScottSeed:
    """u = 1, v = 0.5 at a few vertices (minimal initializer for a test)."""

    def __init__(self, mesh):
        self._seeds = mesh.farthest_points(10)

    def initialize(self, model):
        y = np.zeros(model.shape_grid)
        y[0] = 1.0
        y[1][:, self._seeds] = 0.5
        model._y_mesh = model.am.array(y, dtype=model.dtype)


# ---------- CUDA ----------

class TestCUDA:

    @pytest.fixture(autouse=True)
    def _cuda(self):
        _need_cuda()

    @pytest.mark.parametrize("solver_name", FIXED_SOLVERS + ["DOPRI5Solver"])
    def test_gpu_matches_cpu(self, sphere_mesh, solver_name):
        out = []
        for device in ("cpu", "cuda:0"):
            model = _make_model(sphere_mesh, device=device)
            _solve(model, solver_name, n_iters=1000)
            out.append(_get_numpy(model.y_mesh))
        np.testing.assert_allclose(out[1], out[0], rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("solver_name", FIXED_SOLVERS)
    def test_native_matches_python_loop(self, sphere_mesh, solver_name,
                                        monkeypatch):
        import lpf.solvers._cuda.base as cuda_base
        if not cuda_base._get_native().supports("surface"):
            pytest.skip("native surface loop not built")
        out = []
        for native in (True, False):
            if not native:
                monkeypatch.setattr(cuda_base, "_native_solver", _NativeOff())
            model = _make_model(sphere_mesh, device="cuda:0")
            res = _solve(model, solver_name, n_iters=777,
                         trj_waypoints=[1, 300, 777])
            out.append((_get_numpy(res["trj"]), _get_numpy(model.y_mesh)))
        # Same kernels in the same order; the AOT utility kernels (fmaf) may
        # differ from the native loop's in the last bit.
        np.testing.assert_allclose(out[0][0], out[1][0], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(out[0][1], out[1][1], rtol=1e-5, atol=1e-7)

    def test_euler_jit_aot_native_bit_identical(self, sphere_mesh,
                                                monkeypatch):
        import lpf.kernels.compiler as compiler
        import lpf.solvers._cuda.base as cuda_base
        from lpf.kernels.aot.loader import AOTKernelLoader
        out = {}

        def run(label):
            compiler._KERNEL_CACHE.clear()
            model = _make_model(sphere_mesh, device="cuda:0")
            _solve(model, "EulerSolver", n_iters=500)
            out[label] = _get_numpy(model.y_mesh)

        if cuda_base._get_native().supports("surface"):
            run("native")
        monkeypatch.setattr(cuda_base, "_native_solver", _NativeOff())
        if AOTKernelLoader().is_available():
            run("aot")
        monkeypatch.setattr(AOTKernelLoader, "is_available", lambda self: False)
        run("jit")
        compiler._KERNEL_CACHE.clear()
        for label, y in out.items():
            np.testing.assert_array_equal(y, out["jit"], err_msg=label)

    def test_early_stopping_matches_cpu(self, sphere_mesh):
        n_out = []
        for device in ("cpu", "cuda:0"):
            dicts = _liaw_dicts(sphere_mesh, batch_size=1, Du=0.2, Dv=0.2,
                                ru=0.0, rv=0.0, su=0.0, sv=0.0, mu=0.0)
            model = _make_model(sphere_mesh, device=device, model_dicts=dicts)
            trj = S.EulerSolver().solve(model=model, dt=0.01, n_iters=100000,
                                        rtol=1e-5, period_output=1000,
                                        get_trj=True)
            n_out.append(_get_numpy(trj).shape[0])
        assert n_out[0] == n_out[1] < 101


# ---------- other backends ----------

def _backend_devices():
    devices = []
    try:
        import torch
        devices.append("torch:cpu")
        if torch.cuda.is_available():
            devices.append("torch:cuda:0")
    except ImportError:
        pass
    try:
        import jax
        devices.append("jax:cpu")
        if any(d.platform == "gpu" for d in jax.devices()):
            devices.append("jax:gpu:0")
    except Exception:
        pass
    return devices or [pytest.param("none", marks=pytest.mark.skip(
        reason="neither PyTorch nor JAX installed"))]


@pytest.mark.parametrize("device", _backend_devices())
@pytest.mark.parametrize("solver_name", ["EulerSolver", "RungeKuttaSolver",
                                         "AdamsBashforth2Solver",
                                         "DOPRI5Solver"])
def test_backend_matches_numpy(sphere_mesh, device, solver_name):
    out = []
    for dev in ("cpu", device):
        model = _make_model(sphere_mesh, device=dev)
        res = _solve(model, solver_name, n_iters=300, trj_waypoints=[1, 300])
        out.append(_get_numpy(res["trj"]))
    np.testing.assert_allclose(out[1], out[0], rtol=1e-5, atol=1e-6)


def test_unstable_dt_warns(sphere_mesh):
    model = _make_model(sphere_mesh)
    euler = S.EulerSolver().max_stable_dt(model)
    rk4 = S.RungeKuttaSolver().max_stable_dt(model)
    assert S.AdamsBashforth2Solver().max_stable_dt(model) < euler < rk4
    assert euler == pytest.approx(sphere_mesh.max_stable_dt(0.075, exact=True))
    dt = 0.5 * (euler + rk4)
    with pytest.warns(UserWarning, match="stability limit"):
        S.EulerSolver().solve(model=model, dt=dt, n_iters=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        S.RungeKuttaSolver().solve(model=model, dt=dt, n_iters=1)
