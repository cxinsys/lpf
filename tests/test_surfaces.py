"""Tests for reaction-diffusion on triangle-mesh surfaces (lpf.surfaces).

Covers:
  - Mesh operator: Laplace-Beltrami eigenvalues of the sphere, area, quad
    splitting, open meshes (zero-flux boundary), stability bounds, errors
  - CPU solver: conservation under pure diffusion, seeded initial state,
    trajectories
  - CUDA: fused kernel matches the CPU path (Liaw, Gray-Scott), CUDA-graph
    replay is bit-identical to plain launches, CuPy fallback path
  - All 5 solvers: CPU vs GPU, native loop vs Python loop, JIT vs AOT,
    early stopping, per-solver stability bound, vertex reordering
"""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from lpf.models import LiawModel, GrayScottModel
from lpf.surfaces import (
    SurfaceMesh, SurfaceModel, SurfaceEulerSolver, SurfaceHeunSolver,
    SurfaceRungeKuttaSolver, SurfaceRK23Solver, SurfaceAdamsBashforth2Solver,
)
from lpf.surfaces.solver import _to_ell

ALL_SURFACE_SOLVERS = [SurfaceEulerSolver, SurfaceHeunSolver,
                       SurfaceRungeKuttaSolver, SurfaceRK23Solver,
                       SurfaceAdamsBashforth2Solver]


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


LIAW_PARAMS = np.array([
    [0.0005, 0.075, 0.18, 0.0287, 0.2, 0.001, 0.025, 0.08],
    [0.0005, 0.075, 0.18, 0.0287, 0.084, 0.001, 0.025, 0.08],
], dtype=np.float32)


def _liaw_surface(device, mesh, dtype=np.float32, params=LIAW_PARAMS):
    seeds = [mesh.farthest_points(20, start=b) for b in range(len(params))]
    y0 = SurfaceModel.seeded_state(mesh, [[2.0, 1.0]] * len(params), seeds)
    model = LiawModel(params=params, device=device, dtype=dtype)
    return SurfaceModel(model, mesh, y0)


def _get_numpy(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


@pytest.fixture(scope="module")
def sphere_mesh():
    v, f = icosphere(3)
    return SurfaceMesh(v, f, scale=4.0)


# ---------- mesh operator ----------

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

    def test_save_load_roundtrip(self, sphere_mesh, tmp_path):
        fpath = tmp_path / "mesh.npz"
        sphere_mesh.save(fpath)
        loaded = SurfaceMesh.load(fpath)
        assert (loaded.operator != sphere_mesh.operator).nnz == 0

    def test_invalid_input(self):
        v, f = icosphere(1)
        with pytest.raises(ValueError):  # unreferenced vertex
            SurfaceMesh(np.vstack([v, [[5, 5, 5]]]), f)
        with pytest.raises(ValueError):
            SurfaceMesh(v, f + len(v))
        with pytest.raises(ValueError):
            SurfaceMesh(v[:, :2], f)


# ---------- CPU solver ----------

class TestSolverCPU:

    def test_pure_diffusion_conserves_mass(self, sphere_mesh):
        params = LIAW_PARAMS.astype(np.float64).copy()
        params[:, 2:] = 0.0
        params[:, 0] = 0.05
        sm = _liaw_surface("cpu", sphere_mesh, dtype=np.float64, params=params)
        u0 = _get_numpy(sm.u)[0]
        mass0 = sphere_mesh.mass @ u0
        SurfaceEulerSolver().solve(sm, dt=0.01, n_iters=2000)
        u = _get_numpy(sm.u)[0]
        assert sphere_mesh.mass @ u == pytest.approx(mass0, rel=1e-12)
        assert u.std() < 0.1 * u0.std()  # it did spread

    def test_seeded_state(self, sphere_mesh):
        y0 = SurfaceModel.seeded_state(sphere_mesh, [[2.0, 1.0], [3.0, 0.5]],
                                       [[0, 5], [7]])
        assert y0.shape == (2, 2, sphere_mesh.n_vertices)
        assert y0[0, 0, 5] == 2.0 and y0[0, 1, 7] == 3.0
        assert y0[0, 0].sum() == 4.0 and (y0[1, 1] == 0.5).all()

    def test_trajectory(self, sphere_mesh):
        sm = _liaw_surface("cpu", sphere_mesh)
        res = SurfaceEulerSolver().solve(sm, dt=0.01, n_iters=300,
                                         trj_waypoints=[1, 150, 300, 999])
        assert res["iters"] == [1, 150, 300]
        trj = _get_numpy(res["trj"])
        assert trj.shape == (3, 2, 2, sphere_mesh.n_vertices)
        np.testing.assert_array_equal(trj[-1], _get_numpy(sm.y))
        assert np.isfinite(trj).all()

    def test_period_output_matches_grid_solvers(self, sphere_mesh):
        sm = _liaw_surface("cpu", sphere_mesh)
        res = SurfaceEulerSolver().solve(sm, dt=0.01, n_iters=250,
                                         period_output=100)
        assert res["iters"] == [1, 100, 200]

    def test_unstable_dt_warns(self, sphere_mesh):
        sm = _liaw_surface("cpu", sphere_mesh)
        dt = 2 * sphere_mesh.max_stable_dt(0.075, exact=True)
        with pytest.warns(UserWarning, match="stability"):
            SurfaceEulerSolver().solve(sm, dt=dt, n_iters=1)

    def test_bad_initial_shape(self, sphere_mesh):
        model = LiawModel(params=LIAW_PARAMS)
        with pytest.raises(ValueError):
            SurfaceModel(model, sphere_mesh, np.zeros((2, 1, 10)))


# ---------- CUDA ----------

class TestCUDA:

    @pytest.fixture(autouse=True)
    def _need_cupy(self):
        cupy = pytest.importorskip("cupy")
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")

    def test_fused_matches_cpu_liaw(self, sphere_mesh):
        results = []
        for device in ("cpu", "cuda:0"):
            sm = _liaw_surface(device, sphere_mesh)
            assert sm.has_fused_kernel == (device != "cpu")
            SurfaceEulerSolver().solve(sm, dt=0.01, n_iters=3000)
            results.append(_get_numpy(sm.y))
        np.testing.assert_allclose(results[1], results[0], rtol=1e-4, atol=1e-6)

    def test_fused_matches_cpu_grayscott(self, sphere_mesh):
        params = np.array([[0.02, 0.01, 0.04, 0.06]], dtype=np.float32)
        y0 = np.zeros((2, 1, sphere_mesh.n_vertices))
        y0[0] = 1.0
        y0[1, 0, sphere_mesh.farthest_points(10)] = 0.5
        results = []
        for device in ("cpu", "cuda:0"):
            model = GrayScottModel(params=params, device=device)
            sm = SurfaceModel(model, sphere_mesh, y0)
            SurfaceEulerSolver().solve(sm, dt=0.1, n_iters=1000)
            results.append(_get_numpy(sm.y))
        np.testing.assert_allclose(results[1], results[0], rtol=1e-4, atol=1e-6)

    def test_graph_replay_is_bit_identical(self, sphere_mesh):
        out = []
        for use_graph in (False, True):
            sm = _liaw_surface("cuda:0", sphere_mesh)
            res = SurfaceEulerSolver(use_native=False, use_graph=use_graph,
                                     graph_steps=100).solve(
                sm, dt=0.01, n_iters=1234, trj_waypoints=[1, 333, 1234])
            out.append((_get_numpy(res["trj"]), _get_numpy(sm.y)))
        np.testing.assert_array_equal(out[0][0], out[1][0])
        np.testing.assert_array_equal(out[0][1], out[1][1])

    def test_cupy_fallback_path(self, sphere_mesh, monkeypatch):
        fused = _liaw_surface("cuda:0", sphere_mesh)
        SurfaceEulerSolver().solve(fused, dt=0.01, n_iters=500)
        monkeypatch.setattr(SurfaceModel, "has_fused_kernel",
                            property(lambda self: False))
        generic = _liaw_surface("cuda:0", sphere_mesh)
        SurfaceEulerSolver().solve(generic, dt=0.01, n_iters=500)
        np.testing.assert_allclose(_get_numpy(generic.y), _get_numpy(fused.y),
                                   rtol=1e-4, atol=1e-6)


# ---------- all solvers and execution paths ----------

@pytest.fixture
def jit_only(monkeypatch):
    """Kernel managers without AOT kernels (JIT compilation only)."""
    import lpf.kernels.compiler as compiler
    from lpf.kernels.aot.loader import AOTKernelLoader
    monkeypatch.setattr(AOTKernelLoader, "is_available", lambda self: False)
    compiler._KERNEL_CACHE.clear()
    yield
    compiler._KERNEL_CACHE.clear()


def _native_has_surface():
    from lpf.solvers._cuda.base import _get_native
    return _get_native().supports("surface")


def _aot_available():
    from lpf.kernels.aot.loader import AOTKernelLoader
    return AOTKernelLoader().is_available()


class TestAllSolvers:

    def test_ell_matches_csr(self, sphere_mesh):
        op = sphere_mesh.operator
        nbr, wts = _to_ell(op)
        u = np.random.default_rng(0).random(op.shape[0])
        ell = (wts * u[nbr]).sum(axis=0)
        np.testing.assert_allclose(ell, op @ u, rtol=1e-12)

    @pytest.mark.parametrize("cls", ALL_SURFACE_SOLVERS)
    def test_reordering_keeps_mesh_order(self, sphere_mesh, cls):
        out = []
        for reorder in (False, True):
            seeds = [sphere_mesh.farthest_points(20, start=b) for b in range(2)]
            y0 = SurfaceModel.seeded_state(sphere_mesh, [[2.0, 1.0]] * 2, seeds)
            sm = SurfaceModel(LiawModel(params=LIAW_PARAMS, dtype=np.float64),
                              sphere_mesh, y0, reorder=reorder)
            res = cls().solve(sm, dt=0.01, n_iters=300, trj_waypoints=[150, 300])
            out.append((_get_numpy(res["trj"]), _get_numpy(sm.y)))
        np.testing.assert_allclose(out[1][0], out[0][0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(out[1][1], out[0][1], rtol=1e-10, atol=1e-12)

    def test_relabeled_mesh_gives_same_field(self):
        """Renumbering the mesh's vertices only renumbers the solution."""
        v, f = icosphere(3)
        perm = np.random.default_rng(3).permutation(len(v))  # new k = old perm[k]
        inv = np.argsort(perm)                               # old i -> new inv[i]
        seeds = np.arange(0, len(v), 37)
        out = []
        for verts, faces, seed_ids in ((v, f, seeds), (v[perm], inv[f], inv[seeds])):
            mesh = SurfaceMesh(verts, faces, scale=4.0)
            y0 = SurfaceModel.seeded_state(mesh, [[2.0, 1.0]], [seed_ids])
            sm = SurfaceModel(LiawModel(params=LIAW_PARAMS[:1], dtype=np.float64), mesh, y0)
            SurfaceEulerSolver().solve(sm, dt=0.01, n_iters=300)
            out.append(_get_numpy(sm.y))
        np.testing.assert_allclose(out[1][..., inv], out[0], rtol=1e-10, atol=1e-12)

    def test_stability_bound_depends_on_solver(self, sphere_mesh):
        sm = _liaw_surface("cpu", sphere_mesh)
        euler = SurfaceEulerSolver().max_stable_dt(sm)
        rk4 = SurfaceRungeKuttaSolver().max_stable_dt(sm)
        ab2 = SurfaceAdamsBashforth2Solver().max_stable_dt(sm)
        assert ab2 < euler < rk4
        dt = 0.5 * (euler + rk4)
        with pytest.warns(UserWarning, match="stability"):
            SurfaceEulerSolver().solve(sm, dt=dt, n_iters=1)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SurfaceRungeKuttaSolver().solve(sm, dt=dt, n_iters=1)

    def test_early_stopping_cpu(self, sphere_mesh):
        params = LIAW_PARAMS[:1].astype(np.float64).copy()
        params[0, 2:] = 0.0
        params[0, 0] = params[0, 1] = 0.2
        sm = _liaw_surface("cpu", sphere_mesh, dtype=np.float64, params=params)
        solver = SurfaceEulerSolver()
        res = solver.solve(sm, dt=0.01, n_iters=200000, rtol=1e-6,
                           period_output=1000)
        assert 0 < solver.stopped_at < 200000
        assert solver.stopped_at % 1000 == 0
        assert res["iters"][-1] == solver.stopped_at


class TestAllSolversCUDA:

    @pytest.fixture(autouse=True)
    def _need_cupy(self):
        cupy = pytest.importorskip("cupy")
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")

    @pytest.mark.parametrize("cls", ALL_SURFACE_SOLVERS)
    def test_gpu_matches_cpu(self, sphere_mesh, cls):
        results = []
        for device in ("cpu", "cuda:0"):
            sm = _liaw_surface(device, sphere_mesh)
            cls().solve(sm, dt=0.01, n_iters=1000)
            results.append(_get_numpy(sm.y))
        np.testing.assert_allclose(results[1], results[0], rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("cls", ALL_SURFACE_SOLVERS)
    def test_native_matches_python_loop(self, sphere_mesh, cls):
        if not _native_has_surface():
            pytest.skip("native surface loop not built")
        out = {}
        for use_native in (True, False):
            sm = _liaw_surface("cuda:0", sphere_mesh)
            solver = cls(use_native=use_native, graph_steps=100)
            res = solver.solve(sm, dt=0.01, n_iters=777, trj_waypoints=[1, 300, 777])
            out[solver.backend] = (_get_numpy(res["trj"]), _get_numpy(sm.y))
        assert set(out) == {"native", "fused"}
        # Same kernel sequence; only the AOT utility kernels (fmaf) may
        # differ from the native loop's in the last bit.
        np.testing.assert_allclose(out["native"][0], out["fused"][0], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(out["native"][1], out["fused"][1], rtol=1e-5, atol=1e-7)

    def test_euler_native_matches_jit(self, sphere_mesh, jit_only):
        if not _native_has_surface():
            pytest.skip("native surface loop not built")
        out = {}
        for use_native in (False, True):
            sm = _liaw_surface("cuda:0", sphere_mesh)
            solver = SurfaceEulerSolver(use_native=use_native)
            solver.solve(sm, dt=0.01, n_iters=500)
            out[solver.backend] = _get_numpy(sm.y)
        np.testing.assert_array_equal(out["native"], out["fused"])

    def test_euler_jit_matches_aot(self, sphere_mesh, monkeypatch):
        if not _aot_available():
            pytest.skip("AOT kernels not built")
        import lpf.kernels.compiler as compiler
        from lpf.kernels.aot.loader import AOTKernelLoader
        out = []
        for use_aot in (True, False):
            if not use_aot:
                monkeypatch.setattr(AOTKernelLoader, "is_available", lambda self: False)
            compiler._KERNEL_CACHE.clear()
            sm = _liaw_surface("cuda:0", sphere_mesh)
            SurfaceEulerSolver(use_native=False).solve(sm, dt=0.01, n_iters=500)
            out.append(_get_numpy(sm.y))
        compiler._KERNEL_CACHE.clear()
        np.testing.assert_array_equal(out[0], out[1])

    @pytest.mark.parametrize("cls", ALL_SURFACE_SOLVERS)
    def test_graph_replay_matches_plain_launches(self, sphere_mesh, cls):
        out = []
        for use_graph in (False, True):
            sm = _liaw_surface("cuda:0", sphere_mesh)
            res = cls(use_native=False, use_graph=use_graph, graph_steps=100).solve(
                sm, dt=0.01, n_iters=555, trj_waypoints=[1, 250, 555])
            out.append((_get_numpy(res["trj"]), _get_numpy(sm.y)))
        np.testing.assert_array_equal(out[0][0], out[1][0])
        np.testing.assert_array_equal(out[0][1], out[1][1])

    @pytest.mark.parametrize("use_native", [True, False])
    def test_early_stopping_gpu_matches_cpu(self, sphere_mesh, use_native):
        if use_native and not _native_has_surface():
            pytest.skip("native surface loop not built")
        params = LIAW_PARAMS[:1].copy()
        params[0, 2:] = 0.0
        params[0, 0] = params[0, 1] = 0.2
        stops = []
        for device in ("cpu", "cuda:0"):
            sm = _liaw_surface(device, sphere_mesh, params=params)
            solver = SurfaceEulerSolver(use_native=use_native)
            res = solver.solve(sm, dt=0.01, n_iters=200000, rtol=1e-5,
                               period_output=1000)
            stops.append(solver.stopped_at)
            assert res["iters"][-1] == solver.stopped_at
        assert stops[0] == stops[1] > 0
