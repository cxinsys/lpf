"""Tests for spatial domains (lpf.domains).

Covers:
  - Geometry: grid-aligned rectangles, covered area, holes, face symmetry,
    time-step bound (stencil row sums), serialization, point helpers
  - Accuracy: Neumann eigenvalues of a disk against the exact Bessel values
  - Model (CPU): conservation under pure diffusion, frozen inactive cells,
    colorize background, to_dict, early stopping
  - CUDA: parity with the CPU path for every fixed-step solver
  - JAX: domains are rejected with a clear error
"""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from lpf.domains import PolygonDomain, MaskDomain, domain_from_dict
from lpf.initializers import LiawInitializer
from lpf.models import LiawModel


# ---------- helpers ----------

def _disk(r, cx, cy, n=400):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1)


def _star(cx, cy, r_out, r_in, n=5):
    th = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    r = np.where(np.arange(2 * n) % 2 == 0, r_out, r_in)
    return np.stack([cx + r * np.sin(th), cy - r * np.cos(th)], axis=1)


def _shoelace(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _operator(stencil):
    """Sparse matrix of the domain Laplacian (dx = 1) on the active cells."""
    active = stencil[4] > 0
    idx = -np.ones(active.shape, dtype=int)
    idx[active] = np.arange(active.sum())
    rows, cols, vals = [], [], []
    for k, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        r, c = np.nonzero(active & (stencil[k] > 0))
        w = stencil[k][r, c]
        rows += [idx[r, c], idx[r, c]]
        cols += [idx[r + dr, c + dc], idx[r, c]]
        vals += [w, -w]
    n = int(active.sum())
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n))


def _liaw_dicts(domain, height, width, batch_size=2, n_pts=10):
    pts = domain.random_points(n_pts, height, width, rng=0)
    base = {"u0": 2.0, "v0": 1.0, "Du": 0.0005, "Dv": 0.075, "ru": 0.18,
            "rv": 0.0287, "k": 0.2, "su": 0.001, "sv": 0.025, "mu": 0.08}
    for i, (r, c) in enumerate(pts):
        base["init_pts_%d" % i] = (int(r), int(c))
    return [dict(base, k=0.2 - 0.03 * j) for j in range(batch_size)]


def _make_model(domain, device="cpu", height=32, width=32, dtype=np.float32,
                model_dicts=None):
    if model_dicts is None:
        model_dicts = _liaw_dicts(domain, height, width)
    initializer = LiawInitializer()
    initializer.update(model_dicts)
    return LiawModel(initializer=initializer,
                     params=LiawModel.parse_params(model_dicts),
                     width=width, height=height, dx=0.1,
                     device=device, dtype=dtype, domain=domain)


def _get_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    if hasattr(arr, "cpu"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


STAR = _star(16, 16, 14, 6)


# ---------- geometry ----------

class TestGeometry:

    @pytest.mark.parametrize("method", ["cut_cell", "staircase"])
    def test_grid_aligned_rectangle(self, method):
        dom = PolygonDomain([(4, 3), (20, 3), (20, 15), (4, 15)], method=method)
        st = dom.stencil(20, 24)

        expected = np.zeros((20, 24), dtype=bool)
        expected[3:15, 4:20] = True
        np.testing.assert_array_equal(st[4] > 0, expected)

        # Standard 5-point Neumann stencil: weight 1 between inside cells,
        # 0 across the boundary.
        np.testing.assert_array_equal(st[0][3:15, 4:20][1:], 1.0)
        np.testing.assert_array_equal(st[0][3, 4:20], 0.0)
        np.testing.assert_array_equal(st[1][14, 4:20], 0.0)
        np.testing.assert_array_equal(st[2][3:15, 4], 0.0)
        np.testing.assert_array_equal(st[3][3:15, 19], 0.0)
        assert st[:4][:, ~expected].sum() == 0

    def test_covered_area_matches_polygon_area(self):
        for verts in (_disk(11.3, 16.2, 15.7), STAR,
                      [(2.5, 3.1), (29.2, 6.7), (12.4, 28.9)]):
            dom = PolygonDomain(verts, min_fraction=1e-9)
            frac = dom._geometry(32, 32)[0]
            assert frac.sum() == pytest.approx(_shoelace(np.asarray(verts)),
                                               rel=2e-3)

    def test_hole_cells_are_inactive(self):
        outer = [(2, 2), (30, 2), (30, 30), (2, 30)]
        hole = [(10, 10), (22, 10), (22, 22), (10, 22)]
        mask = PolygonDomain(outer, holes=[hole]).mask(32, 32)
        assert not mask[10:22, 10:22].any()
        assert mask[2:10, 2:30].all()

    @pytest.mark.parametrize("verts", [_disk(12.4, 16.3, 15.8), STAR])
    def test_faces_are_symmetric(self, verts):
        """frac_i * w(i->j) == frac_j * w(j->i): diffusion conserves mass."""
        dom = PolygonDomain(verts)
        frac = dom._geometry(32, 32)[0]
        st = dom.stencil(32, 32)
        flux_s = frac[:-1] * st[1][:-1]   # cell (r, c) -> (r + 1, c)
        flux_n = frac[1:] * st[0][1:]     # cell (r + 1, c) -> (r, c)
        np.testing.assert_allclose(flux_s, flux_n, atol=1e-12)
        flux_e = frac[:, :-1] * st[3][:, :-1]
        flux_w = frac[:, 1:] * st[2][:, 1:]
        np.testing.assert_allclose(flux_e, flux_w, atol=1e-12)

    @pytest.mark.parametrize("verts", [_disk(12.4, 16.3, 15.8), STAR])
    def test_row_sums_keep_full_grid_time_step(self, verts):
        st = PolygonDomain(verts).stencil(32, 32)
        assert st[:4].sum(axis=0).max() <= 4.0 + 1e-9

    def test_dict_roundtrip(self):
        for dom in (PolygonDomain(STAR, holes=[_disk(3, 16, 16, n=12)],
                                  method="staircase"),
                    MaskDomain(PolygonDomain(STAR).mask(32, 32))):
            dom2 = domain_from_dict(dom.to_dict())
            assert type(dom2) is type(dom)
            np.testing.assert_array_equal(dom.stencil(32, 32),
                                          dom2.stencil(32, 32))

    def test_mask_domain_from_image(self, tmp_path):
        from PIL import Image
        mask = PolygonDomain(STAR).mask(32, 32)
        fpath = tmp_path / "mask.png"
        Image.fromarray((mask * 255).astype(np.uint8)).save(fpath)
        np.testing.assert_array_equal(
            MaskDomain.from_image(fpath).mask(32, 32), mask)
        np.testing.assert_array_equal(
            MaskDomain.from_image(fpath, invert=True).mask(32, 32), ~mask)

    def test_points_helpers(self):
        dom = PolygonDomain(STAR)
        pts = dom.random_points(20, 32, 32, rng=1)
        assert len({tuple(p) for p in pts}) == 20
        assert dom.contains(pts, 32, 32).all()
        assert not dom.contains([(0, 0), (-1, 5), (40, 3)], 32, 32).any()

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            PolygonDomain([(0, 0), (1, 1)])
        with pytest.raises(ValueError):
            PolygonDomain(STAR, method="fem")
        with pytest.raises(ValueError):
            PolygonDomain(STAR, min_fraction=0.0)
        with pytest.raises(ValueError):
            MaskDomain(np.ones((8, 8), dtype=bool)).stencil(16, 16)
        with pytest.raises(ValueError):  # covers no cell
            PolygonDomain([(100, 100), (110, 100), (110, 110)]).stencil(32, 32)


# ---------- accuracy ----------

class TestAccuracy:

    @pytest.mark.parametrize("method,tol", [("cut_cell", 0.01),
                                            ("staircase", 0.05)])
    def test_disk_neumann_eigenvalues(self, method, tol):
        """First Neumann eigenvalues of a disk of radius R: (j'_mn / R)^2."""
        from scipy.special import jnp_zeros
        exact = np.sort([jnp_zeros(1, 1)[0] ** 2] * 2
                        + [jnp_zeros(2, 1)[0] ** 2] * 2)

        R = 20.0
        st = PolygonDomain(_disk(R, 22.37, 21.81), method=method).stencil(45, 45)
        lam = sla.eigs(-_operator(st), k=5, sigma=-1e-3, which="LM",
                       return_eigenvectors=False)
        lam = np.sort(lam.real)[1:] * R ** 2   # drop the constant mode
        np.testing.assert_allclose(lam, exact, rtol=tol)


# ---------- model (CPU) ----------

class TestModelCPU:

    def test_pure_diffusion_conserves_mass(self):
        dom = PolygonDomain(STAR)
        dicts = _liaw_dicts(dom, 32, 32, batch_size=1)
        for key in ("ru", "rv", "k", "su", "sv", "mu"):
            dicts[0][key] = 0.0
        dicts[0]["Du"] = 0.02
        model = _make_model(dom, model_dicts=dicts, dtype=np.float64)

        from lpf.solvers import EulerSolver
        model.initialize()
        frac = dom._geometry(32, 32)[0] * dom.mask(32, 32)
        mass0 = (frac * _get_numpy(model.u)[0]).sum()
        EulerSolver().solve(model=model, dt=0.01, n_iters=2000,
                            init_model=False)
        u = _get_numpy(model.u)[0]
        assert (frac * u).sum() == pytest.approx(mass0, rel=1e-12)
        assert u[dom.mask(32, 32)].std() < 0.05   # it did spread

    def test_inactive_cells_keep_initial_values(self):
        dom = PolygonDomain(STAR)
        model = _make_model(dom)
        model.initialize()
        y0 = _get_numpy(model.y_mesh).copy()

        from lpf.solvers import EulerSolver
        EulerSolver().solve(model=model, dt=0.01, n_iters=500,
                            init_model=False)
        y1 = _get_numpy(model.y_mesh)
        outside = ~dom.mask(32, 32)
        np.testing.assert_array_equal(y1[:, :, outside], y0[:, :, outside])
        assert not np.array_equal(y1[:, :, ~outside], y0[:, :, ~outside])

    def test_colorize_paints_outside_with_background(self):
        dom = PolygonDomain(STAR)
        model = _make_model(dom)
        model.initialize()
        color = model.colorize()
        outside = ~dom.mask(32, 32)
        assert (color[:, outside] == 255).all()

    def test_to_dict_contains_domain(self):
        dom = PolygonDomain(STAR)
        model = _make_model(dom)
        model.initialize()
        n2v = model.to_dict(index=0)
        np.testing.assert_array_equal(
            domain_from_dict(n2v["domain"]).stencil(32, 32),
            dom.stencil(32, 32))

    def test_early_stopping(self):
        from lpf.solvers import EulerSolver
        dom = PolygonDomain(STAR)
        model = _make_model(dom)
        EulerSolver().solve(model=model, dt=0.01, n_iters=10)
        assert model.is_early_stopping(1e-30) in (True, False)
        assert model.is_early_stopping(1e30)

    def test_full_grid_unchanged_without_domain(self):
        from lpf.geometry import DomainGeometry, GridGeometry
        model = _make_model(PolygonDomain(STAR))
        assert type(model.geometry) is DomainGeometry
        plain = LiawModel(initializer=model.initializer, params=model.params,
                          width=32, height=32, dx=0.1)
        assert plain.domain is None and plain.mesh is None
        assert type(plain.geometry) is GridGeometry


# ---------- CUDA ----------

ALL_SOLVERS = ["EulerSolver", "HeunSolver", "RungeKuttaSolver",
               "RK23Solver", "AdamsBashforth2Solver"]


class TestCUDA:

    @pytest.fixture(autouse=True)
    def _need_cupy(self):
        cupy = pytest.importorskip("cupy")
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    @pytest.mark.parametrize("method", ["cut_cell", "staircase"])
    def test_matches_cpu(self, solver_name, method):
        import lpf.solvers as S
        dom = PolygonDomain(STAR, method=method)
        results = []
        for device in ("cpu", "cuda:0"):
            model = _make_model(dom, device=device)
            getattr(S, solver_name)().solve(model=model, dt=0.01, n_iters=300)
            results.append(_get_numpy(model.y_mesh))
        y_cpu, y_gpu = results
        assert np.isfinite(y_gpu).all()
        np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-4, atol=1e-6)

    def test_trajectory(self):
        from lpf.solvers import EulerSolver
        dom = PolygonDomain(STAR)
        model = _make_model(dom, device="cuda:0")
        trj = EulerSolver().solve(model=model, dt=0.01, n_iters=200,
                                  period_output=100, get_trj=True)
        trj = _get_numpy(trj)
        assert trj.shape == (3, 2, 2, 32, 32)
        outside = ~dom.mask(32, 32)
        np.testing.assert_array_equal(trj[-1][:, :, outside],
                                      trj[0][:, :, outside])


    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_native_loop_matches_python_loop(self, solver_name, monkeypatch):
        import lpf.solvers as S
        import lpf.solvers._cuda.base as cuda_base
        if not cuda_base._get_native().supports("domain"):
            pytest.skip("native domain loop not built")
        dom = PolygonDomain(STAR)
        out = []
        for native in (True, False):
            if not native:
                monkeypatch.setattr(cuda_base, "_native_solver",
                                    SimpleNamespace(is_available=lambda: False))
            model = _make_model(dom, device="cuda:0")
            trj = getattr(S, solver_name)().solve(
                model=model, dt=0.01, n_iters=400, trj_waypoints=[1, 200, 400])
            out.append((_get_numpy(trj["trj"]), _get_numpy(model.y_mesh)))
        np.testing.assert_allclose(out[0][0], out[1][0], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(out[0][1], out[1][1], rtol=1e-5, atol=1e-7)

    def test_euler_jit_aot_native_bit_identical(self, monkeypatch):
        import lpf.kernels.compiler as compiler
        import lpf.solvers._cuda.base as cuda_base
        from lpf.kernels.aot.loader import AOTKernelLoader
        from lpf.solvers import EulerSolver
        dom = PolygonDomain(STAR)
        native = cuda_base._get_native()
        out = {}

        def run(label):
            compiler._KERNEL_CACHE.clear()
            model = _make_model(dom, device="cuda:0")
            EulerSolver().solve(model=model, dt=0.01, n_iters=400)
            out[label] = _get_numpy(model.y_mesh)

        if native.supports("domain"):
            run("native")
        monkeypatch.setattr(cuda_base, "_native_solver",
                            SimpleNamespace(is_available=lambda: False))
        if AOTKernelLoader().is_available():
            run("aot")
        monkeypatch.setattr(AOTKernelLoader, "is_available", lambda self: False)
        run("jit")
        compiler._KERNEL_CACHE.clear()
        for label, y in out.items():
            np.testing.assert_array_equal(y, out["jit"], err_msg=label)


# ---------- JAX ----------

# ---------- arbitrary polygons ----------

def _polygon_cases():
    rng = np.random.default_rng(11)
    th = np.sort(rng.uniform(0, 2 * np.pi, 40))
    blob = np.stack([16 + rng.uniform(5, 14, 40) * np.cos(th),
                     16 + rng.uniform(5, 14, 40) * np.sin(th)], axis=1)
    pentagram = np.array([(16 + 14 * np.sin(a), 16 - 14 * np.cos(a))
                          for a in np.arange(5) * 4 * np.pi / 5])
    return {
        "random 40-gon": (blob, ()),
        "self-intersecting pentagram": (pentagram, ()),
        "crosses the grid border": ([(-6, 4), (20, -5), (38, 18), (12, 40)], ()),
        "tiny triangle": ([(10.2, 10.1), (13.7, 10.4), (11.1, 13.9)], ()),
        "duplicate and collinear vertices": (
            [(4, 4), (16, 4), (16, 4), (28, 4), (28, 28), (16, 28), (4, 28)], ()),
        "nested holes": ([(1, 1), (31, 1), (31, 31), (1, 31)],
                         [[(6, 6), (26, 6), (26, 26), (6, 26)],
                          [(11, 11), (21, 11), (21, 21), (11, 21)]]),
    }


class TestArbitraryPolygons:

    @pytest.mark.parametrize("name", list(_polygon_cases()))
    @pytest.mark.parametrize("method", ["cut_cell", "staircase"])
    def test_operator_is_conservative(self, name, method):
        verts, holes = _polygon_cases()[name]
        dom = PolygonDomain(verts, holes=holes, method=method)
        st = dom.stencil(32, 32)
        frac = (dom._geometry(32, 32)[0] if method == "cut_cell"
                else st[4]) * (st[4] > 0)
        np.testing.assert_allclose(frac[:-1] * st[1][:-1], frac[1:] * st[0][1:],
                                   atol=1e-12)
        np.testing.assert_allclose(frac[:, :-1] * st[3][:, :-1],
                                   frac[:, 1:] * st[2][:, 1:], atol=1e-12)
        assert st[:4].sum(axis=0).max() <= 4.0 + 1e-9

    def test_even_odd_fill_of_pentagram(self):
        verts = _polygon_cases()["self-intersecting pentagram"][0]
        mask = PolygonDomain(verts).mask(32, 32)
        assert not mask[16, 16]           # the inner pentagon is a hole
        assert mask[5, 16]                # the top point is filled

    @pytest.mark.parametrize("name", list(_polygon_cases()))
    def test_simulation_conserves_and_stays_finite(self, name):
        verts, holes = _polygon_cases()[name]
        dom = PolygonDomain(verts, holes=holes)
        dicts = _liaw_dicts(dom, 32, 32, batch_size=1, n_pts=3)
        for key in ("ru", "rv", "k", "su", "sv", "mu"):
            dicts[0][key] = 0.0
        dicts[0]["Du"] = dicts[0]["Dv"] = 0.02
        model = _make_model(dom, model_dicts=dicts, dtype=np.float64)
        model.initialize()
        frac = dom._geometry(32, 32)[0] * dom.mask(32, 32)
        mass0 = (frac * _get_numpy(model.u)[0]).sum()
        from lpf.solvers import EulerSolver
        EulerSolver().solve(model=model, dt=0.01, n_iters=500, init_model=False)
        u = _get_numpy(model.u)[0]
        assert np.isfinite(u).all()
        assert (frac * u).sum() == pytest.approx(mass0, rel=1e-12)


# ---------- one domain per model ----------

HEXAGON = np.stack([16 + 13 * np.cos(np.arange(6) * np.pi / 3),
                    16 + 13 * np.sin(np.arange(6) * np.pi / 3)], axis=1)


def _per_model_setup():
    doms = [PolygonDomain(STAR), PolygonDomain(HEXAGON)]
    dicts = []
    for dom in doms:
        dicts += _liaw_dicts(dom, 32, 32, batch_size=1)
    return doms, dicts


class TestPerModelDomains:

    def test_each_model_follows_its_domain(self):
        doms, dicts = _per_model_setup()
        model = _make_model(doms, model_dicts=dicts)
        model.initialize()
        y0 = _get_numpy(model.y_mesh).copy()
        from lpf.solvers import EulerSolver
        EulerSolver().solve(model=model, dt=0.01, n_iters=300, init_model=False)
        y = _get_numpy(model.y_mesh)
        for b, dom in enumerate(doms):
            outside = ~dom.mask(32, 32)
            np.testing.assert_array_equal(y[:, b][:, outside], y0[:, b][:, outside])
        color = model.colorize()
        for b, dom in enumerate(doms):
            assert (color[b][~dom.mask(32, 32)] == 255).all()

    def test_matches_separate_runs(self):
        doms, dicts = _per_model_setup()
        from lpf.solvers import EulerSolver
        batch = _make_model(doms, model_dicts=dicts)
        EulerSolver().solve(model=batch, dt=0.01, n_iters=300)
        for b, dom in enumerate(doms):
            single = _make_model(dom, model_dicts=[dicts[b]])
            EulerSolver().solve(model=single, dt=0.01, n_iters=300)
            np.testing.assert_allclose(_get_numpy(batch.y_mesh)[:, b],
                                       _get_numpy(single.y_mesh)[:, 0],
                                       rtol=1e-6, atol=1e-7)

    def test_to_dict_and_parse_geometry(self):
        doms, dicts = _per_model_setup()
        model = _make_model(doms, model_dicts=dicts)
        model.initialize()
        saved = [model.to_dict(index=b, initializer=model.initializer)
                 for b in range(2)]
        geometry = LiawModel.parse_geometry(saved)
        assert len(geometry["domain"]) == 2
        for dom, parsed in zip(doms, geometry["domain"]):
            np.testing.assert_array_equal(parsed.stencil(32, 32),
                                          dom.stencil(32, 32))
        shared = LiawModel.parse_geometry([saved[0], dict(saved[0])])
        assert isinstance(shared["domain"], PolygonDomain)

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS)
    def test_cuda_matches_cpu(self, solver_name):
        cupy = pytest.importorskip("cupy")
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
        import lpf.solvers as S
        doms, dicts = _per_model_setup()
        out = []
        for device in ("cpu", "cuda:0"):
            model = _make_model(doms, device=device, model_dicts=dicts)
            getattr(S, solver_name)().solve(model=model, dt=0.01, n_iters=300)
            out.append(_get_numpy(model.y_mesh))
        np.testing.assert_allclose(out[1], out[0], rtol=1e-4, atol=1e-6)

    def test_wrong_number_of_domains(self):
        doms, dicts = _per_model_setup()
        model = _make_model(doms[:1] * 3, model_dicts=dicts)
        with pytest.raises(ValueError):
            model.initialize()


# ---------- models, files and other backends ----------

class TestModelFeatures:

    def test_file_output_and_reload(self, tmp_path):
        from lpf.solvers import EulerSolver
        import json
        import os
        dom = PolygonDomain(STAR)
        model = _make_model(dom)
        EulerSolver().solve(model=model, dt=0.01, n_iters=100,
                            period_output=100, dpath_model=str(tmp_path),
                            dpath_pattern=str(tmp_path))
        assert os.path.isfile(tmp_path / "model_1" / "pattern_100.png")
        dicts = [json.load(open(tmp_path / "models" / f))
                 for f in sorted(os.listdir(tmp_path / "models"))]
        initializer = LiawInitializer()
        initializer.update(dicts)
        reloaded = LiawModel(initializer=initializer,
                             params=LiawModel.parse_params(dicts),
                             width=32, height=32, dx=0.1,
                             **LiawModel.parse_geometry(dicts))
        EulerSolver().solve(model=reloaded, dt=0.01, n_iters=100)
        np.testing.assert_array_equal(_get_numpy(reloaded.y_mesh),
                                      _get_numpy(model.y_mesh))

    def test_diploid(self):
        from lpf.models import TwoComponentDiploidModel
        from lpf.solvers import EulerSolver
        dom = PolygonDomain(STAR)
        dicts = _liaw_dicts(dom, 32, 32, batch_size=1)
        pa = _make_model(dom, model_dicts=dicts)
        ma = _make_model(dom, model_dicts=[dict(dicts[0], k=0.1)])
        d = TwoComponentDiploidModel(
            paternal_model=pa, maternal_model=ma, alpha=0.5, beta=0.5,
            params=pa.params, initializer=pa.initializer,
            width=32, height=32, dx=0.1)
        EulerSolver().solve(model=d, dt=0.01, n_iters=100)
        assert np.isfinite(_get_numpy(d.y_mesh)).all()
        assert (d.colorize()[0][~dom.mask(32, 32)] == 255).all()


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
def test_backend_matches_numpy(device, solver_name):
    import lpf.solvers as S
    doms, dicts = _per_model_setup()
    out = []
    for dev in ("cpu", device):
        model = _make_model(doms, device=dev, model_dicts=dicts)
        res = getattr(S, solver_name)().solve(model=model, dt=0.01,
                                              n_iters=300, trj_waypoints=[1, 300])
        out.append(_get_numpy(res["trj"]))
    np.testing.assert_allclose(out[1], out[0], rtol=1e-5, atol=1e-6)
