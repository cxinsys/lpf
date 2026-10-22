"""Solve LPF two-component models on the vertices of a SurfaceMesh.

Execution paths, chosen automatically like lpf.solvers on CUDA:

  1. Native .so   - the whole solve loop in C/CUDA (lpf_native_solve_surface
                    in libsolver.so), zero Python during iteration
  2. Fused kernels - AOT (or JIT) surface kernels launched from a Python loop;
                    blocks of steps are captured once as a CUDA graph and
                    replayed
  3. Generic      - SurfaceModel.pdefunc() with NumPy/SciPy (CPU), or CuPy for
                    models without a CUDA reaction kernel

The stage formulas match the corresponding lpf.solvers classes, and paths 1
and 2 launch the same kernel sequence as the native grid loops.
"""

import time
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

from lpf.array.module import CupyModule, NumpyModule
from lpf.kernels.reactions import REACTION_REGISTRY

_RTOL_PERIOD = 1000  # check early stopping every N iterations (as lpf.solvers)


class SurfaceModel:
    """An LPF two-component model on the vertices of a SurfaceMesh.

    The kinetics come from ``model`` (e.g. ``LiawModel(params=...,
    device=...)``): its parameters, ``reactions()``, dtype and device are
    used; its grid settings (width, height, dx, initializer) are not.

    Args:
        model: A TwoComponentModel with params, on a NumPy or CuPy device.
        mesh: The SurfaceMesh to solve on.
        y0: Initial state of shape (2, B, N): u and v per batch and vertex,
            in the mesh's vertex order.
        reorder: Renumber the vertices internally (reverse Cuthill-McKee) so
            that neighbors are close in memory. This makes the GPU kernels
            up to ~4x faster on badly ordered meshes. Inputs and outputs
            always use the mesh's vertex order.
    """

    def __init__(self, model, mesh, y0, reorder=True):
        if type(model.am) not in (NumpyModule, CupyModule):
            raise NotImplementedError(
                "Surface models support NumPy ('cpu') and CuPy ('cuda') "
                "devices.")

        y0 = np.asarray(y0)
        expected = (2, model.batch_size, mesh.n_vertices)
        if y0.shape != expected:
            raise ValueError("y0 should have shape %s, not %s."
                             % (expected, y0.shape))

        self._model = model
        self._mesh = mesh
        self._on_gpu = isinstance(model.am, CupyModule)

        op = mesh.operator
        n = mesh.n_vertices
        if reorder:
            perm = reverse_cuthill_mckee(op, symmetric_mode=True).astype(np.int64)
        else:
            perm = np.arange(n)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(n)
        op = sp.csr_matrix(op[perm][:, perm])
        op.sort_indices()

        nbr, wts = _to_ell(op)
        self._K = nbr.shape[0]
        row_sum = np.asarray(op.sum(axis=1)).ravel()

        with model.am:
            # Kernels index the state as a C-ordered (2, B, N) array.
            self._y = model.am.array(np.ascontiguousarray(y0[..., perm]),
                                     dtype=model.dtype)
            self._row_sum = model.am.array(row_sum, dtype=model.dtype)
            if self._on_gpu:
                import cupy as cp
                import cupyx.scipy.sparse as cusp
                self._perm = cp.asarray(perm)
                self._inv = cp.asarray(inv)
                self._nbr = cp.asarray(nbr, dtype=cp.int32)
                self._wts = cp.asarray(wts, dtype=model.dtype)
                self._op = cusp.csr_matrix(op.astype(model.dtype))
            else:
                self._perm, self._inv = perm, inv
                self._op = op.astype(model.dtype)

    @staticmethod
    def seeded_state(mesh, init_states, init_pts):
        """Initial state in the style of LiawInitializer.

        u = u0 at the seed vertices and 0 elsewhere; v = v0 everywhere.

        Args:
            init_states: (B, 2) array of (u0, v0).
            init_pts: (B, P) vertex indices of the seeds.
        """
        init_states = np.asarray(init_states, dtype=np.float64)
        y0 = np.zeros((2, len(init_states), mesh.n_vertices))
        for b, pts in enumerate(init_pts):
            y0[0, b, np.asarray(pts, dtype=np.int64)] = init_states[b, 0]
            y0[1, b, :] = init_states[b, 1]
        return y0

    @property
    def model(self):
        return self._model

    @property
    def mesh(self):
        return self._mesh

    @property
    def am(self):
        return self._model.am

    @property
    def y(self):  # (2, B, N) in the mesh's vertex order
        return self.to_mesh_order(self._y)

    @property
    def u(self):
        return self.y[0]

    @property
    def v(self):
        return self.y[1]

    @property
    def batch_size(self):
        return self._model.batch_size

    @property
    def n_vertices(self):
        return self._mesh.n_vertices

    @property
    def has_fused_kernel(self):
        return self._on_gpu and self._model.name in REACTION_REGISTRY

    def to_mesh_order(self, arr):
        """Map an array over the internal vertex order (last axis) back to
        the mesh's vertex order."""
        return _array_namespace(arr).ascontiguousarray(arr[..., self._inv])

    def laplacian(self, a):
        """Laplace-Beltrami operator of a (B, N) array in internal order."""
        return (self._op @ a.T).T - self._row_sum * a

    def pdefunc(self, t, y):
        """dy/dt of a (2, B, N) state in internal vertex order."""
        model = self._model
        B = model.batch_size
        u, v = y[0], y[1]

        Du = model.params[:, 0].reshape(B, 1)
        Dv = model.params[:, 1].reshape(B, 1)

        # Models broadcast their params as (B, 1, 1), so give them (B, 1, N).
        f, g = model.reactions(t, u[:, None, :], v[:, None, :])

        du = Du * self.laplacian(u) + f[:, 0, :]
        dv = Dv * self.laplacian(v) + g[:, 0, :]
        return _array_namespace(y).stack([du, dv])


def _to_ell(op):
    """CSR (no diagonal) -> transposed ELL (K, N); short rows padded with
    the row's own index and weight 0."""
    n = op.shape[0]
    counts = np.diff(op.indptr)
    k = max(int(counts.max(initial=0)), 1)
    rows = np.repeat(np.arange(n), counts)
    slot = np.arange(op.nnz) - np.repeat(op.indptr[:-1], counts)
    nbr = np.tile(np.arange(n), (k, 1))
    wts = np.zeros((k, n))
    nbr[slot, rows] = op.indices
    wts[slot, rows] = op.data
    return nbr, wts


def _get_native():
    from lpf.solvers._cuda.base import _get_native as get
    return get()


def _array_namespace(arr):
    if type(arr).__module__.startswith("cupy"):
        import cupy
        return cupy
    return np


class SurfaceSolver:
    """Base class of the surface solvers (see the module docstring).

    Args:
        use_native: Use the native solve loop when libsolver.so has it.
        use_graph: In the fused-kernel path, replay blocks of steps
            through CUDA graphs.
        graph_steps: Steps per captured graph (rounded up to even).

    After solve(), ``backend`` is "native", "fused" or "generic", and
    ``stopped_at`` the iteration where early stopping triggered (-1 if the
    run completed).
    """

    _name = None
    _native_code = None     # solver code of the native loop
    _stability = 2.0        # stable interval of dt * lambda on the negative axis
    _work_names = ()        # work buffers, in native-loop order
    _warmup_steps = 1       # plain steps before capturing a graph

    def __init__(self, use_native=True, use_graph=True, graph_steps=1000):
        self._use_native = use_native
        self._use_graph = use_graph
        self._graph_steps = graph_steps + graph_steps % 2
        self._graphs = {}
        self._stream = None
        self.backend = None
        self.stopped_at = None

    @property
    def name(self):
        return self._name

    def max_stable_dt(self, smodel):
        params = np.asarray(smodel.am.get(smodel.model.params))
        d_max = float(np.max(params[:, :2]))
        return self._stability / (d_max * smodel.mesh.lambda_max())

    def solve(self, smodel, dt=0.01, n_iters=None, rtol=None,
              trj_waypoints=None, period_output=None, check_stability=True,
              verbose=0):
        """Advance smodel by up to n_iters steps.

        Returns {"iters": [...], "trj": (T, 2, B, N) array in the mesh's
        vertex order} when trj_waypoints or period_output is given
        (recorded after those iterations, as in lpf.solvers), otherwise
        None. With rtol, stops once max|y_new - y_old| / max|y_old| < rtol
        at a check every 1000 iterations; the trajectory then ends there.
        """
        if n_iters is None or n_iters < 1:
            raise ValueError("n_iters should be a positive integer.")
        if rtol is not None and rtol < 0:
            raise ValueError("rtol should be greater than 0.")

        if check_stability:
            limit = self.max_stable_dt(smodel)
            if dt > limit:
                warnings.warn(
                    "dt=%g exceeds the %s stability bound %.3g for this mesh; "
                    "refine dt or coarsen the mesh." % (dt, self._name, limit))

        if trj_waypoints is not None:
            iters = sorted(set(int(w) for w in trj_waypoints
                               if 1 <= w <= n_iters))
        elif period_output is not None:
            iters = sorted({1} | set(range(period_output, n_iters + 1,
                                           period_output)))
        else:
            iters = []

        xp = _array_namespace(smodel._y)
        trj = None
        if iters:
            with smodel.am:
                trj = xp.zeros((len(iters), *smodel._y.shape),
                               dtype=smodel._y.dtype)

        t_beg = time.time()
        if smodel.has_fused_kernel:
            native = _get_native()
            if self._use_native and native.supports("surface"):
                self.backend = "native"
                stopped = self._solve_native(native, smodel, dt, n_iters,
                                             iters, trj, rtol)
            else:
                self.backend = "fused"
                stopped = self._solve_fused(smodel, dt, n_iters, iters, trj,
                                            rtol, verbose)
        else:
            self.backend = "generic"
            stopped = self._solve_generic(smodel, dt, n_iters, iters, trj,
                                          rtol, verbose)
        self.stopped_at = stopped

        if verbose >= 1:
            if stopped > 0:
                print("- [Early stopping at #%d]" % stopped)
            print("- [Duration] : %.5e sec. (%s)" % (time.time() - t_beg,
                                                    self.backend))

        if trj is None:
            return None
        n_done = len(iters) if stopped < 0 else sum(i <= stopped for i in iters)
        return {"iters": iters[:n_done],
                "trj": smodel.to_mesh_order(trj[:n_done])}

    # ---- generic path ----

    def _step_generic(self, smodel, t, dt, y):
        raise NotImplementedError

    def _solve_generic(self, smodel, dt, n_iters, iters, trj, rtol, verbose):
        xp = _array_namespace(smodel._y)
        record = {it: k for k, it in enumerate(iters)}
        self._prev_dydt = None
        y = smodel._y
        t = 0.0
        stopped = -1
        with smodel.am:
            for i in range(1, n_iters + 1):
                y_old = y
                y = y_old + self._step_generic(smodel, t, dt, y_old)
                t += dt
                if i in record:
                    trj[record[i]] = y
                    _progress(verbose, i, n_iters)
                if rtol is not None and i % _RTOL_PERIOD == 0 \
                        and _converged(xp, y, y_old, rtol):
                    stopped = i
                    break
            smodel._y = y
        return stopped

    # ---- native path ----

    def _alloc_work(self, smodel):
        import cupy as cp
        with smodel.am:
            return [cp.empty_like(smodel._y) for _ in self._work_names]

    def _solve_native(self, native, smodel, dt, n_iters, iters, trj, rtol):
        import cupy as cp
        model = smodel.model
        with smodel.am:
            y_a = cp.ascontiguousarray(smodel._y)
            y_b = cp.empty_like(y_a)
            res = native.solve_surface(
                solver_type=self._native_code, model_name=model.name,
                dtype=model.dtype, y_a=y_a, y_b=y_b, params=model.params,
                nbr=smodel._nbr, wts=smodel._wts, K=smodel._K,
                B=smodel.batch_size, N=smodel.n_vertices, dt=dt,
                n_iters=n_iters, work_bufs=self._alloc_work(smodel),
                trj_buf=trj,
                trj_iters=np.array(iters, dtype=np.int32) if iters else None,
                rtol=rtol if rtol else 0.0, rtol_period=_RTOL_PERIOD)
            smodel._y = y_a if res["result_buf"] == 0 else y_b
        return res["stopped_at"]

    # ---- fused-kernel path ----

    def _step_fused(self, km, smodel, src, dst, dt):
        raise NotImplementedError

    def _solve_fused(self, smodel, dt, n_iters, iters, trj, rtol, verbose):
        import cupy as cp
        from lpf.kernels.compiler import get_kernel_manager

        km = get_kernel_manager(smodel.model.name, smodel.model.dtype)
        if self._stream is None:
            self._stream = cp.cuda.Stream(non_blocking=True)
        self._work = dict(zip(self._work_names, self._alloc_work(smodel)))
        self._steps_done = 0
        self._graphs = {}

        record = {it: k for k, it in enumerate(iters)}
        stops = set(iters) | {n_iters}
        if rtol is not None:
            stops |= set(range(_RTOL_PERIOD, n_iters + 1, _RTOL_PERIOD))

        stopped = -1
        with smodel.am, self._stream:
            y_a = cp.ascontiguousarray(smodel._y)
            y_b = cp.empty_like(y_a)
            done = 0
            for stop in sorted(stops):
                y_a, y_b = self._run(km, smodel, y_a, y_b, stop - done, dt)
                done = stop
                if stop in record:
                    trj[record[stop]] = y_a
                    _progress(verbose, stop, n_iters)
                if rtol is not None and stop % _RTOL_PERIOD == 0 \
                        and _converged(cp, y_a, y_b, rtol):
                    stopped = stop
                    break
            self._stream.synchronize()
            smodel._y = y_a
        return stopped

    def _run(self, km, smodel, y_a, y_b, n_steps, dt):
        """Advance n_steps; returns (current, previous) state buffers."""
        # Plain steps first: they load every kernel, so none is compiled
        # or loaded while a graph is being captured.
        while n_steps > 0 and self._steps_done < self._warmup_steps:
            self._step_fused(km, smodel, y_a, y_b, dt)
            y_a, y_b = y_b, y_a
            self._steps_done += 1
            n_steps -= 1

        if self._use_graph and n_steps >= self._graph_steps:
            graph = self._get_graph(km, smodel, y_a, y_b, dt)
            if graph is not None:
                for _ in range(n_steps // self._graph_steps):
                    graph.launch(stream=self._stream)
                self._steps_done += n_steps - n_steps % self._graph_steps
                n_steps %= self._graph_steps  # even block: result in y_a

        for _ in range(n_steps):
            self._step_fused(km, smodel, y_a, y_b, dt)
            y_a, y_b = y_b, y_a
            self._steps_done += 1
        return y_a, y_b

    def _get_graph(self, km, smodel, y_a, y_b, dt):
        key = (y_a.data.ptr, y_b.data.ptr,
               tuple(w.data.ptr for w in self._work.values()), float(dt))
        if key in self._graphs:
            return self._graphs[key]
        try:
            self._stream.begin_capture()
            src, dst = y_a, y_b
            for _ in range(self._graph_steps):
                self._step_fused(km, smodel, src, dst, dt)
                src, dst = dst, src
            graph = self._stream.end_capture()
        except Exception as e:  # e.g. graphs unsupported by this setup
            warnings.warn("CUDA graph capture failed (%s); using plain "
                          "kernel launches." % e)
            self._use_graph = False
            graph = None
        self._graphs[key] = graph
        return graph

    def _pdefunc(self, km, smodel, src, dydt):
        km.launch_surface_pdefunc(
            src, dydt, smodel.model.params, smodel._nbr, smodel._wts,
            smodel._K, smodel.batch_size, smodel.n_vertices)


class SurfaceEulerSolver(SurfaceSolver):
    _name = "SurfaceEulerSolver"
    _native_code = 0
    _stability = 2.0

    def _step_generic(self, smodel, t, dt, y):
        return dt * smodel.pdefunc(t, y)

    def _step_fused(self, km, smodel, src, dst, dt):
        km.launch_surface_euler(
            src, dst, smodel.model.params, smodel._nbr, smodel._wts,
            smodel._K, smodel.batch_size, smodel.n_vertices, dt)


class SurfaceHeunSolver(SurfaceSolver):
    _name = "SurfaceHeunSolver"
    _native_code = 1
    _stability = 2.0
    _work_names = ("k1", "k2", "y_temp")

    def _step_generic(self, smodel, t, dt, y):
        k1 = dt * smodel.pdefunc(t, y)
        k2 = dt * smodel.pdefunc(t + dt, y + k1)
        return 0.5 * (k1 + k2)

    def _step_fused(self, km, smodel, src, dst, dt):
        w, n = self._work, src.size
        self._pdefunc(km, smodel, src, w["k1"])
        km.launch_rk_stage(src, w["k1"], w["y_temp"], dt, n)
        self._pdefunc(km, smodel, w["y_temp"], w["k2"])
        km.launch_linear_combine2(w["k1"], w["k2"], w["y_temp"], 0.5 * dt, 0.5 * dt, n)
        km.launch_rk_stage(src, w["y_temp"], dst, 1.0, n)


class SurfaceRungeKuttaSolver(SurfaceSolver):
    _name = "SurfaceRungeKuttaSolver"
    _native_code = 2
    _stability = 2.785
    _work_names = ("k1", "k2", "k3", "k4", "y_temp", "delta")

    def _step_generic(self, smodel, t, dt, y):
        k1 = dt * smodel.pdefunc(t, y)
        k2 = dt * smodel.pdefunc(t + 0.5 * dt, y + 0.5 * k1)
        k3 = dt * smodel.pdefunc(t + 0.5 * dt, y + 0.5 * k2)
        k4 = dt * smodel.pdefunc(t + dt, y + k3)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def _step_fused(self, km, smodel, src, dst, dt):
        w, n = self._work, src.size
        self._pdefunc(km, smodel, src, w["k1"])
        km.launch_rk_stage(src, w["k1"], w["y_temp"], 0.5 * dt, n)
        self._pdefunc(km, smodel, w["y_temp"], w["k2"])
        km.launch_rk_stage(src, w["k2"], w["y_temp"], 0.5 * dt, n)
        self._pdefunc(km, smodel, w["y_temp"], w["k3"])
        km.launch_rk_stage(src, w["k3"], w["y_temp"], dt, n)
        self._pdefunc(km, smodel, w["y_temp"], w["k4"])
        km.launch_rk4_combine(w["delta"], w["k1"], w["k2"], w["k3"], w["k4"],
                              dt / 6.0, n)
        km.launch_rk_stage(src, w["delta"], dst, 1.0, n)


class SurfaceRK23Solver(SurfaceSolver):
    _name = "SurfaceRK23Solver"
    _native_code = 3
    _stability = 2.512
    _work_names = ("k1", "k2", "k3", "y_temp", "delta")

    def _step_generic(self, smodel, t, dt, y):
        k1 = smodel.pdefunc(t, y)
        k2 = smodel.pdefunc(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = smodel.pdefunc(t + 0.75 * dt, y + 0.75 * dt * k2)
        return dt * ((2.0 / 9.0) * k1 + (1.0 / 3.0) * k2 + (4.0 / 9.0) * k3)

    def _step_fused(self, km, smodel, src, dst, dt):
        w, n = self._work, src.size
        self._pdefunc(km, smodel, src, w["k1"])
        km.launch_rk_stage(src, w["k1"], w["y_temp"], 0.5 * dt, n)
        self._pdefunc(km, smodel, w["y_temp"], w["k2"])
        km.launch_rk_stage(src, w["k2"], w["y_temp"], 0.75 * dt, n)
        self._pdefunc(km, smodel, w["y_temp"], w["k3"])
        km.launch_linear_combine3(w["k1"], w["k2"], w["k3"], w["delta"],
                                  dt * (2.0 / 9.0), dt * (1.0 / 3.0),
                                  dt * (4.0 / 9.0), n)
        km.launch_rk_stage(src, w["delta"], dst, 1.0, n)


class SurfaceAdamsBashforth2Solver(SurfaceSolver):
    _name = "SurfaceAdamsBashforth2Solver"
    _native_code = 4
    _stability = 1.0
    _work_names = ("dydt_cur", "dydt_prev", "delta")
    _warmup_steps = 2  # the first step (Euler bootstrap) uses other kernels

    def _step_generic(self, smodel, t, dt, y):
        dydt = smodel.pdefunc(t, y)
        if self._prev_dydt is None:
            delta = dt * dydt
        else:
            delta = dt * (1.5 * dydt - 0.5 * self._prev_dydt)
        self._prev_dydt = dydt
        return delta

    def _step_fused(self, km, smodel, src, dst, dt):
        w, n = self._work, src.size
        self._pdefunc(km, smodel, src, w["dydt_cur"])
        if self._steps_done == 0:
            km.launch_rk_stage(src, w["dydt_cur"], dst, dt, n)
        else:
            km.launch_linear_combine2(w["dydt_cur"], w["dydt_prev"], w["delta"],
                                      1.5 * dt, -0.5 * dt, n)
            km.launch_rk_stage(src, w["delta"], dst, 1.0, n)
        w["dydt_cur"], w["dydt_prev"] = w["dydt_prev"], w["dydt_cur"]


def _converged(xp, y_new, y_old, rtol):
    diff = float(xp.max(xp.abs(y_new - y_old)))
    scale = float(xp.max(xp.abs(y_old))) + 1e-30
    return diff / scale < rtol


def _progress(verbose, i, n_iters):
    if verbose >= 1:
        print("- [Iteration #%d / %d]" % (i, n_iters), flush=True)
