import os
from os.path import join as pjoin
import gc
import time

import numpy as np


class Solver:

    def __init__(self,
                 model=None,
                 dt=None,
                 n_iters=None,
                 rtol=None,
                 period_output=None,
                 dpath_model=None,
                 dpath_morph=None,
                 dpath_pattern=None,
                 dpath_states=None,
                 verbose=None):

        self._name = None
        self._model = model
        self._dt = dt
        self._n_iters = n_iters
        self._rtol = rtol
        self._period_output = period_output
        self._dpath_model = dpath_model
        self._dpath_morph = dpath_morph
        self._dpath_pattern = dpath_pattern
        self._dpath_states = dpath_states
        self._verbose = verbose

    @property
    def name(self):
        return self._name

    @property
    def trj_y(self):
        return self._trj_y

    def solve(self,
              model=None,
              dt=None,
              n_iters=None,
              rtol=None,
              period_output=None,
              dpath_model=None,
              dpath_morph=None,
              dpath_pattern=None,
              dpath_states=None,
              init_model=True,
              iter_begin=0,
              iter_end=None,
              get_trj=False,
              trj_waypoints=None,
              verbose=0):
        """Run the solver loop.

        Note: ``iter_begin`` only affects iteration numbering for output
        and trajectory capture.  The simulation clock always starts at
        ``t = 0`` because all built-in reaction models are autonomous
        (time-independent).  CUDA kernels do not accept a time parameter.
        """

        t_total_beg = time.time()

        if model is None:
            if self._model is None:
                raise ValueError("model should be defined.")
            model = self._model

        # Clear stale trajectory from previous solve
        if not get_trj and trj_waypoints is None:
            if hasattr(self, '_trj_y'):
                del self._trj_y

        # ---- CUDA auto-dispatch ----
        cuda_solver = self._get_cuda_solver() if self._is_cuda(model) else None
        if cuda_solver is not None:
            return self._forward_to_cuda(
                model, dt, n_iters, rtol, period_output,
                dict(dpath_model=dpath_model, dpath_morph=dpath_morph,
                     dpath_pattern=dpath_pattern, dpath_states=dpath_states,
                     init_model=init_model, iter_begin=iter_begin,
                     iter_end=iter_end, get_trj=get_trj,
                     trj_waypoints=trj_waypoints, verbose=verbose))

        if dt is None:
            if self._dt is None:
                self._dt = dt = 0.01
            dt = self._dt

        if rtol is None:
            if self._rtol is not None:
                rtol = self._rtol

        if rtol is not None and rtol < 0:
            raise ValueError("rtol should be greater than 0.")

        if period_output is None:
            if self._period_output is not None:
                period_output = self._period_output

        if period_output is not None and period_output < 1:
            raise ValueError("period_output should be greater than 0.")

        if not model.has_initializer():
            raise ValueError("model should have an initializer.")

        if init_model:
            model.initialize()

        # ---- resolve n_iters / iter_end BEFORE file I/O setup ----
        if iter_end is None:
            if n_iters is None:
                if self._n_iters is None:
                    raise ValueError("n_iters should be defined.")
                n_iters = self._n_iters

            if n_iters < 1:
                raise ValueError("n_iters should be greater than or equal to 1.")

            iter_end = iter_begin + n_iters
        else:
            if iter_end <= iter_begin:
                raise ValueError(
                    f"iter_end ({iter_end}) must be greater than "
                    f"iter_begin ({iter_begin}).")
            # Derive n_iters from iter_end when not explicitly given
            if n_iters is None:
                n_iters = iter_end - iter_begin

        batch_size = model.batch_size # model.params.shape[0]
        dname_model = "model_%0{}d".format(int(np.floor(np.log10(batch_size))) + 1)

        if dpath_model:
            fstr_fname_model \
                = "model_%0{}d.json".format(int(np.floor(np.log10(batch_size))) + 1)

            dict_solver = self.to_dict()
            dict_solver["solver"] = self.name
            dict_solver["dt"] = dt
            dict_solver["n_iters"] = n_iters

            if rtol is not None:
                dict_solver["rtol"] = rtol

            for i in range(batch_size):
                dpath_models = pjoin(dpath_model, "models")
                os.makedirs(dpath_models, exist_ok=True)
                fpath_model = pjoin(dpath_models, fstr_fname_model % (i + 1))

                model.save_model(index=i,
                                 fpath=fpath_model,
                                 initializer=model.initializer,
                                 solver=dict_solver)
            # end of for

        if dpath_morph:
            for i in range(batch_size):
                os.makedirs(pjoin(dpath_morph, dname_model % (i + 1)), exist_ok=True)
            # end of for

            fstr_fname_morph \
                = "morph_%0{}d.png".format(int(np.floor(np.log10(n_iters))) + 1)

        if dpath_pattern:
            for i in range(batch_size):
                os.makedirs(pjoin(dpath_pattern, dname_model % (i + 1)), exist_ok=True)
            # end of for

            fstr_fname_pattern \
                = "pattern_%0{}d.png".format(int(np.floor(np.log10(n_iters))) + 1)

        if dpath_states:
            for i in range(batch_size):
                os.makedirs(pjoin(dpath_states, dname_model%(i+1)), exist_ok=True)
            # end of for

            fstr_fname_states \
                = "states_%0{}d".format(int(np.floor(np.log10(n_iters))) + 1)


        # Normalize trj_waypoints to a set for O(1) lookup
        if trj_waypoints is not None:
            # Filter to valid range [iter_begin+1, iter_end]
            valid_range = range(iter_begin + 1, iter_end + 1)
            waypoints_set = set(w for w in trj_waypoints if w in valid_range)
            get_trj = True
        else:
            waypoints_set = None

        # Default: capture every iteration when get_trj=True without guidance
        if get_trj and period_output is None and waypoints_set is None:
            period_output = 1

        if get_trj:
            with model.am:
                if hasattr(self, "_trj_y"):
                    del self._trj_y

                if waypoints_set is not None:
                    n_time_points = len(waypoints_set)
                    self._trj_iters = sorted(waypoints_set)
                else:
                    # Count exactly how many iterations match the recording condition
                    n_time_points = sum(
                        1 for i in range(iter_begin, iter_end)
                        if i == iter_begin or (i + 1) % period_output == 0)
                    self._trj_iters = None

                shape_trj = (n_time_points, *model.shape_grid)
                self._trj_y = model.am.zeros(shape_trj, dtype=model.y_mesh.dtype)

                

        t = iter_begin * dt  # resume from correct time offset
        t_beg = time.time()

        # with model.am:
        #     y_mesh = model.y_mesh

        ix_trj = 0
        for i in range(iter_begin, iter_end, 1):
            with model.am:
                delta_y = self.step(model, t, dt, model.y_mesh)
                # Cache pre-update scale for early stopping (matches CUDA: |old|)
                if rtol is not None:
                    _y_scale = float(model.am.get(
                        model.am.abs(model.y_mesh).max()))
                model.y_mesh = model.y_mesh + delta_y
            t += dt

            # Determine whether to capture at this iteration
            if waypoints_set is not None:
                should_record = (i + 1) in waypoints_set
            elif period_output is None:
                should_record = False
            else:
                should_record = (i == iter_begin or (i + 1) % period_output == 0)

            if should_record:
                if get_trj:
                    self._trj_y = model.am.set(
                        self._trj_y, ix_trj, model.y_mesh)
                    ix_trj += 1

                if dpath_morph or dpath_pattern:
                    for j in range(batch_size):
                        fpath_morph_j = None
                        if dpath_morph:
                            fpath_morph_j = pjoin(dpath_morph,
                                                  dname_model % (j + 1),
                                                  fstr_fname_morph % (i + 1))

                        fpath_pattern_j = None
                        if dpath_pattern:
                            fpath_pattern_j = pjoin(dpath_pattern,
                                                    dname_model % (j + 1),
                                                    fstr_fname_pattern % (i + 1))

                        model.save_image(j, fpath_morph_j, fpath_pattern_j)

                if dpath_states:
                    for j in range(batch_size):
                        fpath_states = pjoin(dpath_states,
                                             dname_model % (j + 1),
                                             fstr_fname_states%(i + 1))

                        model.save_states(j, fpath_states)

                if verbose >= 1:
                    print("- [Iteration #%d] elapsed time: %.5e sec." % (i + 1, time.time() - t_beg))
                    t_beg = time.time()
            # end of if

            # Early stopping: relative change metric (same as CUDA path)
            # Uses pre-update |y| as denominator, checked every 1000 steps.
            if rtol is not None and (i + 1) % 1000 == 0:
                with model.am:
                    rel_change = float(model.am.get(
                        model.am.abs(delta_y).max()
                    )) / (_y_scale + 1e-30)
                if rel_change < rtol:
                    break

        # end of for i

        gc.collect()

        if verbose >= 1:
            print("- [Duration] : %.5e sec." % (time.time() - t_total_beg))

        if get_trj:
            # Trim trajectory buffer to actual captured count
            trj = self._trj_y[:ix_trj] if ix_trj < self._trj_y.shape[0] else self._trj_y
            if self._trj_iters is not None:
                return {"iters": self._trj_iters[:ix_trj], "trj": trj}
            return trj

    # end of solve

    def step(self, model, t, dt, y_mesh):
        raise NotImplementedError

    def to_dict(self):
        n2v = {}  # Mapping variable names to values.
        n2v["solver"] = self.name
        n2v["dt"] = self._dt
        n2v["n_iters"] = self._n_iters

        if self._rtol is not None:
            n2v["rtol"] = self._rtol

        return n2v

    # --- CUDA auto-dispatch (shared by all subclasses) ---

    @staticmethod
    def _is_cuda(model):
        """True if model is on a CUDA device (CuPy or PyTorch CUDA)."""
        from lpf.array.module import CupyModule, TorchModule
        if model is None:
            return False
        if isinstance(model.am, CupyModule):
            return True
        if isinstance(model.am, TorchModule):
            return hasattr(model.am, "_device") and "cuda" in str(model.am._device)
        return False

    def _make_cuda_solver(self):
        """Override in subclass to return a Cu* solver instance."""
        return None

    def _get_cuda_solver(self):
        """Lazy-create the CUDA solver implementation."""
        if not hasattr(self, '_cuda_impl'):
            self._cuda_impl = None
        if self._cuda_impl is None:
            self._cuda_impl = self._make_cuda_solver()
        return self._cuda_impl

    def _forward_to_cuda(self, model, dt, n_iters, rtol, period_output, kwargs):
        """Forward stored params to the Cu* solver's solve()."""
        try:
            result = self._get_cuda_solver().solve(
                model=model,
                dt=dt if dt is not None else self._dt,
                n_iters=n_iters if n_iters is not None else self._n_iters,
                rtol=rtol if rtol is not None else self._rtol,
                period_output=(period_output if period_output is not None
                               else self._period_output),
                **kwargs)
        except ValueError as e:
            if "Unsupported model" in str(e):
                raise ValueError(
                    f"{e}  Use device='cpu' for models without CUDA kernel support."
                ) from None
            raise
        # Store trajectory on outer solver for solver.trj_y API compatibility
        if isinstance(result, dict) and 'trj' in result:
            self._trj_y = result['trj']
        elif hasattr(result, 'shape'):
            self._trj_y = result
        return result
