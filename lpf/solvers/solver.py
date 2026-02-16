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
              verbose=0):

        t_total_beg = time.time()

        if model is None:
            if self._model is None:
                raise ValueError("model should be defined.")
            model = self._model

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


        if iter_end is None:

            if n_iters is None:
                if self._n_iters is None:
                    raise ValueError("n_iters should be defined.")
                n_iters = self._n_iters
                
            if n_iters < 1:
                raise ValueError("n_iters should be greater than or equal to 1.")

            iter_end = iter_begin + n_iters
        elif iter_end < 1:
            raise ValueError("iter_end should be greater than or equal to 1.")


        if get_trj:
            with model.am:
                if hasattr(self, "_trj_y"):
                    del self._trj_y
                    
                n_time_points = int((iter_end - iter_begin) // period_output + 1)
                
                shape_trj = (n_time_points, *model.shape_grid)
                self._trj_y = model.am.zeros(shape_trj, dtype=model.y_mesh.dtype)
                
                

        t = 0.0
        t_beg = time.time()

        # with model.am:
        #     y_mesh = model.y_mesh

        ix_trj = 0
        for i in range(iter_begin, iter_end, 1):
            t += dt

            with model.am:
                delta_y = self.step(model, t, dt, model.y_mesh)
                model.y_mesh = model.y_mesh + delta_y

            if period_output is None:
                pass
            elif i == iter_begin or (i + 1) % period_output == 0:
                if get_trj:
                    self._trj_y[ix_trj, ...] = model.y_mesh
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

            if rtol is not None and model.is_early_stopping(rtol):
                break
            # end of if

        # end of for i

        gc.collect()

        if verbose >= 1:
            print("- [Duration] : %.5e sec." % (time.time() - t_total_beg))

        if get_trj:
            return self._trj_y

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
