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
                 dpath_ladybird=None,
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
        self._dpath_ladybird = dpath_ladybird
        self._dpath_pattern = dpath_pattern
        self._dpath_states = dpath_states
        self._verbose = verbose

    @property
    def name(self):
        return self._name

    def solve(self,
              model=None,
              dt=None,
              n_iters=None,
              rtol=None,
              period_output=None,
              dpath_model=None,
              dpath_ladybird=None,
              dpath_pattern=None,
              dpath_states=None,
              verbose=0):

        t_total_beg = time.time()

        if not model:
            if not self._model:
                raise ValueError("model should be defined.")
            model = self._model

        if not dt:
            if not self._dt:
                self._dt = dt = 0.01
            dt = self._dt

        if not n_iters:
            if not self._n_iters:
                raise ValueError("n_iters should be defined.")
            n_iters = self._n_iters

        if n_iters < 1:
            raise ValueError("n_iters should be greater than or equal to 1.")

        if not rtol:
            if self._rtol:
                rtol = self._rtol

        if rtol and rtol < 0:
            raise ValueError("rtol should be greater than 0.")

        if not period_output:
            if self._period_output:
                period_output = self._period_output

        if period_output is not None and period_output < 1:
            raise ValueError("period_output should be greater than 0.")

        if not model.has_initializer():
            raise ValueError("model should have an initializer.")

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

            if rtol:
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

        if dpath_ladybird:
            for i in range(batch_size):
                os.makedirs(pjoin(dpath_ladybird, dname_model % (i + 1)), exist_ok=True)
            # end of for

            fstr_fname_ladybird \
                = "ladybird_%0{}d.png".format(int(np.floor(np.log10(n_iters))) + 1)

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

        t = 0.0
        t_beg = time.time()

        with model.am:
            y_mesh = model.y_mesh

        for i in range(n_iters):
            t += dt

            with model.am:
                y_mesh += self.step(model, t, dt, y_mesh)

            if not period_output:
                pass
            elif i == 0 or (i + 1) % period_output == 0:
                if dpath_ladybird:
                    for j in range(batch_size):
                        fpath_ladybird = pjoin(dpath_ladybird,
                                               dname_model % (j + 1),
                                               fstr_fname_ladybird % (i + 1))

                        fpath_pattern = None
                        if dpath_pattern:
                            fpath_pattern = pjoin(dpath_pattern,
                                                  dname_model % (j + 1),
                                                  fstr_fname_pattern % (i + 1))

                        model.save_image(j, fpath_ladybird, fpath_pattern)

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

            if rtol and model.is_early_stopping(rtol):
                break
            # end of if

        # end of for i

        gc.collect()

        if verbose >= 1:
            print("- [Duration] : %.5e sec." % (time.time() - t_total_beg))

    # end of solve

    def step(self, model, t, dt, y_mesh):
        raise NotImplementedError

    def to_dict(self):
        n2v = {}  # Mapping variable names to values.
        n2v["solver"] = self.name
        n2v["dt"] = self._dt
        n2v["n_iters"] = self._n_iters
        n2v["rtol"] = self._rtol

        return n2v
