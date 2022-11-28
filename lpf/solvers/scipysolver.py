import os
from os.path import join as pjoin
import gc
import time
import warnings
warnings.filterwarnings(action='default')

import numpy as np
from scipy.integrate import solve_ivp

from lpf.solvers.solver import Solver


class ScipySolver(Solver):
    """ (Experimental) ODE solvers in SciPy
        (To be deprecated...) too slow and too complicated for this problems...
    """

    def __init__(self, method=None):

        super().__init__()

        if not method:
            method = 'RK45'

        self._method = method

        warnings.warn("Maybe deprecated in the future...", PendingDeprecationWarning)


    def solve(self,
              model=None,
              dt=None,
              n_iters=None,
              rtol=None,
              period_output=1,
              dpath_model=None,
              dpath_ladybird=None,
              dpath_pattern=None,
              dpath_states=None,
              verbose=0):

        t_total_beg = time.time()

        if not model:
            raise ValueError("model should be defined.")

        if not dt:
            dt = 0.01

        self._dt = dt

        if not n_iters:
            raise ValueError("n_iters should be defined.")

        if n_iters < 1:
            raise ValueError("n_iters should be greater than 0.")

        if not rtol:
           rtol = 1e-3

        if period_output < 1:
            raise ValueError("period_output should be greater than 0.")

        # if init_states.shape[0] != params.shape[0]:
        #     raise ValueError("The batch size of init_states and " \
        #                      "the batch size of params should be equal.")

        model.initialize()

        batch_size = model.params.shape[0]
        dname_model = "model_%0{}d".format(int(np.floor(np.log10(batch_size))) + 1)

        if dpath_model:
            fstr_fname_model \
                = "model_%0{}d.json".format(int(np.floor(np.log10(batch_size))) + 1)

            for i in range(batch_size):
                dpath_models = pjoin(dpath_model, "models")
                os.makedirs(dpath_models, exist_ok=True)
                fpath_model = pjoin(dpath_models, fstr_fname_model % (i + 1))

                model.save_model(index=i,
                                 fpath=fpath_model,
                                 initializer=model.initializer,
                                 params=model.params)
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

        # if dpath_states:
        #     for i in range(batch_size):
        #         os.makedirs(pjoin(dpath_states, dname_individual%(i+1)), exist_ok=True)
        #     # end of for
        #
        #     fstr_fname_states \
        #         = "states_%0{}d.png".format(int(np.floor(np.log10(n_iters))) + 1)

        t = 0.0
        t_beg = time.time()

        with model.am:
            y_linear = model.y_linear


        def _pdefunc(t, y):
            dydt = model.pdefunc(t, y)
            if t == 0 or (t + 1) % period_output == 0:
                if dpath_ladybird:
                    for j in range(batch_size):
                        fpath_ladybird = pjoin(dpath_ladybird,
                                               dname_model % (j + 1),
                                               fstr_fname_ladybird % (t + 1))
                        if dpath_pattern:
                            fpath_pattern = pjoin(dpath_pattern,
                                                  dname_model % (j + 1),
                                                  fstr_fname_pattern % (t + 1))
                            model.save_image(j, fpath_ladybird, fpath_pattern)
                        else:
                            model.save_image(j, fpath_ladybird)

                if verbose >= 1:
                    print("- [Iteration #%d] elapsed time: %.5e sec." % (t + 1, time.time() - self._t_beg))
                    self._t_beg = time.time()
            # end of if
            return dydt


        duration = [0, n_iters * dt]
        t_eval = np.arange(*duration)

        self._t_beg = time.time()

        sol = solve_ivp(_pdefunc, duration, y_linear,
                        method=self._method,
                        t_eval=t_eval,
                        rtol=rtol)


        gc.collect()

        if verbose >= 1:
            print("- [Duration] : %.5e sec." % (time.time() - t_total_beg))

    # end of solve
