import os
from os.path import join as pjoin
import gc
import time

import numpy as np


class Solver:

    def solve(self,
              model=None,
              dt=None,
              n_iters=None,
              rtol_early_stop=None,
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
                                 init_states=model.initializer.init_states,
                                 init_pts=model.initializer.init_pts,
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

        for i in range(n_iters):
            t += self._dt

            with model.am:
                y_linear += self.step(model, t, dt, y_linear)


            # model.check_invalid_values()  # This code can be a bottleneck.

            if i == 0 or (i + 1) % period_output == 0:
                if dpath_ladybird:
                    for j in range(batch_size):
                        fpath_ladybird = pjoin(dpath_ladybird,
                                               dname_model % (j + 1),
                                               fstr_fname_ladybird % (i + 1))
                        if dpath_pattern:
                            fpath_pattern = pjoin(dpath_pattern,
                                                  dname_model % (j + 1),
                                                  fstr_fname_pattern % (i + 1))
                            model.save_image(j, fpath_ladybird, fpath_pattern)
                        else:
                            model.save_image(j, fpath_ladybird)

                # if dpath_states:
                #     for j in range(batch_size):
                #         fpath_states = pjoin(dpath_states, dname_individual%(j+1), fstr_fname_states%(i+1))
                #         self.save_states(j, fpath_states)

                if verbose >= 1:
                    print("- [Iteration #%d] elapsed time: %.5e sec." % (i + 1, time.time() - t_beg))
                    t_beg = time.time()
            # end of if

            if rtol_early_stop and model.is_early_stopping(rtol_early_stop):
                break
            # end of if

        # end of for i

        gc.collect()

        if verbose >= 1:
            print("- [Duration] : %.5e sec." % (time.time() - t_total_beg))

    # end of solve

    def step(self, model, t, dt, y_linear):
        raise NotImplementedError
