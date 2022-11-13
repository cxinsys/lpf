import os
from os.path import join as pjoin
import time

import numpy as np

from lpf.array import get_array_module


class ReactionDiffusionModel(object):

    def __init__(self, device=None):
        self._am = get_array_module(device)

    @property
    def am(self):  # ArrayModule object
        return self._am

    @property
    def name(self):
        return self._name

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def solve(self,
              params,
              initializer=None,
              n_iters=None,
              rtol_early_stop=None,
              period_output=1,
              dpath_ladybird=None,
              dpath_pattern=None,
              dpath_states=None,
              verbose=0):

        t_total_beg = time.time()

        if not n_iters:
            n_iters = self._n_iters

        if not rtol_early_stop:
            rtol_early_stop = self._rtol_early_stop

        # if init_states.shape[0] != params.shape[0]:
        #     raise ValueError("The batch size of init_states and " \
        #                      "the batch size of params should be equal.")

        if initializer:
            initializer.initialize(self)
        else:
            self._initializer.initialize(self)

        batch_size = params.shape[0]
        dname_individual = "individual_%0{}d".format(int(np.floor(np.log10(batch_size))) + 1)

        if dpath_ladybird:
            for i in range(batch_size):
                os.makedirs(pjoin(dpath_ladybird, dname_individual%(i+1)),
                            exist_ok=True)
            # end of for

            fstr_fname_ladybird \
                = "ladybird_%0{}d.png".format(int(np.floor(np.log10(n_iters))))
                
        
        if dpath_pattern:
            for i in range(batch_size):
                os.makedirs(pjoin(dpath_pattern, dname_individual%(i+1)),
                            exist_ok=True)
            # end of for

            fstr_fname_pattern \
                = "pattern_%0{}d.png".format(int(np.floor(np.log10(n_iters))))
            
        if dpath_states:
            for i in range(batch_size):
                os.makedirs(pjoin(dpath_states, dname_individual%(i+1)),
                            exist_ok=True)
            # end of for

            fstr_fname_states \
                = "states_%0{}d.png".format(int(np.floor(np.log10(n_iters))))

        params = self.am.array(params, dtype=params.dtype)

        t_beg = time.time()
        for i in range(n_iters):
            self.t += self._dt
            self.update(i, params)
            self.check_invalid_values()

            if (i+1) % period_output == 0:
                if dpath_ladybird:
                    for j in range(batch_size):
                        fpath_ladybird = pjoin(dpath_ladybird,
                                               dname_individual%(j+1),
                                               fstr_fname_ladybird%(i+1))
                        if dpath_pattern:
                            fpath_pattern = pjoin(dpath_pattern,
                                                  dname_individual%(j+1),
                                                  fstr_fname_pattern%(i+1))
                            self.save_image(j, fpath_ladybird, fpath_pattern)
                        else:
                            self.save_image(j, fpath_ladybird)
                                    
                if dpath_states:
                    for j in range(batch_size):
                        fpath_states = pjoin(dpath_states, dname_individual%(j+1), fstr_fname_states%(i+1))
                        self.save_states(j, fpath_states)

                if verbose >= 1:
                    print("- [Iteration #%d] elapsed time: %.5e sec."%(i+1, time.time() - t_beg))
                    t_beg = time.time()

            if rtol_early_stop and self.is_early_stopping(rtol_early_stop):
                break
        # end of for

        if verbose >= 1:
            print("- [Duration] : %.5e sec." % (time.time() - t_total_beg))

    # end of solve
        
    def update(self):
        raise NotImplementedError()
        
    def is_early_stopping(self, rtol):       
        raise NotImplementedError()
        
    def save_image(self, i, fpath):
        raise NotImplementedError()
        
    def save_states(self, i, fpath):
        raise NotImplementedError()
        
    def save_model(self, i, fpath):
        raise NotImplementedError()
        
    def get_param_bounds(self):
        raise NotImplementedError()

    def check_invalid_values(self):
        raise NotImplementedError()
