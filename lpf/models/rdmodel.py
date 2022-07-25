import os
from os.path import join as pjoin
import numpy as np


class ReactionDiffusionModel:
    
    def solve(self,
              init_states,
              param_batch,
              n_iters=None,
              rtol_early_stop=None,
              initializer=None,
              period_output=1,
              dpath_images=None,
              dpath_states=None):
        
        if not n_iters:
            n_iters = self.n_iters 

        if not rtol_early_stop:
            rtol_early_stop = self.rtol_early_stop
            
            
        if init_states.shape[0] != param_batch.shape[0]:
            raise ValueError("The batch size of init_states and " \
                             "the batch size of params should be equal.")

        
        if initializer:
            initializer.initialize(self, init_states, param_batch)
        else:
            self.initializer.initialize(self, init_states, param_batch)

        batch_size = init_states.shape[0]
        dname_individual = "individual_%0{}d".format(int(np.floor(np.log10(batch_size))))

        if dpath_images:
            for i in range(batch_size):
                os.makedirs(pjoin(dpath_images, dname_individual%(i+1)), exist_ok=True)
            # end of for

            fstr_fname_image \
                = "img_%0{}d.png".format(int(np.floor(np.log10(n_iters))))
            
        if dpath_states:
            for i in range(batch_size):
                os.makedirs(pjoin(dpath_images, dname_individual%(i+1)), exist_ok=True)
            # end of for

            fstr_fname_states \
                = "states_%0{}d.png".format(int(np.floor(np.log10(n_iters))))
        
        for i in range(n_iters):
            self.t += self.dt
            self.update(i, param_batch)
            
            if np.any(np.isnan(self.u)) or np.any(np.isnan(self.v)):
                raise ValueError("Invalid value occurs!")
            
            if i % period_output == 0:
                if dpath_images:
                    for j in range(batch_size):
                        fpath_image = pjoin(dpath_images, dname_individual%(j+1), fstr_fname_image%(i+1))
                        self.save_image(fpath_image)
                    
                if dpath_states:
                    for j in range(batch_size):
                        fpath_states = pjoin(dpath_states, dname_individual%(j+1), fstr_fname_states%(i+1))
                        self.save_states(fpath_states)

            if rtol_early_stop and self.is_early_stopping(rtol_early_stop):
                break

    # end of solve

        
    def update(self):
        raise NotImplementedError()
        
    def is_early_stopping(self, rtol):       
        raise NotImplementedError()
        
    def save_image(self, fpath):
        raise NotImplementedError()
        
    def save_states(self, fpath):
        raise NotImplementedError()
        
    def save_model(self, fpath):
        raise NotImplementedError()
        
    def get_param_bounds(self):
        raise NotImplementedError()
 
