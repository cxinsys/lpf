from os.path import join as pjoin
import numpy as np


class ReactionDiffusionModel:
    
    def solve(self,
              init_states,
              params,
              n_iters=None,
              initializer=None,
              period_output=1,
              dpath_images=None,
              dpath_states=None,
              early_stop=False):
        
        if not n_iters:
            n_iters = self.n_iters            
        
        if initializer:
            initializer.initialize(self, init_states, params)
        else:
            self.initializer.initialize(self, init_states, params)
        
        if dpath_images:
            fstr_fname_image \
                = "img_%0{}d.png".format(int(np.floor(np.log10(n_iters))))
            
        if dpath_states:
            fstr_fname_states \
                = "states_%0{}d.png".format(int(np.floor(np.log10(n_iters))))
        
        for i in range(n_iters):
            self.t += self.dt
            self.update(i, params)
            if np.any(np.isnan(self.u)) or np.any(np.isnan(self.v)):
                raise ValueError("Invalid value occurs!")
            
            if i % period_output == 0:
                if dpath_images:                
                    fpath_image = pjoin(dpath_images, fstr_fname_image%(i+1))
                    self.save_image(fpath_image)
                    
                if dpath_states:
                    fpath_states = pjoin(dpath_states, fstr_fname_states%(i+1))
                    self.save_states(fpath_states)
            
            if early_stop:
                pass
                
            
    # end of solve
        
    def update(self):
        raise NotImplementedError()
        
    def save_image(self, fpath):
        raise NotImplementedError()
        
    def save_states(self, fpath):
        raise NotImplementedError()
        
    def save_model(self, fpath):
        raise NotImplementedError()
        
    def get_param_bounds(self):
        raise NotImplementedError()
 
