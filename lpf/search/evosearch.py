import os
from os.path import join as pjoin
from datetime import datetime
from collections.abc import Sequence

import yaml
import numpy as np

from lpf.utils import get_hash_digest


class EvoSearch:
    def __init__(self,
                 config=None,
                 model=None,
                 solver=None,
                 converter=None,
                 targets=None,
                 objectives=None,
                 droot_output=None):
        
        self.config = config
        self.model = model
        self.solver = solver
        self.converter = converter

        if isinstance(targets, Sequence) and len(targets) < 1:
            raise ValueError("targets should be a sequence, "\
                             "which must have at least one target.")

        self.targets = targets

        self.objectives = objectives

        self.bounds_min, self.bounds_max = self.model.get_param_bounds()

        # Create a cache using dict.
        self.cache = {}

        # Create output directories.
        str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.dpath_output = pjoin(droot_output, "search_%s"%(str_now))        
        self.dpath_population = pjoin(self.dpath_output, "population")
        self.dpath_best = pjoin(self.dpath_output, "best")

        os.makedirs(self.dpath_output, exist_ok=True)
        os.makedirs(self.dpath_population, exist_ok=True)
        os.makedirs(self.dpath_best, exist_ok=True)        
        
        # Write the config file.
        fpath_config = pjoin(self.dpath_output, "config.yaml")
        with open(fpath_config, 'wt') as fout:
            yaml.dump(config, fout, default_flow_style=False)

    def fitness(self, x):
        digest = get_hash_digest(x)

        if digest in self.cache:
            arr_color = self.cache[digest]
        else:
            x = x[None, :]
            initializer = self.converter.to_initializer(x)
            params = self.converter.to_params(x)

            self.model.initializer = initializer
            self.model.params = params

            # Check constraints and ignore the decision vector if it does not satisfy.
            # if not self.model.check_constraints():
            #    return [np.inf]

            try:
                self.solver.solve(self.model)
            except (ValueError, FloatingPointError) as err:
                print("[ERROR IN FITNESS EVALUATION]", err)
                return [np.inf]

            # idx = self.model.u > self.model.thr
            #
            # if not idx.any():
            #     return [np.inf]
            # elif self.model.u.size == idx.sum():
            #     return [np.inf]
            # elif np.allclose(self.model.u[idx], self.model.u[idx].mean()):
            #     return [np.inf]

            # Colorize the ladybird model.
            arr_color = self.model.colorize()

            # Store the colorized object in the cache.
            self.cache[digest] = arr_color
        # end of if-else

        # Evaluate objectives.
        ladybird, pattern = self.model.create_image(0, arr_color)
        sum_obj = 0
        for obj in self.objectives:
            val = obj.compute(ladybird.convert("RGB"), self.targets)
            sum_obj += val

        return [sum_obj]

    def get_bounds(self):
        return (self.bounds_min, self.bounds_max)

    def save(self, 
             mode,
             dv,             
             max_generation=None,
             generation=None,
             fitness=None,
             arr_color=None):

        dv = dv[None, :]

        self.model.initializer = self.converter.to_initializer(dv)
        self.model.params = self.converter.to_params(dv)

        str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        if not generation:
            str_gen = ""
        else:
            if not max_generation:
                max_generation = 1000000
                
            fstr_gen = "gen-%0{}d_".format(int(np.ceil(np.log10(max_generation)))+1)
            str_gen = fstr_gen%(int(generation))
        
        if mode == "pop":
            fpath_model = pjoin(self.dpath_population,
                                "%smodel_%s.json"%(str_gen, str_now))    

            fpath_ladybird = pjoin(self.dpath_population,
                                   "%sladybird_%s.png"%(str_gen, str_now))

            fpath_pattern = pjoin(self.dpath_population,
                                  "%spattern_%s.png"%(str_gen, str_now))
            
        elif mode == "best":            
            fpath_model = pjoin(self.dpath_best,
                                "%smodel_%s.json"%(str_gen, str_now))
            
            fpath_ladybird = pjoin(self.dpath_best,
                                   "%sladybird_%s.png"%(str_gen, str_now))

            fpath_pattern = pjoin(self.dpath_best,
                                  "%spattern_%s.png"%(str_gen, str_now))
        else:
            raise ValueError("mode should be 'pop' or 'best'")

        if arr_color is None:
            digest = get_hash_digest(dv)            
            if digest not in self.cache:                
                try:
                    self.solver.solve(model=self.model)
                except (ValueError, FloatingPointError) as err:
                    return False
                
                arr_color = self.model.colorize()
                self.cache[digest] = arr_color
            else: 
                # Fetch the stored array from the cache.
                arr_color = self.cache[digest]
        # end of if

        self.model.save_model(index=0,
                              fpath=fpath_model,
                              initializer=self.model.initializer,
                              params=self.model.params,
                              solver=self.solver,
                              generation=generation,
                              fitness=fitness)
        
        self.model.save_image(index=0,
                              fpath_ladybird=fpath_ladybird,
                              fpath_pattern=fpath_pattern,
                              arr_color=arr_color)
            
        return True
