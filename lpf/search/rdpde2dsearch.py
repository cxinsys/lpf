import time
import json
import os
import os.path as osp
from os.path import join as pjoin
from os.path import abspath as apath
import shutil
from datetime import datetime
import argparse

import yaml
import numpy as np
import pygmo as pg
from PIL import Image
import matplotlib.pyplot as plt

from lpf.utils import get_module_dpath
from lpf.utils import get_hash_digest


class RdPde2dSearch:
    def __init__(self,
                 config,
                 model,
                 converter,                 
                 targets,
                 objectives,
                 droot_output=None):
        
        # Load hyperparameters.
        self.config = config
        self.model = model
        self.converter = converter
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
            yaml.dump(config, fout)
            
        
    def fitness(self, x):
        params = self.converter.to_params(x)
        init_states = self.converter.to_init_states(x)        
        initializer = self.converter.to_initializer(x)
        
        self.initializer = initializer
        
        try:
            self.model.solve(init_states,
                             params=params,
                             initializer=initializer)
        except (ValueError, FloatingPointError) as err:
            return [np.inf]
                        
        idx = self.model.u > self.model.thr

        if not idx.any():
            return [np.inf]        
        elif self.model.u.size == idx.sum():
            return [np.inf]
        elif np.allclose(self.model.u[idx], self.model.u[idx].mean()):
            return [np.inf]

        # Colorize the ladybird model.
        arr_color = self.model.colorize()    
                
        # Store the colored object in the cache.
        digest = get_hash_digest(x)
        self.cache[digest] = arr_color         
               
        # Evaluate objectives.
        ladybird = self.model.create_image(arr_color)        
        sum_obj = 0
        for obj in self.objectives:
            val = obj.compute(ladybird.convert("RGB"), self.targets)
            sum_obj += val


        return [sum_obj]

    def get_bounds(self):        
        return (self.bounds_min, self.bounds_max)

        
    def save(self, 
             mode,
             x,             
             generation=None,
             fitness=None,
             arr_color=None):                

        params = self.converter.to_params(x)
        init_states = self.converter.to_init_states(x)
        init_pts = self.converter.to_init_pts(x)        
        
        str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
        if mode == "pop":
            fpath_model = pjoin(self.dpath_population,
                                "model_%s.json"%(str_now))            
            fpath_image = pjoin(self.dpath_population,
                                "image_%s.png"%(str_now))        
            
        elif mode == "best":            
            fpath_model = pjoin(self.dpath_best,
                                "model_%s.json"%(str_now))            
            fpath_image = pjoin(self.dpath_best,
                                "image_%s.png"%(str_now))        
        else:
            raise ValueError("mode should be 'pop' or 'best'")
            
        
        
        if arr_color is None:            
            digest = get_hash_digest(x)            
            if digest not in self.cache:                
                try:
                    initializer = self.converter.to_initializer(x)
                    self.model.solve(init_states=init_states,
                                     params=params,
                                     initializer=initializer)
                    
                except (ValueError, FloatingPointError):
                    return False
                
                arr_color = self.model.colorize() 
            # end of if
            
            # Fetch the stored array from the cache.
            arr_color = self.cache[digest]
        # end of if
            
        self.model.save_model(fpath_model,
                              init_states,
                              init_pts,
                              params,
                              generation=generation,
                              fitness=fitness)
        
        self.model.save_image(fpath_image, arr_color)
            
        return True
