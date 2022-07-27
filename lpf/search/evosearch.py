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

from lpf.utils import get_module_dpath
from lpf.utils import get_hash_digest


class EvoSearch:
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
        self.len_dv = self.model.get_len_dv()
        
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

    def fitness(self, dv):
        return [100000.0]

    def has_batch_fitness(self):
        return True

    def batch_fitness(self, dvs):
        # digest = get_hash_digest(dvs)
        dvs = dvs.reshape(-1, self.len_dv)
        batch_size = dvs.shape[0]

        if False: #digest in self.cache:
            arr_color = self.cache[digest]
        else:
            params = self.converter.to_params(dvs)
            init_states = self.converter.to_init_states(dvs)        
            initializer = self.converter.to_initializer(dvs)            
            self.initializer = initializer

            print("type(params):", type(params))
            print("type(init_states):", type(init_states))
            
            try:
                self.model.solve(init_states,
                                 params,
                                 initializer=initializer)
            except (ValueError, FloatingPointError) as err:
                print("[ERROR]", err)
                #raise err
                return np.full(batch_size, np.inf)


            # [TODO] have to remove the exposed u and v variables
            # idx = self.model.u > self.model.thr
            #
            # if not idx.any():
            #     return [np.inf]
            # elif self.model.u.size == idx.sum():
            #     return [np.inf]
            # elif np.allclose(self.model.u[idx], self.model.u[idx].mean()):
            #     return [np.inf]
            #
            # #################################################################

            # Colorize the ladybird model.
            arr_color = self.model.colorize()    
                    
            # Store the colored object in the cache.
            # self.cache[digest] = arr_color[0, :, :]
               
        # Evaluate objectives.
        fvs = np.zeros((batch_size,), dtype=np.float64)
        for i,  in enumerate(arr_color):
            img = self.model.create_image(i, arr_color)
            sum_obj = 0
            for obj in self.objectives:
                val = obj.compute(img.convert("RGB"), self.targets)
                sum_obj += val

            fvs[i] = sum_obj

        print("fvs.shape:", fvs.shape)
        return fvs

    def get_bounds(self):        
        return (self.bounds_min, self.bounds_max)

        
    def save(self, 
             mode,
             dvs,             
             generation=None,
             fitness=None,
             arr_color=None):                

        params = self.converter.to_params(dvs)
        init_states = self.converter.to_init_states(dvs)
        init_pts = self.converter.to_init_pts(dvs)        
        
        initializer = self.converter.to_initializer(dvs)            
        # self.model._initializer = initializer
        
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
            digest = get_hash_digest(dvs)            
            if digest not in self.cache:                
                try:
                    self.model.solve(init_states=init_states,
                                     params=params,
                                     initializer=initializer)
                    
                except (ValueError, FloatingPointError):
                    return False
                
                arr_color = self.model.colorize() 
            else: 
                # Fetch the stored array from the cache.
                arr_color = self.cache[digest]
        # end of if

        # [TODO]
        # self.model.save_model(fpath_model,
        #                       init_states,
        #                       init_pts,
        #                       params,
        #                       generation=generation,
        #                       fitness=fitness)
        
        self.model.save_image(fpath_image, arr_color)
            
        return True
