import os
import os.path as osp
from os.path import join as pjoin
import shutil
import time
from datetime import datetime
import shutil
import json
import argparse
from collections import defaultdict


import yaml

import numpy as np
np.seterr(all='raise')

from PIL import Image
from PIL import ImageColor
import xxhash

from lpf.initializers import InitializerFactory
from lpf.models import ModelFactory
from lpf.solvers import SolverFactory
from lpf.utils import is_param_invalid
from lpf.utils import is_morph_invalid


def get_data(batch, initializer, model):
    model_dicts = []
    list_morphs = []
    
    for dict_fpaths in batch:
        fpath_model = dict_fpaths["model"]
        with open(fpath_model, "rt") as fin:
            n2v = json.load(fin)
        model_dicts.append(n2v)
        
        fpath_ladybird = dict_fpaths["ladybird"]
        with Image.open(fpath_ladybird) as img:                
            img_rgb = img.convert('RGB')            
            list_morphs.append(np.asarray(img_rgb))
    # end of for   

    # Create initializer       
    initializer = InitializerFactory.create(
        name=initializer,
    )
    
    # Update the initializer.
    initializer.update(model_dicts)
    
    # Create a model.
    model = ModelFactory.create(
        name=model,
        initializer=initializer,
    )    
    
    params = model.parse_params(model_dicts)
    
    return (list_morphs,
            initializer.init_pts,
            initializer.init_states,
            params)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Parse the options.'
    )
    
    parser.add_argument('--dir',
                        dest='dpath_dataset',
                        action='store',
                        type=str,
                        help="Designate the dataset directory.")
    
    args = parser.parse_args()

    dpath_dataset = args.dpath_dataset
    if not os.path.isdir(dpath_dataset):
        raise FileNotFoundError("This is not a directory: %s"%(dpath_dataset))
    

    # Collect model files within dpath_dataset.
    list_dict_fpaths = []
    fpaths_removed = []
    dpaths_removed = []
    
    
    dict_morphs = defaultdict(set)
    dict_model_id = {}
    
    h = xxhash.xxh64()

    n_unique = 0
    n_redundant = 0
    
    for dname in os.listdir(dpath_dataset):
        dpath_morph = pjoin(dpath_dataset, dname)
        
        if not osp.isdir(dpath_morph):
            continue            
                
        dpath_models = pjoin(dpath_morph, "models")
        dpath_ladybirds = pjoin(dpath_morph, "ladybirds")
        dpath_patterns = pjoin(dpath_morph, "patterns")
        dpath_states = pjoin(dpath_morph, "states")
            
        for fname in os.listdir(dpath_models):
            
            fpath_model = osp.join(dpath_models, fname)
            if not osp.isfile(fpath_model) \
                or not fname.startswith("model_"):
                continue
            
            # Get the model ID
            fname, ext = osp.splitext(fname)
            items = fname.split('_')        
            model_id = '_'.join(items[1:])
            
            fpath_ladybird = osp.join(dpath_ladybirds,
                                      "ladybird_%s.png"%(model_id))
            
            fpath_pattern = osp.join(dpath_patterns,
                                     "pattern_%s.png"%(model_id))
            
            fpath_states = osp.join(dpath_states,
                                    "states_%s.npz"%(model_id))
            
            
            if not osp.isfile(fpath_ladybird):
                raise FileNotFoundError(fpath_ladybird)
            
            if not osp.isfile(fpath_pattern):
                raise FileNotFoundError(fpath_pattern)
                
            if not osp.isfile(fpath_states):
                raise FileNotFoundError(fpath_states)
                
            
            dict_fpaths = {"model": fpath_model,
                           "ladybird": fpath_ladybird,
                           "pattern": fpath_pattern,
                           "states": fpath_states}
            
            # Hash this model.
            list_morphs, init_pts, init_states, params \
                                = get_data([dict_fpaths],
                                           "LiawInitializer",
                                           "LiawModel")
                               
            h.update(list_morphs[0])
            hash_morph = h.intdigest()
            h.reset()  

            h.update(init_pts[0, ...])
            h.update(init_states[0, ...]) 
            h.update(params[0, ...])
            hash_model = h.intdigest()  # It plays a role as genotype.
            h.reset()            
            
            if hash_model not in dict_morphs[hash_morph]:
                dict_morphs[hash_morph].add(hash_model)
                dict_model_id[hash_model] = dict_fpaths
                list_dict_fpaths.append(dict_fpaths)
                n_unique += 1
                print("[UNIQUE MODEL #%d] %s"%(n_unique, fpath_model))
            else:
                os.remove(fpath_model)
                os.remove(fpath_ladybird)
                os.remove(fpath_pattern)
                fpaths_removed.append(fpath_states)
                
                entities = os.listdir(dpath_ladybirds)
                if len(entities) == 0:                
                    dpaths_removed.append(dpath_morph)
                
                n_redundant += 1
                print("[REDUNDANT MODEL #%d] %s"%(n_redundant, fpath_model))
            
            
        # end of for
    # end of for

    for fpath in fpaths_removed:
        os.remove(fpath)
    
    for dpath in dpaths_removed:
        print("[REMOVE DIR]", dpath)
        shutil.rmtree(dpath)
        
        
    
    # Get the statistics from all parameter sets.        
    # n_total = len(list_dict_fpaths)

    print()
    print("The total number of unique models:", n_unique)
    print("The number of removed models:", n_redundant)

