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
            
    # list_init_pts = []
    # list_init_states = []
    # list_params = []    
    
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
            
            # list_init_pts.append(init_pts)
            # list_init_states.append(init_states)
            # list_params.append(params)
                               
            h.update(list_morphs[0])
            hash_morph = h.intdigest()
            h.reset()  

            h.update(init_pts[0, ...])
            h.update(init_states[0, ...]) 
            h.update(params[0, ...])
            hash_model = h.intdigest()  # It plays a role as genotype.
            
            if hash_model not in dict_morphs[hash_morph]:
                dict_morphs[hash_morph].add(hash_model)
                dict_model_id[hash_model] = dict_fpaths
                list_dict_fpaths.append(dict_fpaths)
                n_unique += 1
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
            
            
            h.reset()            
        # end of for
    # end of for

    for fpath in fpaths_removed:
        os.remove(fpath)
    
    for dpath in dpaths_removed:
        print("[REMOVE DIR]", dpath)
        shutil.rmtree(dpath)
        
        
    
    # Get the statistics from all parameter sets.        
    n_total = len(list_dict_fpaths)

    print()
    print("The total number of unique models:", n_unique)
    print("The number of removed models:", n_redundant)

    # arr_init_pts = np.concatenate(list_init_pts, axis=0)
    # arr_init_states = np.concatenate(list_init_states, axis=0)
    # arr_params = np.concatenate(list_params, axis=0)
        
    # mean_init_pts = arr_init_pts.mean(axis=0)
    # std_init_pts = arr_init_pts.std(axis=0)
    
    # mean_init_states = arr_init_states.mean(axis=0)
    # std_init_states = arr_init_states.std(axis=0)
    # min_init_states = arr_init_states.min(axis=0)
    
    # mean_params = arr_params.mean(axis=0)
    # std_params = arr_params.std(axis=0) 
    
    # min_params = arr_params.min(axis=0)
    # max_params = arr_params.max(axis=0)
    
    # print("Mean. states:", mean_init_states)
    # print("Std. states:", std_init_states)
                
    # print("Mean. params:", mean_params)    
    # print("Std. params:", std_params)

    # print("[DEVICE]", device, end='\n\n')

    # # Perform numerical simulation
    # for epoch in range(n_epochs):
    #     ix_batch = 1
    #     for i in range(0, len(list_dict_fpaths), batch_size):
    #         t_beg = time.time()
    
    #         batch = list_dict_fpaths[i:i+batch_size]
    #         current_batch_size = len(batch)
                        
    #         model_dicts = []
    #         for dict_fpaths in batch:
    #             fpath_model = dict_fpaths["model"]
    #             with open(fpath_model, "rt") as fin:
    #                 n2v = json.load(fin)
                    
    #             model_dicts.append(n2v)
    #         # end of for
                
    #         # Create an initializer
    #         initializer = InitializerFactory.create(
    #             name=config["INITIALIZER"],
    #         )
            
    #         # Update the initializer.
    #         initializer.update(model_dicts)

    #         # Randomly generate the half of initial points.
    #         shape = (current_batch_size, *initializer.init_pts.shape[1:])
    #         init_pts_rand = np.random.normal(mean_init_pts,
    #                                          std_init_pts,
    #                                          size=shape)     
                   
    #         init_pts_rand = np.clip(init_pts_rand,
    #                                 a_min=(0, 0),
    #                                 a_max=(height-1, width-1))
            
    #         initializer.init_pts = \
    #                     np.asarray(init_pts_rand, dtype=initializer.init_pts.dtype)

    #         # Randomly generate the half of initial states.
    #         shape = (current_batch_size, initializer.init_states.shape[1])
    #         init_states_rand = np.random.normal(mean_init_states,
    #                                             std_init_states,
    #                                             size=shape)      
            
    #         init_states_rand = np.clip(init_states_rand,
    #                                    a_min=min_init_states,
    #                                    a_max=None)
            
    #         initializer.init_states = init_states_rand

    #         # Create a model.
    #         model = ModelFactory.create(
    #             name=config["MODEL"],
    #             initializer=initializer,
    #             width=width,
    #             height=height,                 
    #             dx=dx,
    #             color_u=color_u,
    #             color_v=color_v,
    #             thr_color=thr_color,
    #             device=device
    #         )
    
    #         params = model.parse_params(model_dicts)
    
    #         # Randomly generate the half of parameter sets.
    #         shape = (current_batch_size, params.shape[1])
    #         params_rand = np.random.normal(mean_params,
    #                                        std_params,
    #                                        size=shape)        
            
    #         params_rand = np.clip(params_rand,
    #                               a_min=1e-8,
    #                               a_max=None)
    
    #         print("[Batch #%d] %d models"%(ix_batch, current_batch_size), end="\n\n")        
    #         ix_batch += 1
            
    #         model.params = params_rand
            
    #         solver.solve(
    #             model=model,
    #             dt=dt,
    #             n_iters=n_iters,
    #             verbose=verbose
    #         )       
    
    #         for j in range(current_batch_size):

    #             # Check numerical errors.
    #             # Ignore this model if numerical errors has occurred.
    #             if model.is_state_invalid(index=j):
    #                 print("[Numerical error] Ignore model #%d in the batch #%d..."%(j+1, i+1))
    #                 continue

    #             img_ladybird, pattern = model.create_image(index=j)
    #             img_ladybird = img_ladybird.convert('RGB')            
    #             arr_ladybird = np.asarray(img_ladybird)
                
    #             if is_morph_invalid(img_ladybird):
    #                 print("[Invalid morph] Ignore model #%d in the batch #%d..."%(j+1, i+1))
    #                 continue
                      
    #             # Hashing ladybird image.
    #             h.update(arr_ladybird)
    #             hash_morph = h.intdigest()
    #             h.reset()                  
                
    #             # Hashing model data.
    #             h.update(init_pts_rand[j, ...])
    #             h.update(init_states_rand[j, ...])
    #             h.update(params_rand[j, ...])
    #             hash_model = h.intdigest()  # It plays a role as genotype.   
    #             h.reset()     
    
    #             if hash_model in dict_morphs[hash_morph]:
    #                 continue
                
    #             dict_morphs[hash_morph].add(hash_model)
    #             dict_model_id[hash_model] = dict_fpaths
    #             list_dict_fpaths.append(dict_fpaths)
                
    #             dpath_morph = pjoin(dpath_output_dataset, str(hash_morph))
                
    #             dpath_models = pjoin(dpath_morph, "models")
    #             dpath_ladybirds = pjoin(dpath_morph, "ladybirds")                
    #             dpath_patterns = pjoin(dpath_morph, "patterns")
    #             dpath_states = pjoin(dpath_morph, "states")                

    #             os.makedirs(dpath_models, exist_ok=True)
    #             os.makedirs(dpath_ladybirds, exist_ok=True)
    #             os.makedirs(dpath_patterns, exist_ok=True)
    #             os.makedirs(dpath_states, exist_ok=True)
                
    #             str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    #             fpath_model_new = pjoin(dpath_models,
    #                                     "model_%s_%d.json"%(str_now, j))            
    #             fpath_ladybird_new = pjoin(dpath_ladybirds,
    #                                        "ladybird_%s_%d.png"%(str_now, j))
    #             fpath_pattern_new = pjoin(dpath_patterns,
    #                                       "pattern_%s_%d.png"%(str_now, j))            
    #             fpath_states_new = pjoin(dpath_states,
    #                                      "states_%s_%d.npz"%(str_now, j))
                                
    #             model.save_model(index=j,
    #                              fpath=fpath_model_new,
    #                              initializer=initializer,
    #                              solver=solver)
                
    #             model.save_image(index=j,
    #                              fpath_ladybird=fpath_ladybird_new,
    #                              fpath_pattern=fpath_pattern_new)
                
    #             model.save_states(index=j, fpath=fpath_states_new)

    #             print("[New model]", fpath_model_new, end='\n')

    #         # end of for
    #         t_end = time.time()
        
    #         print("- [Batch duration] %f sec." % (t_end - t_beg), end="\n\n")
    #     # end of for i
    # # end of for epoch