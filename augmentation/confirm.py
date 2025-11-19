import os
import os.path as osp
from os.path import join as pjoin
import time
from datetime import datetime
import shutil
import json
import argparse
from collections import defaultdict

import random
import multiprocessing
from multiprocessing import Process

import yaml

import numpy as np
np.seterr(all='raise')

from PIL import Image
from PIL import ImageColor
import xxhash

from lpf.initializers import InitializerFactory
from lpf.models import ModelFactory
from lpf.solvers import SolverFactory



def parse_devices(device):
    if isinstance(device, str):
        devices = device.split(',')            
        if len(devices) <= 1:
            device = str(device).lower()
            
        devices = [_device.lower() for _device in devices]
        
    if isinstance(device, list):
        devices = [_device.lower() for _device in device]
        
        
    return devices



def get_data(config, batch):
    model_dicts = []
    list_morphs = []
    
    for dict_fpaths in batch:
        fpath_model = dict_fpaths["model"]
        with open(fpath_model, "rt") as fin:
            n2v = json.load(fin)
        model_dicts.append(n2v)
        
        fpath_img_morph = dict_fpaths["morph"]
        with Image.open(fpath_img_morph) as img:                
            img_rgb = img.convert('RGB')            
            list_morphs.append(np.asarray(img_rgb))
    # end of for   

    # Create initializer       
    initializer = InitializerFactory.create(
        name=config["INITIALIZER"],
    )
    
    # Update the initializer.
    initializer.update(model_dicts)
    
    # Create a model.
    model = ModelFactory.create(
        name=config["MODEL"],
        initializer=initializer,
    )    
    
    params = model.parse_params(model_dicts)
    
    return (list_morphs,
            initializer.init_pts,
            initializer.init_states,
            params)



def solve_batches(config, device, list_dict_fpaths):
    print("[DEVICE]", device, end='\n\n')

    dpath_output_dataset = config["DPATH_OUTPUT_DATASET"]

    batch_size = int(config["BATCH_SIZE"])
    
    dx = float(config["DX"])
    dt = float(config["DT"])
    width = int(config["WIDTH"])
    height = int(config["HEIGHT"])
    thr = float(config["THR_COLOR"])
    n_iters = int(config["N_ITERS"])
    n_init_pts = int(config["N_INIT_PTS"])

    color_u = None
    color_v = None
    thr_color = None

    if "COLOR_U" in config:
        color_u = ImageColor.getcolor(config["COLOR_U"], "RGB")

    if "COLOR_V" in config:
        color_v = ImageColor.getcolor(config["COLOR_V"], "RGB")

    if "THR_COLOR" in config:
        thr_color = float(config["THR_COLOR"])

    # Create a solver.
    solver = SolverFactory.create(name=config["SOLVER"],
                                  dt=float(config["DT"]),
                                  n_iters=int(config["N_ITERS"]))
    
    # Create a model.
    model = ModelFactory.create(
        name=config["MODEL"],
        width=width,
        height=height,                 
        dx=dx,
        color_u=color_u,
        color_v=color_v,
        thr_color=thr_color,
        device=device
    )

    h = xxhash.xxh64()
   
    ix_batch = 1
    for i in range(0, len(list_dict_fpaths), batch_size):
        t_beg = time.time()
        
        batch = list_dict_fpaths[i:i+batch_size]
        
        current_batch_size = len(batch)        
        model_dicts = []
        for dict_fpaths in batch:
            fpath_model = dict_fpaths["model"]
            with open(fpath_model, "rt") as fin:
                n2v = json.load(fin)
                
            model_dicts.append(n2v)
        # end of for
            
        # Create an initializer
        initializer = InitializerFactory.create(
            name=config["INITIALIZER"],
        )
        
        # Update the initializer.
        initializer.update(model_dicts)

        model.initializer = initializer

        model.params = model.parse_params(model_dicts)
        model.initialize()
       
        print("[DEVICE-%s][Batch #%d] %d models"%(device, ix_batch, current_batch_size), end="\n\n")        
        ix_batch += 1
                    
        solver.solve(
            model=model,
            dt=dt,
            n_iters=n_iters,
            verbose=0
        )       
        
        with model.am:
            init_pts = model.am.get(initializer.init_pts)
            init_states = model.am.get(initializer.init_states)
            params = model.am.get(model.params)

        for j in range(current_batch_size):
            # Check numerical errors.
            # Ignore this model if numerical errors has occurred.
            if model.is_state_invalid(index=j):
                print("- [DEVICE-%s][Numerical error] Ignore model #%d in the batch #%d..."%(device, j+1, i+1))
                continue

            img_morph, pattern = model.create_image(index=j)
            img_morph = img_morph.convert('RGB')            
            arr_morph = np.asarray(img_morph)
                  
            # Hashing morph image.
            h.update(arr_morph)
            hash_morph = h.intdigest()
            h.reset()                  
            
            # Hashing model data.
            h.update(init_pts[j, ...])
            h.update(init_states[j, ...])
            h.update(params[j, ...])
            hash_model = h.intdigest()  
            h.reset()     
            
            list_dict_fpaths.append(dict_fpaths)
            
            dpath_morph = pjoin(dpath_output_dataset, str(hash_morph))
            dpath_model = pjoin(dpath_morph, "model")
            dpath_img_morph = pjoin(dpath_morph, "morph")                
            dpath_img_pattern = pjoin(dpath_morph, "pattern")
            dpath_state = pjoin(dpath_morph, "state")                

            os.makedirs(dpath_model, exist_ok=True)
            os.makedirs(dpath_img_morph, exist_ok=True)
            os.makedirs(dpath_img_pattern, exist_ok=True)
            os.makedirs(dpath_state, exist_ok=True)
            
            str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
            fpath_model_new = pjoin(dpath_model,
                                    "model_%s_%d.json"%(str_now, j))            
            fpath_img_morph_new = pjoin(dpath_img_morph,
                                    "morph_%s_%d.png"%(str_now, j))
            fpath_img_pattern_new = pjoin(dpath_img_pattern,
                                      "pattern_%s_%d.png"%(str_now, j))            
            fpath_state_new = pjoin(dpath_state,
                                     "state_%s_%d.npz"%(str_now, j))
                            
            model.save_model(index=j,
                             fpath=fpath_model_new,
                             initializer=initializer,
                             solver=solver)
            
            model.save_image(index=j,
                             fpath_morph=fpath_img_morph_new,
                             fpath_pattern=fpath_img_pattern_new)
            
            model.save_states(index=j, fpath=fpath_state_new)

            print("- [Confirmed model]", fpath_model_new, end='\n')

        # end of for
        t_end = time.time()
    
        print("[DEVICE-%s][Batch #%d duration] %f sec." % (device, ix_batch, t_end - t_beg), end="\n\n")
    # end of for i
# end of def solve_batches


if __name__ == "__main__":



    parser = argparse.ArgumentParser(
        description='Parse configuration for augmentation.'
    )

    parser.add_argument('--config',
                        dest='config',
                        action='store',
                        type=str,
                        help="Designate the configuration file path.")

    parser.add_argument('--device',
                        dest='device',
                        action='store',
                        type=str,
                        help='Designate the device.',
                        default="cpu")

                
    args = parser.parse_args()

    fpath_config = osp.abspath(args.config)        
            
    with open(fpath_config, "rt") as fin:
        config = yaml.safe_load(fin)
    
    devices = None
    if "DEVICE" in config:
        devices = parse_devices(config["DEVICE"])
        
    if not devices:
        devices = parse_devices(args.device)

      
    print("[DEVICES]", devices)

    
    # Get the input and output directories.
    dpath_input_dataset = config["DPATH_INPUT_DATASET"]


    # Collect model files within dpath_dataset.
    list_dict_fpaths = []
            
    list_init_pts = []
    list_init_states = []
    list_params = []    
    
    # dict_morphs = defaultdict(set)
    # dict_model_id = {}
    

    n_models = 0
    for dname in os.listdir(dpath_input_dataset):
        dpath_morph = pjoin(dpath_input_dataset, dname)
        
        if not osp.isdir(dpath_morph):
            continue            
                
        dpath_model = pjoin(dpath_morph, "model")
        dpath_img_morph = pjoin(dpath_morph, "morph")
        dpath_img_pattern = pjoin(dpath_morph, "pattern")
        dpath_state = pjoin(dpath_morph, "state")
            
        for fname in os.listdir(dpath_model):
            
            fpath_model = osp.join(dpath_model, fname)
            if not osp.isfile(fpath_model) \
                or not fname.startswith("model_"):
                continue
            
            # Get the model ID
            fname, ext = osp.splitext(fname)
            items = fname.split('_')        
            model_id = '_'.join(items[1:])
            
            fpath_img_morph = osp.join(dpath_img_morph,
                                       "morph_%s.png"%(model_id))
            
            fpath_img_pattern = osp.join(dpath_img_pattern,
                                         "pattern_%s.png"%(model_id))
            
            fpath_state = osp.join(dpath_state,
                                   "state_%s.npz"%(model_id))
            
            
            if not osp.isfile(fpath_img_morph):
                raise FileNotFoundError(fpath_img_morph)
            
            if not osp.isfile(fpath_img_pattern):
                raise FileNotFoundError(fpath_img_pattern)
                
            if not osp.isfile(fpath_state):
                raise FileNotFoundError(fpath_state)
                
            
            dict_fpaths = {"model": fpath_model,
                           "morph": fpath_img_morph,
                           "pattern": fpath_img_pattern,
                           "state": fpath_state}
            
            # Hash this model.
            list_morphs, init_pts, init_states, params \
                                = get_data(config, [dict_fpaths])
            
            list_init_pts.append(init_pts)
            list_init_states.append(init_states)
            list_params.append(params)
                               
            list_dict_fpaths.append(dict_fpaths)

            n_models += 1
            print("[Model #%d] %s"%(n_models, fpath_model))
        # end of for
    # end of for

    random.shuffle(list_dict_fpaths)
    
    # Get the statistics from all parameter sets.        
    n_total = len(list_dict_fpaths)

    print()
    print("The total number of models:", n_models)

    arr_init_pts = np.concatenate(list_init_pts, axis=0)
    arr_init_states = np.concatenate(list_init_states, axis=0)
    arr_params = np.concatenate(list_params, axis=0)
        
    mean_init_pts = arr_init_pts.mean(axis=0)
    std_init_pts = arr_init_pts.std(axis=0)
    
    mean_init_states = arr_init_states.mean(axis=0)
    std_init_states = arr_init_states.std(axis=0)
    min_init_states = arr_init_states.min(axis=0)
    
    mean_params = arr_params.mean(axis=0)
    std_params = arr_params.std(axis=0) 
    
    min_params = arr_params.min(axis=0)
    max_params = arr_params.max(axis=0)
    
    print("Mean. states:", mean_init_states)
    print("Std. states:", std_init_states)
                
    print("Mean. params:", mean_params)    
    print("Std. params:", std_params)


    n_devices = len(devices)
    if n_devices == 1:
        solve_batches(config, devices[0], list_dict_fpaths)
        
    elif n_devices > 1:
        
        # Split batches.
        n_proc_batches = int(np.ceil(len(list_dict_fpaths) / n_devices))
        
        list_proc_batches = []
        for i in range(0, len(list_dict_fpaths), n_proc_batches):
            batches = list_dict_fpaths[i:i+n_proc_batches]
            list_proc_batches.append(batches)
        
        # Start multiprocessing.
        """
        multiprocessing.set_start_method('spawn', force=True)
        procs = []    
        for i in range(n_devices):
            device = devices[i]
            batches = list_proc_batches[i]
            _proc = Process(target=solve_batches, args=(config, device, batches))
            _proc.start()
            procs.append(_proc)

        for _proc in procs:
            _proc.join()
        """
        list_args = []
        for i in range(n_devices):
            list_args.append((config, devices[i], list_proc_batches[i]))

        pool = multiprocessing.Pool(processes=n_devices)

        pool.starmap(solve_batches, list_args)

        pool.close()
        pool.join()

    else:
        raise RuntimeError("The number of devices should be greater than 0.")

