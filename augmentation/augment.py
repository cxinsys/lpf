import os
import os.path as osp
from os.path import join as pjoin
import time
from datetime import datetime
import shutil
import json
import argparse

import yaml
import numpy as np
np.seterr(all='raise')
from PIL import ImageColor

from lpf.initializers import InitializerFactory
from lpf.models import ModelFactory
from lpf.solvers import SolverFactory


def get_data(config, batch):
    model_dicts = []

    for (fpath_model, fpath_ladybird) in batch:
        with open(fpath_model, "rt") as fin:
            n2v = json.load(fin)
        model_dicts.append(n2v)
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
    
    return initializer.init_pts, initializer.init_states, params
    
    

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
                        default=None)

                
    args = parser.parse_args()

    fpath_config = osp.abspath(args.config)        
            
    with open(fpath_config, "rt") as fin:
        config = yaml.safe_load(fin)
   

    device = None
    if not args.device:
        device = "cpu"
    elif args.device:
        device = args.device
    elif "DEVICE" in config:
        device = str(config["DEVICE"]).lower()
    else:
        raise RuntimeError("Device should be defined.")
      
    verbose = int(config["VERBOSE"])
    # period_output = int(config["PERIOD_OUTPUT"])
    
    batch_size = int(config["BATCH_SIZE"])
    
    # Get the input and output directories.
    dpath_dataset = config["DPATH_DATASET"]  # Input dataset
    dpath_augdataset = config["DPATH_AUGDATASET"]  # Output dataset
    
    os.makedirs(dpath_augdataset, exist_ok=True)

    str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    dpath_output = pjoin(osp.abspath(dpath_augdataset),
                         "augment_batch_%s" % (str_now))
    os.makedirs(dpath_output, exist_ok=True)
    
    # Create sub-directories.
    dpath_model = pjoin(dpath_output, "models")
    dpath_pattern = pjoin(dpath_output, "patterns")
    dpath_ladybird = pjoin(dpath_output, "ladybirds")
    dpath_states = pjoin(dpath_output, "states")
    
    os.makedirs(dpath_model, exist_ok=True)
    os.makedirs(dpath_pattern, exist_ok=True)
    os.makedirs(dpath_ladybird, exist_ok=True)
    os.makedirs(dpath_states, exist_ok=True)
    
    # Copy this source file to the output directory for recording purpose.
    fpath_src = pjoin(osp.dirname(__file__), osp.basename(__file__))
    fpath_dst = pjoin(dpath_output, osp.basename(__file__))
    shutil.copyfile(fpath_src, fpath_dst)
        
    # Create the model.
    dx = float(config["DX"])
    dt = float(config["DT"])
    width = int(config["WIDTH"])
    height = int(config["HEIGHT"])
    thr = float(config["THR_COLOR"])
    n_iters = int(config["N_ITERS"])
    rtol = float(config["RTOL_EARLY_STOP"])
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

    # Collect model files within dpath_dataset.
    dict_fpath = {}
    list_fpath = []
    
    for entity in os.listdir(dpath_dataset):
        if not entity.startswith("model_"):
            continue
        
        fpath_model = osp.join(dpath_dataset, entity)
        
        # Get the model ID
        fname, ext = osp.splitext(entity)
        items = fname.split('_')        
        model_id = items[1]
        
        fpath_ladybird = osp.join(dpath_dataset, "ladybird_%s.png"%(model_id))
        
        if not osp.isfile(fpath_model):
            raise FileNotFoundError(fpath_model)
        
        if not osp.isfile(fpath_ladybird):
            raise FileNotFoundError(fpath_ladybird)
        

        dict_fpath[model_id] = (fpath_model, fpath_ladybird)
        list_fpath.append(dict_fpath[model_id])
    # end of for

    # Get the statistics from all parameter sets.        
    n_total = len(list_fpath)
    
    # arr_init_states = np.zeros((n_total, 2), dtype=np.float64)
    # arr_init_pts = np.zeros((n_total, 2), dtype=np.float64)
    # arr_params = np.zeros((n_total, 8), dtype=np.float64)
        
    list_init_pts = []
    list_init_states = []
    list_params = []
    
    ix_batch = 1
    for i in range(0, len(list_fpath), batch_size):
    
        batch = list_fpath[i:i+batch_size]        
        ix_batch += 1
        
        init_pts, init_states, params = get_data(config, batch)
        
        list_init_pts.append(init_pts)
        list_init_states.append(init_states)
        list_params.append(params)
        
        n_total += len(batch)
    # end of for
        
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
    
    
    # Perform numerical simulation
    half_batch_size = batch_size // 2  # The half of batch size
        
    ix_batch = 1
    for i in range(0, len(list_fpath), half_batch_size):
        t_beg = time.time()

        half_batch = list_fpath[i:i+half_batch_size]
        if len(half_batch) != half_batch_size:
            half_batch_size = len(half_batch)        
        
        model_dicts = []
        for (fpath_model, fpath_ladybird) in half_batch:
            with open(fpath_model, "rt") as fin:
                n2v = json.load(fin)
                
            model_dicts.append(n2v)
        # end of for
                
        model_dicts = 2 * model_dicts

        # Create an initializer
        initializer = InitializerFactory.create(
            name=config["INITIALIZER"],
        )
        
        # Update the initializer.
        initializer.update(model_dicts)
        
        
        # Randomly generate the half of initial points.
        shape = (half_batch_size, *initializer.init_pts.shape[1:])
        init_pts_rand = np.random.normal(mean_init_pts,
                                         std_init_pts,
                                         size=shape)     
               
        init_pts_rand = np.clip(init_pts_rand,
                                a_min=(0, 0),
                                a_max=(height-1, width-1))
        
        initializer.init_pts[half_batch_size:, :] = \
                    np.asarray(init_pts_rand, dtype=initializer.init_pts.dtype)
        
        
        # Randomly generate the half of initial states.
        shape = (half_batch_size, initializer.init_states.shape[1])
        init_states_rand = np.random.normal(mean_init_states,
                                            std_init_states,
                                            size=shape)      
        
        init_states_rand = np.clip(init_states_rand,
                                   a_min=min_init_states,
                                   a_max=None)
        
        initializer.init_states[half_batch_size:, :] = init_states_rand
        

        # Create a model.
        model = ModelFactory.create(
            name=config["MODEL"],
            initializer=initializer,
            width=width,
            height=height,                 
            dx=dx,
            color_u=color_u,
            color_v=color_v,
            thr_color=thr_color,
            device=device
        )

        params = model.parse_params(model_dicts)

        # Randomly generate the half of parameter sets.
        shape = (half_batch_size, params.shape[1])
        params_rand = np.random.normal(mean_params,
                                       std_params,
                                       size=shape)        
        
        params[half_batch_size:, :] = np.clip(params_rand,
                                              a_min=min_params,
                                              a_max=None)
               

        print("[Batch #%d] %d models"%(ix_batch, params.shape[0]), end="\n\n")        
        ix_batch += 1
        
        model.params = params
        
        solver.solve(
            model=model,
            dt=dt,
            n_iters=n_iters,
            verbose=verbose
        )       

        for j in range(len(batch)):
            str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
            fpath_model_new = pjoin(dpath_model,
                                    "model_%s_%d.json"%(str_now, j))            
            fpath_ladybird_new = pjoin(dpath_ladybird,
                                       "ladybird_%s_%d.png"%(str_now, j))
            fpath_pattern_new = pjoin(dpath_pattern,
                                      "pattern_%s_%d.png"%(str_now, j))            
            fpath_states_new = pjoin(dpath_states,
                                     "states_%s_%d.npz"%(str_now, j))
            
            model.save_model(index=j,
                             fpath=fpath_model_new,
                             initializer=initializer,
                             params=params)
            
            model.save_image(index=j,
                             fpath_ladybird=fpath_ladybird_new,
                             fpath_pattern=fpath_pattern_new)
            
            model.save_states(index=j, fpath=fpath_states_new)
        # end of for
        t_end = time.time()
    
        print("- [Batch Duration] %f sec." % (t_end - t_beg), end="\n\n")
    # end of for
