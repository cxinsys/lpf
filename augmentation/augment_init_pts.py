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

from lpf.models import LiawModel
from lpf.initializers import LiawInitializer
from lpf.solvers import EulerSolver

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Parse configurations for augmentation.'
    )

    parser.add_argument('--config',
                        dest='config',
                        action='store',
                        type=str,
                        help='Designate the file path of configuration file in YAML')

        
    args = parser.parse_args()

    fpath_config = osp.abspath(args.config)        
            
    with open(fpath_config, "rt") as fin:
        config = yaml.safe_load(fin)
    
    device = str(config["DEVICE"]).lower()
      
    verbose = int(config["VERBOSE"])
    period_output = int(config["PERIOD_OUTPUT"])
    
    n_augmentation = int(config["N_AUGMENTATION"])
    batch_size = int(config["BATCH_SIZE"])
    dpath_dataset = config["DPATH_DATASET"]  # Input dataset
    dpath_augdataset = config["DPATH_AUGDATASET"]  # Output dataset
    
    os.makedirs(dpath_augdataset, exist_ok=True)
    
    
    # Create the model.
    dx = float(config["DX"])
    dt = float(config["DT"])
    width = int(config["WIDTH"])
    height = int(config["HEIGHT"])
    thr = float(config["THR"])
    n_iters = int(config["N_ITERS"])
    rtol = float(config["RTOL_EARLY_STOP"])
    shape = (width, height)
    
    
    dict_fpath = {}
    list_fpath = []
    model_dicts = []
    aug_model_dicts = []
    
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
        
        
        with open(fpath_model, "rt") as fin:
            n2v = json.load(fin)
            model_dicts.append(n2v)
    # end of for
    
    
    # Augment using the given models.    
    for model in model_dicts:
        for i in range(n_augmentation):
            
            aug_model = dict(model)
            
            for j in range(25):
                width = aug_model["width"]
                height = aug_model["height"]
                aug_model["init_pts_%d"%(i)] = [np.random.randint(0, height),
                                                np.random.randint(0, width)]
            
            # end of for
            
            aug_model_dicts.append(aug_model)
            
        # end of for
    # end of for
    
    # Solve the PDEs of models.        
    ix_batch = 1
    for i in range(0, len(aug_model_dicts), batch_size):
        t_beg = time.time()

        batch_model_dicts = aug_model_dicts[i:i+batch_size]

        print("[Batch #%d] %d models"%(ix_batch, len(batch_model_dicts)),
              end="\n\n")        
        ix_batch += 1
        
        
        # Create the output directory.
        str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
        dpath_output = pjoin(dpath_augdataset,
                             "augment_batch_%s" % (str_now))
        os.makedirs(dpath_output, exist_ok=True)
     
        # Copy this source file to the output directory for recording purpose.
        fpath_src = pjoin(osp.dirname(__file__), osp.basename(__file__))
        fpath_dst = pjoin(dpath_output, osp.basename(__file__))
        shutil.copyfile(fpath_src, fpath_dst)
    
        # Create initializer
        initializer = LiawInitializer()
        
        # Update the initializer.
        initializer.update(batch_model_dicts)
    
        # Create a model.
        model = LiawModel(
            width=width,
            height=height,
            dx=dx,
            initializer=initializer,
            device=device
        )
    
        # Solve the PDE.
        params = LiawModel.parse_params(batch_model_dicts)
        solver = EulerSolver()
        solver.solve(model=model,
                     dt=dt,
                     n_iters=n_iters,
                     period_output=period_output,
                     dpath_ladybird=dpath_output,
                     dpath_pattern=dpath_output,
                     verbose=verbose)
        

        for j in range(len(batch_model_dicts)):
            str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
            fpath_model_new = pjoin(dpath_augdataset,
                                    "model_%s_%d.json"%(str_now, j))            
            fpath_ladybird_new = pjoin(dpath_augdataset,
                                       "ladybird_%s_%d.png"%(str_now, j))
            fpath_pattern_new = pjoin(dpath_augdataset,
                                      "pattern_%s_%d.png"%(str_now, j))
            
            model.save_model(index=j,
                             fpath=fpath_model_new,
                             initializer=initializer,
                             solver=solver,
                             params=params)
            
            model.save_image(index=j,
                             fpath_ladybird=fpath_ladybird_new,
                             fpath_pattern=fpath_pattern_new)
        # end of for
        t_end = time.time()
    
        print("[Batch Duration] %f sec." % (t_end - t_beg), end="\n\n")
