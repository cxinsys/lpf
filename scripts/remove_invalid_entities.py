import os
import os.path as osp
from os.path import join as pjoin
import shutil
import json
import argparse

import numpy as np
from PIL import Image

from lpf.models import LiawModel
from lpf.utils import is_state_invalid
from lpf.utils import is_param_invalid
from lpf.utils import is_morph_invalid


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
    
    n_valid = 0
    n_invalid = 0
    n_updated = 0
    
    fpaths_removed = []
    dpaths_removed = []
    
    for dname in os.listdir(dpath_dataset):
    
        dpath_morph = pjoin(dpath_dataset, dname)   
        
        if not osp.isdir(dpath_morph):
            continue            
                
        dpath_models = pjoin(dpath_morph, "models")
        
        if not os.path.isdir(dpath_models) or os.listdir(dpath_models) == 0:
            shutil.rmtree(dpath_morph)
            continue
        
        dpath_ladybirds = pjoin(dpath_morph, "ladybirds")
        dpath_patterns = pjoin(dpath_morph, "patterns")
        dpath_states = pjoin(dpath_morph, "states")
        
        for fname in os.listdir(dpath_models):
            
            fpath_model = osp.join(dpath_models, fname)
            if not osp.isfile(fpath_model) \
                or not fname.startswith("model_"):
                continue
            
            with open(fpath_model, "rt") as fin:
                model_dict = json.load(fin)
                        
            is_model_updated = False
            
            if "model_name" in model_dict:            
                model_name = model_dict["model_name"]
                del model_dict["model_name"]
                model_dict["model"] = model_name
                is_model_updated = True                
                print("[RENAME] model_name to model")
            # end of if
            
            if model_dict["solver"] == None:            
                model_dict.update({'solver': 'EulerSolver',
                                   'dt': 0.01,
                                   'n_iters': 500000})
                is_model_updated = True
                print("[UPDATE] solver information")
            # end of if
            
            if is_model_updated:
                with open(fpath_model, "wt") as fout:                
                    json.dump(model_dict, fout)
                    
                n_updated += 1

            params = LiawModel.parse_params([model_dict])
            
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
            
            
            is_not_found = False
            fpath_not_found = None
            if not osp.isfile(fpath_ladybird):
                is_not_found = True
                fpath_not_found = fpath_ladybird
                # raise FileNotFoundError(fpath_ladybird)
            
            if not osp.isfile(fpath_pattern):
                is_not_found = True
                fpath_not_found = fpath_pattern
                # raise FileNotFoundError(fpath_pattern)
                
            if not osp.isfile(fpath_states):
                is_not_found = True
                fpath_not_found = fpath_states
                # raise FileNotFoundError(fpath_states)
            
            if is_not_found:
                n_invalid += 1
                print("[INVALID #%d due to NOTFOUND] %s"%(n_invalid, fpath_not_found))
                
                entities = os.listdir(dpath_morph)
                if len(entities) == 0:                
                    dpaths_removed.append(dpath_morph)
                continue
                    
            if is_param_invalid(params):
                n_invalid += 1
                print("[INVALID #%d due to PARAMS] %s"%(n_invalid, fpath_model))
                
                os.remove(fpath_model)
                os.remove(fpath_ladybird)
                os.remove(fpath_pattern)
                fpaths_removed.append(fpath_states)
                
                entities = os.listdir(dpath_ladybirds)
                if len(entities) == 0:                
                    dpaths_removed.append(dpath_morph)
                continue
            
            dict_states = np.load(fpath_states)
            
            arr_u = dict_states['u']
            arr_v = dict_states['v']        
            
            if is_state_invalid(arr_u, arr_v):
                n_invalid += 1
                print("[INVALID #%d due to STATES] %s"%(n_invalid, fpath_states))
                
                os.remove(fpath_model)
                os.remove(fpath_ladybird)
                os.remove(fpath_pattern)
                fpaths_removed.append(fpath_states)
                
                entities = os.listdir(dpath_ladybirds)
                if len(entities) == 0:
                    dpaths_removed.append(dpath_morph)
                continue
            
            
            img_ladybird = Image.open(fpath_ladybird)
            
            if is_morph_invalid(img_ladybird):
                n_invalid += 1
                print("[INVALID #%d due to CP] %s"%(n_invalid, fpath_ladybird))
                
                os.remove(fpath_model)
                os.remove(fpath_ladybird)
                os.remove(fpath_pattern)
                fpaths_removed.append(fpath_states)
                
                entities = os.listdir(dpath_ladybirds)
                if len(entities) == 0:
                    dpaths_removed.append(dpath_morph)
                continue
    
            n_valid += 1
            # print("[VALID #%d] %s"%(n_valid, fpath_model))
        # end of for
    # end of for
            
    for fpath in fpaths_removed:
        os.remove(fpath)
    
    for dpath in dpaths_removed:
        print("[REMOVE DIR]", dpath)
        shutil.rmtree(dpath)
        
    print("[Num. Valid Models]", n_valid)
    print("[Num. Invalid Models]", n_invalid)
    print("[Num. Updated Models]", n_updated)
