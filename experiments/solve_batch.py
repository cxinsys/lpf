import time
import os
import os.path as osp
from os.path import join as pjoin
from datetime import datetime
import shutil

import numpy as np
from lpf.utils import get_module_dpath
from lpf.models import LiawModel
from lpf.initializers import InitializerFactory as InitFac
from lpf.initializers import LiawInitializer

if __name__ == "__main__":

    dx = 0.1
    dt = 0.01
    width = 128
    height = 128
    thr = 0.5
    n_iters = 500000
    shape = (width, height)
    
    # Define directories.    
    ladybird_type = "haxyridis"
    dpath_data = pjoin(get_module_dpath("data"), ladybird_type)
    dpath_template = pjoin(dpath_data, "template")
    
    str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    dpath_output = pjoin(osp.abspath("./output"), "single_%s"%(str_now))
    os.makedirs(dpath_output, exist_ok=True)
        
    print(__file__)
    fpath_src = pjoin(osp.dirname(__file__), osp.basename(__file__))
    fpath_dst = pjoin(dpath_output, osp.basename(__file__))
    shutil.copyfile(fpath_src, fpath_dst)

            
    # Define ladybird type and load the corresponding data.
    ladybird_type = "haxyridis"
    dpath_data = pjoin(get_module_dpath("data"), ladybird_type)
    dpath_template = pjoin(dpath_data, "template")
    
    fpath_template = pjoin(dpath_template, "ladybird.png")    
    fpath_mask = pjoin(dpath_template, "mask.png")

    # n2v = {"fitness": 12.36867523754235, "u0": 1.1309170035240579, "v0": 2.506183016259239, "Du": 0.0004999999999999999, "Dv": 0.07500000000000001, "ru": 0.1736527042346181, "rv": 0.08028530394751626, "k": 0.14881975947367232, "su": 0.001, "sv": 0.025000000000000005, "mu": 0.07999999999999999, "init_pts_0": ["46", "0"], "init_pts_1": ["82", "13"], "init_pts_2": ["103", "57"], "init_pts_3": ["36", "59"], "init_pts_4": ["29", "93"], "init_pts_5": ["88", "5"], "init_pts_6": ["50", "115"], "init_pts_7": ["62", "50"], "init_pts_8": ["50", "69"], "init_pts_9": ["50", "76"], "init_pts_10": ["34", "32"], "init_pts_11": ["52", "20"], "init_pts_12": ["32", "109"], "init_pts_13": ["79", "40"], "init_pts_14": ["61", "61"], "init_pts_15": ["70", "52"], "init_pts_16": ["70", "42"], "init_pts_17": ["71", "51"], "init_pts_18": ["110", "29"], "init_pts_19": ["70", "90"], "init_pts_20": ["0", "0"], "init_pts_21": ["0", "0"], "init_pts_22": ["0", "0"], "init_pts_23": ["0", "0"], "init_pts_24": ["0", "0"], "width": 128, "height": 128, "dt": 0.01, "dx": 0.1, "n_iters": 500000, "thr": 0.5, "initializer": "NoneType"}
    
    
    # Create initializer
    initializer = LiawInitializer()

    param_dicts = [
        {"fitness": 12.36867523754235, "u0": 1.1309170035240579, "v0": 2.506183016259239, "Du": 0.0004999999999999999, "Dv": 0.07500000000000001, "ru": 0.1736527042346181, "rv": 0.08028530394751626, "k": 0.14881975947367232, "su": 0.001, "sv": 0.025000000000000005, "mu": 0.07999999999999999, "init_pts_0": ["46", "0"], "init_pts_1": ["82", "13"], "init_pts_2": ["103", "57"], "init_pts_3": ["36", "59"], "init_pts_4": ["29", "93"], "init_pts_5": ["88", "5"], "init_pts_6": ["50", "115"], "init_pts_7": ["62", "50"], "init_pts_8": ["50", "69"], "init_pts_9": ["50", "76"], "init_pts_10": ["34", "32"], "init_pts_11": ["52", "20"], "init_pts_12": ["32", "109"], "init_pts_13": ["79", "40"], "init_pts_14": ["61", "61"], "init_pts_15": ["70", "52"], "init_pts_16": ["70", "42"], "init_pts_17": ["71", "51"], "init_pts_18": ["110", "29"], "init_pts_19": ["70", "90"], "init_pts_20": ["0", "0"], "init_pts_21": ["0", "0"], "init_pts_22": ["0", "0"], "init_pts_23": ["0", "0"], "init_pts_24": ["0", "0"], "width": 128, "height": 128, "dt": 0.01, "dx": 0.1, "n_iters": 500000, "thr": 0.5, "initializer": "NoneType"},
        {"fitness": 12.36867523754235, "u0": 1.1309170035240579, "v0": 2.506183016259239, "Du": 0.0004999999999999999, "Dv": 0.07500000000000001, "ru": 0.1736527042346181, "rv": 0.08028530394751626, "k": 0.14881975947367232, "su": 0.001, "sv": 0.025000000000000005, "mu": 0.07999999999999999, "init_pts_0": ["46", "0"], "init_pts_1": ["82", "13"], "init_pts_2": ["103", "57"], "init_pts_3": ["36", "59"], "init_pts_4": ["29", "93"], "init_pts_5": ["88", "5"], "init_pts_6": ["50", "115"], "init_pts_7": ["62", "50"], "init_pts_8": ["50", "69"], "init_pts_9": ["50", "76"], "init_pts_10": ["34", "32"], "init_pts_11": ["52", "20"], "init_pts_12": ["32", "109"], "init_pts_13": ["79", "40"], "init_pts_14": ["61", "61"], "init_pts_15": ["70", "52"], "init_pts_16": ["70", "42"], "init_pts_17": ["71", "51"], "init_pts_18": ["110", "29"], "init_pts_19": ["70", "90"], "init_pts_20": ["0", "0"], "init_pts_21": ["0", "0"], "init_pts_22": ["0", "0"], "init_pts_23": ["0", "0"], "init_pts_24": ["0", "0"], "width": 128, "height": 128, "dt": 0.01, "dx": 0.1, "n_iters": 500000, "thr": 0.5, "initializer": "NoneType"}
    ]
    
    initializer.from_param_dicts(param_dicts)
    
    
    # num_init_pts = 0
    # init_pts = {}
    # for name, val in n2v.items():
    #     if "init-pts" in name:
    #         # print(name, val)
    #         init_pts[name] = (int(val[0]), int(val[1]))
    #         num_init_pts += 1
    
    # ir_init = np.zeros(num_init_pts, dtype=np.int32)
    # ic_init = np.zeros(num_init_pts, dtype=np.int32)
    
    # for i, (name, val) in enumerate(init_pts.items()):
    #     ir_init[i] = val[0]
    #     ic_init[i] = val[1]
    
    
    
    # Create a model.
    model = LiawModel(
         width=width,
         height=height,
         dx=dx,
         dt=dt,
         n_iters=n_iters,
         initializer=initializer,
         fpath_template=fpath_template,
         fpath_mask=fpath_mask
     )
    
    batch_size = 2
    param_batch = np.zeros((batch_size, 8), dtype=np.float64)
    init_states = np.zeros((batch_size, 2), dtype=np.float64)

    for i, n2v in enumerate(param_dicts):
        param_batch[i, 0] = n2v["Du"]
        param_batch[i, 1] = n2v["Dv"]
        param_batch[i, 2] = n2v["ru"]
        param_batch[i, 3] = n2v["rv"]
        param_batch[i, 4] = n2v["k"]
        param_batch[i, 5] = n2v["su"]
        param_batch[i, 6] = n2v["sv"]
        param_batch[i, 7] = n2v["mu"]
        init_states[i, 0] = n2v["u0"]
        init_states[i, 1] = n2v["v0"]

    t_beg = time.time()
    model.solve(init_states,
                param_batch=param_batch,
                n_iters=n_iters,
                period_output=500,                      
                dpath_images=dpath_output)
    t_end = time.time()
    
    print("Elapsed time: %f sec."%(t_end - t_beg))
