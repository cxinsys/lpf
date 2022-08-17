import os
import os.path as osp
from os.path import join as pjoin
import time
from datetime import datetime
import shutil

import numpy as np
np.seterr(all='raise')

from lpf.models import LiawModel
from lpf.initializers import LiawInitializer


if __name__ == "__main__":

    n_repeats = 1000000
    batch_size = 4
    
    device = "cpu"  #"cuda:0"
    dx = 0.1
    dt = 0.01
    width = 128
    height = 128
    thr = 0.5
    n_iters = 500000
    shape = (width, height)


    # Create the output directory.
    str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    dname = "experiment_rand_params_%s" % (str_now)
    droot = pjoin(osp.abspath("./output"), dname)    
    os.makedirs(droot, exist_ok=True)
        
    # Copy this source file to the output directory for recording purpose.
    fpath_src = pjoin(osp.dirname(__file__), osp.basename(__file__))
    fpath_dst = pjoin(droot, osp.basename(__file__))
    shutil.copyfile(fpath_src, fpath_dst)

    # Create a parameter array
    init_states = np.zeros((batch_size, 2), dtype=np.float64)
    params = np.zeros((batch_size, 8, 1, 1), dtype=np.float64)
       
    
    i = 0
    while i < n_repeats:
        
        # Create the output directory.
        str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
        dpath_output = pjoin(droot, "repeat_%d"%(i+1))
        os.makedirs(dpath_output, exist_ok=True)
        
    
        # Generate the random initial points.
        init_pts = np.random.randint(0, 128, size=(batch_size, 25, 2),
                                     dtype=np.uint32)
           
        
        # Create the initializer
        initializer = LiawInitializer(init_pts)
        
        # Create the model.
        model = LiawModel(
            width=width,
            height=height,
            dx=dx,
            dt=dt,
            n_iters=n_iters,
            initializer=initializer,
            device=device
        )

        init_pts = np.random.randint(0, 128, size=(batch_size, 25, 2))        
        
        init_states[:, 0] = 10 ** np.random.uniform(-1, 1.0, size=(batch_size))  # u0
        init_states[:, 1] = 10 ** np.random.uniform(-1, 1.0, size=(batch_size))  # v0
        
        params[:, 0] = 10 ** np.random.uniform(-4,  0, size=(batch_size, 1, 1))  # Du
        params[:, 1] = 10 ** np.random.uniform(-4,  0, size=(batch_size, 1, 1))  # Dv
        params[:, 2] = 10 ** np.random.uniform(-2,  2, size=(batch_size, 1, 1))  # ru
        params[:, 3] = 10 ** np.random.uniform(-2,  2, size=(batch_size, 1, 1))  # rv
        params[:, 4] = 10 ** np.random.uniform(-4,  0, size=(batch_size, 1, 1))  # k
        params[:, 5] = 10 ** np.random.uniform(-4,  0, size=(batch_size, 1, 1))  # su
        params[:, 6] = 10 ** np.random.uniform(-4,  0, size=(batch_size, 1, 1))  # sv
        params[:, 7] = 10 ** np.random.uniform(-3, -1, size=(batch_size, 1, 1))  # mu
        
    
        # init_states, params = model.parse_model_dicts(model_dicts)
        t_beg = time.time()
        try:
            model.solve(init_states,
                        params,
                        init_pts=init_pts,
                        n_iters=n_iters,
                        period_output=n_iters,
                        dpath_images=dpath_output,
                        verbose=1)
        except Exception as err:
            print("[IGNORE THE INDIVIDUAL WITH ERROR]", err)
            shutil.rmtree(dpath_output)
            continue
        t_end = time.time()
        print("Elapsed time: %f sec." % (t_end - t_beg))
        
        
        for j in range(batch_size):
            fpath=pjoin(dpath_output, "individual_%d"%(j+1), "model.json")
            model.save_model(fpath,
                             i=j, 
                             init_states=init_states,
                             init_pts=init_pts,
                             params=params)
        # end of for
        
        i += 1
    # end of while
