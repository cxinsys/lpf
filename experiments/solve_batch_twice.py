import os
import os.path as osp
from os.path import join as pjoin
import time
from datetime import datetime
import shutil

import numpy as np
np.seterr(all='raise')

import matplotlib.pyplot as plt

from lpf.data import load_model_dicts
from lpf.initializers import LiawInitializer
from lpf.models import LiawModel
from lpf.solvers import EulerSolver


if __name__ == "__main__":

    device = "cuda:0"  # torch:gpu
    dx = 0.1
    dt = 0.01  # A too big dt causes an overflow in the solver.
    width = 128
    height = 128
    thr = 0.5
    n_iters = 500000
    shape = (height, width)

    # Create the output directory.
    str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    dpath_output = pjoin(osp.abspath("./output"),
                         "experiment_batch_%s" % (str_now))
    os.makedirs(dpath_output, exist_ok=True)

    # Copy this source file to the output directory for recording purpose.
    fpath_code_src = pjoin(osp.dirname(__file__), osp.basename(__file__))
    fpath_code_dst = pjoin(dpath_output, osp.basename(__file__))
    shutil.copyfile(fpath_code_src, fpath_code_dst)

    # Create initializer
    initializer = LiawInitializer()

    # To test the batch processing, add a single dict repeatedly.
    model_dicts = []

    # To test the batch processing, add model JSON files.
    model_dicts = load_model_dicts("../population/init_pop_axyridis/")

    batch_size = len(model_dicts)


    # Update the initializer.
    initializer.update(model_dicts)

    # Create a model.
    params = LiawModel.parse_params(model_dicts)
    model = LiawModel(
        initializer=initializer,
        params=params,
        width=width,
        height=height,
        dx=dx,
        #color_u=[161, 102, 0],  # model
        #color_v=[59, 161, 90],  # model
        device=device  # solver and model
    )

    solver = EulerSolver()

    t_beg = time.time()
   
    trj_1 = solver.solve(
        model=model,
        dt=dt,
        n_iters=n_iters,
        period_output=100,
        dpath_model=dpath_output,
        dpath_ladybird=dpath_output,
        dpath_pattern=dpath_output,
        dpath_states=dpath_output,
        init_model=True,
        iter_end=10000,
        get_trj=True,
        verbose=1
    )

    trj_2 = solver.solve(
        model=model,
        dt=dt,
        n_iters=n_iters,
        period_output=1000,
        dpath_model=dpath_output,
        dpath_ladybird=dpath_output,
        dpath_pattern=dpath_output,
        dpath_states=dpath_output,
        init_model=False,
        iter_begin=10000,
        iter_end=n_iters,
        get_trj=True,
        verbose=1
    )


    t_end = time.time()

    print("Elapsed time: %f sec." % (t_end - t_beg))
    
    
    ix_morph = 0  # Select a morph in the batch.
        
    np_trj_1 = trj_1.get()
    np_trj_2 = trj_2.get()

    # Visualize the changes in the adjacent states using MSE.
    diff_trj_1 = np.mean(np.sqrt((np_trj_1[0:-1] - np_trj_1[1:])**2),
                         axis=(3, 4))
    
    
    fig1 = plt.figure()
    plt.plot(diff_trj_1[:, :, ix_morph])
    plt.legend(['u', 'v'])


    diff_trj_2 = np.mean(np.sqrt((np_trj_2[0:-1] - np_trj_2[1:])**2),
                         axis=(3, 4))    
    fig2 = plt.figure()
    plt.plot(diff_trj_2[:, :, ix_morph])
    plt.legend(['u', 'v'])


    # Visualize the states of u.
    plt.figure()
    plt.imshow(np_trj_1[-1, 0, ix_morph, ...])
    
    plt.figure()
    plt.imshow(np_trj_2[-1, 0, ix_morph, ...])

