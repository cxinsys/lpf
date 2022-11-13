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
from lpf.data import load_model_dicts


if __name__ == "__main__":

    device = "cuda:0"
    dx = 0.1
    dt = 0.01
    width = 128
    height = 128
    thr = 0.5
    n_iters = 500000
    shape = (width, height)

    # Create the output directory.
    str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    dpath_output = pjoin(osp.abspath("./output"),
                         "experiment_batch_%s" % (str_now))
    os.makedirs(dpath_output, exist_ok=True)

    # Copy this source file to the output directory for recording purpose.
    fpath_src = pjoin(osp.dirname(__file__), osp.basename(__file__))
    fpath_dst = pjoin(dpath_output, osp.basename(__file__))
    shutil.copyfile(fpath_src, fpath_dst)

    # Create initializer
    initializer = LiawInitializer()

    # To test the batch processing, add a single dict repeatedly.
    batch_size = 1
    model_dicts = []

    # n2v = {"u0": 2.26149602133276, "v0": 1.6635557745335625,
    #        "Du": 0.00027545326069340044, "Dv": 0.07983333454476473,
    #        "ru": 0.17999999999999997, "rv": 0.09116599746769462,
    #        "k": 0.19096243208916244, "su": 0.0010010607170663603,
    #        "sv": 0.02448138845767171, "mu": 0.08050631184429496,
    #        "init_pts_0": ["119", "1"], "init_pts_1": ["52", "40"], "init_pts_2": ["26", "49"], "init_pts_3": ["40", "55"], "init_pts_4": ["49", "52"], "init_pts_5": ["77", "15"], "init_pts_6": ["65", "105"], "init_pts_7": ["57", "50"], "init_pts_8": ["15", "20"], "init_pts_9": ["42", "76"], "init_pts_10": ["28", "47"], "init_pts_11": ["104", "9"], "init_pts_12": ["37", "70"], "init_pts_13": ["49", "90"], "init_pts_14": ["78", "78"], "init_pts_15": ["77", "11"], "init_pts_16": ["84", "92"], "init_pts_17": ["9", "35"], "init_pts_18": ["90", "69"], "init_pts_19": ["17", "73"], "init_pts_20": ["0", "0"], "init_pts_21": ["0", "0"], "init_pts_22": ["0", "0"], "init_pts_23": ["0", "0"], "init_pts_24": ["0", "0"], "width": 128, "height": 128, "dt": 0.01, "dx": 0.1, "n_iters": 500000, "thr": 0.5, "initializer": "LiawInitializer"}

    # for i in range(batch_size):
    #     model_dicts.append(n2v)

    # To test the batch processing, add model JSON files.
    model_dicts = load_model_dicts("../population/init_pop_succinea/")

    # Update the initializer.
    initializer.update(model_dicts)

    # Create a model.
    model = LiawModel(
        width=width,
        height=height,
        dx=dx,
        dt=dt,
        n_iters=n_iters,
        initializer=initializer,
        device=device
    )

    params = model.parse_params(model_dicts)

    t_beg = time.time()

    model.solve(params,
                n_iters=n_iters,
                period_output=1000,  # n_iters - 1,
                dpath_ladybird=dpath_output,
                dpath_pattern=dpath_output,
                verbose=1)
    
    t_end = time.time()

    print("Elapsed time: %f sec." % (t_end - t_beg))
