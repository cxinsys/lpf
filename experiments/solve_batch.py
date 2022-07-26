import os
import os.path as osp
from os.path import join as pjoin
import time
from datetime import datetime
import shutil

from lpf.models import LiawModel
from lpf.initializers import LiawInitializer
from lpf.data import load_model_dicts

if __name__ == "__main__":

    device = "cuda:1"
    dx = 0.1
    dt = 0.01
    width = 128
    height = 128
    thr = 0.5
    n_iters = 500000
    shape = (width, height)

    # Create the output directory.
    str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    dpath_output = pjoin(osp.abspath("./output"), "experiment_%s" % (str_now))
    os.makedirs(dpath_output, exist_ok=True)

    # Copy this source file to the output directory for recording.
    fpath_src = pjoin(osp.dirname(__file__), osp.basename(__file__))
    fpath_dst = pjoin(dpath_output, osp.basename(__file__))
    shutil.copyfile(fpath_src, fpath_dst)

    # Create initializer
    initializer = LiawInitializer()

    # To test the batch processing, add a single dict repeatedly.
    batch_size = 64
    model_dicts = []
    n2v = {
            "fitness": 12.36867523754235, "u0": 1.1309170035240579, "v0": 2.506183016259239, "Du": 0.0004999999999999999,
            "Dv": 0.07500000000000001, "ru": 0.1736527042346181, "rv": 0.08028530394751626, "k": 0.14881975947367232,
            "su": 0.001, "sv": 0.025000000000000005, "mu": 0.07999999999999999, "init_pts_0": ["46", "0"],
            "init_pts_1": ["82", "13"], "init_pts_2": ["103", "57"], "init_pts_3": ["36", "59"], "init_pts_4": ["29", "93"],
            "init_pts_5": ["88", "5"], "init_pts_6": ["50", "115"], "init_pts_7": ["62", "50"], "init_pts_8": ["50", "69"],
            "init_pts_9": ["50", "76"], "init_pts_10": ["34", "32"], "init_pts_11": ["52", "20"], "init_pts_12": ["32", "109"],
            "init_pts_13": ["79", "40"], "init_pts_14": ["61", "61"], "init_pts_15": ["70", "52"], "init_pts_16": ["70", "42"],
            "init_pts_17": ["71", "51"], "init_pts_18": ["110", "29"], "init_pts_19": ["70", "90"], "init_pts_20": ["0", "0"],
            "init_pts_21": ["0", "0"], "init_pts_22": ["0", "0"], "init_pts_23": ["0", "0"], "init_pts_24": ["0", "0"],
            "width": 128, "height": 128, "dt": 0.01, "dx": 0.1, "n_iters": 500000, "thr": 0.5, "initializer": "NoneType"
    }

    for i in range(batch_size):
        model_dicts.append(n2v)

    # To test the batch processing, add model JSON files.
    model_dicts = load_model_dicts("../population/init_pop_axyridis/")

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

    init_states, param_batch = model.parse_model_dicts(model_dicts)
    t_beg = time.time()
    model.solve(init_states,
                param_batch,
                n_iters=n_iters,
                period_output=100, #n_iters - 1,
                dpath_images=dpath_output,
                verbose=1)
    t_end = time.time()

    print("Elapsed time: %f sec." % (t_end - t_beg))
