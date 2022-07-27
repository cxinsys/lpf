import os
import os.path as osp
import json

import numpy as np
from PIL import Image

from lpf.utils import get_module_dpath


def load_model_dicts(dpath):
    model_dicts = []

    for fname in os.listdir(dpath):
        fpath = osp.join(dpath, fname)
        with open(fpath, "rt") as fin:
            n2v = json.load(fin)
        model_dicts.append(n2v)

    return model_dicts


# def load_init_pop(dpath):
#     if not osp.isdir(dpath):
#         raise NotADirectoryError(dpath)
#
#     x = np.zeros((10 + 2*num_init_pts,), dtype=np.float64)
#
#     for i, fname in enumerate(os.listdir(dpath_init_pop)):
#         if i == pop_size:
#             break
#
#         if not fname.endswith("json"):
#             continue
#
#         fpath_model = pjoin(dpath_init_pop, fname)
#
#         # Load the params.
#         with open(fpath_model, "rt") as fin:
#             n2v = json.load(fin)
#
#         x[0] = np.log10(n2v["Du"])
#         x[1] = np.log10(n2v["Dv"])
#         x[2] = np.log10(n2v["ru"])
#         x[3] = np.log10(n2v["rv"])
#         x[4] = np.log10(n2v["k"])
#         x[5] = np.log10(n2v["su"])
#         x[6] = np.log10(n2v["sv"])
#         x[7] = np.log10(n2v["mu"])
#         x[8] = np.log10(n2v["u0"])
#         x[9] = np.log10(n2v["v0"])
#
#         j = 0
#         for name, val in n2v.items():
#             if "init_pts" in name:
#                 x[10 + 2*j] = int(val[0])
#                 x[11 + 2*j] = int(val[1])
#                 j += 1
#         # end of for
#
#         if j == 0: # if there is no initial point.
#             # rc_product: Production of rows and columns
#             rc_product = product(np.arange(40, 90, 10),
#                                  np.arange(10, 110, 20))
#
#             for j, (ir, ic) in enumerate(rc_product):
#                 x[10 + 2*j] = ir
#                 x[11 + 2*j] = ic
#             # end of for
#         # end of if
#     # end of for


def load_targets(ladybird_type, ladybird_subtypes, resize_shape=None):
    targets = []

    dpath = osp.join(get_module_dpath("data"), ladybird_type)

    for fname in os.listdir(dpath):
        if not fname.startswith("ladybird_type"):
            continue 

        items = fname.split("_")
        subtype = items[1]

        if subtype not in ladybird_subtypes:
            continue

        fpath = osp.join(dpath, fname)
        img = Image.open(fpath)
        
        img_no_trans = Image.new("RGBA", img.size, "WHITE")
        img_no_trans.paste(img, mask=img)
        img = img_no_trans
                    
        if resize_shape:
            img = img.resize(resize_shape)
            
        targets.append(img.convert('RGB'))
        print("[TARGET] %s has been added..."%(fname))
    # end of for
    
    return targets
