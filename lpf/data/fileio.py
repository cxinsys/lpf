import os
import os.path as osp

import numpy as np
from PIL import Image


def load_targets(dpath_target,
                 subtypes, 
                 resize_shape=None):
    targets = []
    for fname in os.listdir(dpath_target):
        if not fname.startswith("ladybird"):
            continue 

        items = fname.split("_")
        subtype = items[1]

        if subtype not in subtypes:
            continue

        fpath = osp.join(dpath_target, fname)
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
    