import numpy as np
from PIL import Image


def is_param_invalid(params):
    """params is a numpy.ndarray.
    """
    return (params < 0).any()


def is_state_invalid(arr_u, arr_v):
    abs_u = np.abs(arr_u.astype(np.float16))
    abs_v = np.abs(arr_v.astype(np.float16))
    return (arr_u < 0).any() or (arr_v < 0).any() \
           or np.isnan(np.min(abs_u)) or np.isnan(np.min(abs_v)) \
           or np.isinf(np.max(abs_u)) or np.isinf(np.max(abs_v))
           
           
def is_morph_invalid(img_morph, cval=116, min_cp=0.0, max_cp=0.47):
    """Check whether it is trivial by calculating color proportion.
       The valid color proportion ~ (min_cp, max_cp]
    
    Args:
        img_morph: PIL image object.
        cval: bright color such as red.
        min_cp: minimum valid color proportion.
        max_cp: maximum valid color proportion.

    """
    img = img_morph.convert("L")
    arr = np.array(img)
                
    n_total = arr.size
    n_reds = (arr == cval).sum()
    cpval = n_reds / n_total
    
    return cpval <= min_cp or cpval > max_cp