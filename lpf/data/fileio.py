import os
import os.path as osp
import json
from typing import Union

from PIL import Image

from lpf.initializers import InitializerFactory
from lpf.initializers import Initializer
from lpf.models import ModelFactory
from lpf.models import ReactionDiffusionModel
from lpf.utils import get_module_dpath


def load_model_dicts(dpath):
    model_dicts = []

    for entity in os.listdir(dpath):
        fpath = osp.join(dpath, entity)
        if not osp.isfile(fpath) or not entity.endswith("json"):
            continue
        
        with open(fpath, "rt") as fin:
            n2v = json.load(fin)
        model_dicts.append(n2v)

    return model_dicts


def load_targets(ladybird_type, ladybird_subtypes, resize_shape=None):
    ladybird_type = ladybird_type.lower()
    ladybird_subtypes = [elem.lower() for elem in ladybird_subtypes]

    targets = []

    dpath = osp.join(get_module_dpath("data"), ladybird_type, "target")

    for fname in os.listdir(dpath):
        if not fname.startswith("ladybird"):
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


def load_custom_targets(dpath, file_header, resize_shape=None):
    targets = []

    for entity in os.listdir(dpath):
        fpath = osp.join(dpath, entity)
        if "~lock." in entity:
            continue
        
        if osp.isfile(fpath) and entity.startswith(file_header) and entity.endswith(".png"):

            fpath = osp.join(dpath, entity)
            img = Image.open(fpath)
    
            img_no_trans = Image.new("RGBA", img.size, "WHITE")
            img_no_trans.paste(img, mask=img)
            img = img_no_trans
    
            if resize_shape:
                img = img.resize(resize_shape)
    
            targets.append(img.convert('RGB'))
            print("[TARGET] %s has been added..." % (entity))
    # end of for

    return targets



def load_as_array(dpath: Union[str, os.PathLike],
                  initializer: Union[str, Initializer],
                  model: Union[str, ReactionDiffusionModel]):
    """Load the initial points, initial states, and parameters 
       as array from the model JSON files.
    """
    model_dicts = []
    for entity in os.listdir(dpath):
        if not entity.startswith("model_"):
             continue
         
        fpath_model = osp.join(dpath, entity)
 
        if not osp.isfile(fpath_model):
            raise FileNotFoundError(fpath_model)
     
        with open(fpath_model, "rt") as fin:
            n2v = json.load(fin)
            model_dicts.append(n2v)
       # end of for   

    # Create initializer      
    if isinstance(initializer, str):
        initializer = InitializerFactory.create(
            name=initializer,
        )
    elif isinstance(initializer, Initializer):
        pass
    else:
        raise TypeError("model should be str or ReactionDiffusionModel.")
    
    # Update the initializer.
    initializer.update(model_dicts)
    
    # Create a model.
    if isinstance(model, str):
        model = ModelFactory.create(
            name=model,
            initializer=initializer,
        )
    elif isinstance(model, ReactionDiffusionModel):
        pass
    else:
        raise TypeError("model should be str or ReactionDiffusionModel.")
    
    params = model.parse_params(model_dicts)
    
    return initializer.init_pts, initializer.init_states, params
    