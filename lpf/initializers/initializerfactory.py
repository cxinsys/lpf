import glob
import os.path as osp
from os.path import join as pjoin

from lpf.initializers.initializer import Initializer
from lpf.initializers.liawinitializer import LiawInitializer
from lpf.utils import get_module_dpath

class InitializerFactory:
    
    @staticmethod
    def create(name):
        _name = name.lower()

        if "liaw" in _name:
            return LiawInitializer(_name)
        
        raise ValueError("%s is not a supported initializer."%(name))


    @staticmethod
    def create_all(substr):
        _substr = substr.lower()        
                                   
        dpath_data = pjoin(get_module_dpath("data"), "haxyridis")
        fpath_patt = pjoin(dpath_data, "init", "*%s*.png"%(substr))
        
        initializers = []
        name_to_init = {}
        name_to_index = {}
        
        for i, fpath in enumerate(glob.glob(fpath_patt)): 
            fname_and_ext = osp.basename(fpath)
            fname, ext = osp.splitext(fname_and_ext)
            
            initializer = LiawInitializer(fname)
            initializers.append(initializer)
            name_to_init[fname] = initializer
            name_to_index[fname] = i
        # end of for
        
        return initializers, name_to_init, name_to_index
            
                        