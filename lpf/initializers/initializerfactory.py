import glob
import os.path as osp
from os.path import join as pjoin

from lpf.initializers.initializer import Initializer
from lpf.initializers.liawinitializer import LiawInitializer
from lpf.utils import get_module_dpath


class InitializerFactory:
    
    @staticmethod
    def create(name, *args, **kwargs):
        _name = name.lower()

        if "liaw" in _name:
            return LiawInitializer(*args, **kwargs)
        
        raise ValueError("%s is not a supported initializer."%(name))