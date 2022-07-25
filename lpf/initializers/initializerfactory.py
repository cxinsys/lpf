import glob
import os.path as osp
from os.path import join as pjoin

from lpf.initializers.initializer import Initializer
from lpf.initializers.liawinitializer import LiawInitializer
from lpf.utils import get_module_dpath


class InitializerFactory:
    
    @staticmethod
    def create(name, device=None):
        _name = name.lower()

        if "liaw" in _name:
            return LiawInitializer(_name, device=device)
        
        raise ValueError("%s is not a supported initializer."%(name))