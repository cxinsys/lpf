from lpf.models import LiawModel
from lpf.models import GrayScottModel
from lpf.models import TwoComponentDiploidModel

class ModelFactory:
    
    @staticmethod
    def create(name, *args, **kwargs):
        _name = name.lower()

        if "liaw" in _name:
            return LiawModel(*args, **kwargs)

        if "grayscott" in _name:
            return GrayScottModel(*args, **kwargs)

        if "twocomponentdiploid" in _name:
            return TwoComponentDiploidModel(*args, **kwargs)

        raise ValueError("%s is not a supported model."%(name))