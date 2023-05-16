from lpf.models import LiawModel
from lpf.models import GrayScottModel
from lpf.models import TwoStateDiploidModel

class ModelFactory:
    
    @staticmethod
    def create(name, *args, **kwargs):
        _name = name.lower()

        if "liaw" in _name:
            return LiawModel(*args, **kwargs)

        if "grayscott" in _name:
            return GrayScottModel(*args, **kwargs)

        if "twostatediploid" in _name:
            return TwoStateDiploidModel(*args, **kwargs)

        raise ValueError("%s is not a supported model."%(name))