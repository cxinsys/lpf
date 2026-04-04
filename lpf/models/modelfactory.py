from lpf.models import LiawModel
from lpf.models import GrayScottModel
from lpf.models import GiererMeinhardtModel
from lpf.models import SchnakenbergModel
from lpf.models import TwoComponentDiploidModel
from lpf.models import BrusselatorModel
from lpf.models import FitzHughNagumoModel
from lpf.models import LengyelEpsteinModel
from lpf.models import ThomasModel
from lpf.models import BarkleyModel


class ModelFactory:

    @staticmethod
    def create(name, *args, **kwargs):
        _name = name.lower()

        if "liaw" in _name:
            return LiawModel(*args, **kwargs)

        if "grayscott" in _name:
            return GrayScottModel(*args, **kwargs)

        if "gierermeinhardt" in _name:
            return GiererMeinhardtModel(*args, **kwargs)

        if "schnakenberg" in _name:
            return SchnakenbergModel(*args, **kwargs)

        if "twocomponentdiploid" in _name:
            return TwoComponentDiploidModel(*args, **kwargs)

        if "brusselator" in _name:
            return BrusselatorModel(*args, **kwargs)

        if "fitzhughnagumo" in _name:
            return FitzHughNagumoModel(*args, **kwargs)

        if "lengyelepstein" in _name:
            return LengyelEpsteinModel(*args, **kwargs)

        if "thomas" in _name:
            return ThomasModel(*args, **kwargs)

        if "barkley" in _name:
            return BarkleyModel(*args, **kwargs)

        raise ValueError("%s is not a supported model."%(name))