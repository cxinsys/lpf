from lpf.converters.liawconverter import LiawConverter
from lpf.converters.grayscottconverter import GrayScottConverter
from lpf.converters.schnakenbergconverter import SchnakenbergConverter
from lpf.converters.gierermeinhardtconverter import GiererMeinhardtConverter

class ConverterFactory:

    @staticmethod
    def create(name):
        _name = name.lower()

        if "liaw" in _name:
            return LiawConverter()
        if "grayscott" in _name or "gray_scott" in _name:
            return GrayScottConverter()
        if "schnakenberg" in _name:
            return SchnakenbergConverter()
        if "gierermeinhardt" in _name or "gierer_meinhardt" in _name:
            return GiererMeinhardtConverter()

        raise ValueError("%s is not a supported converter."%(name))