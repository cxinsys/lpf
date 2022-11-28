from lpf.converters.liawconverter import LiawConverter

class ConverterFactory:
    
    @staticmethod
    def create(name):
        _name = name.lower()

        if "liaw" in _name:
            return LiawConverter()
        
        raise ValueError("%s is not a supported converter."%(name))