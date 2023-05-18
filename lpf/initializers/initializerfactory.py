from lpf.initializers.twostateconstantinitializer import TwoStateConstantInitializer
from lpf.initializers.liawinitializer import LiawInitializer

class InitializerFactory:
    
    @staticmethod
    def create(name, *args, **kwargs):
        _name = name.lower()

        if "twostateconstant" in _name:
            return TwoStateConstantInitializer(*args, **kwargs)

        if "liaw" in _name:
            return LiawInitializer(*args, **kwargs)
        
        raise ValueError("%s is not a supported initializer."%(name))