from lpf.initializers.TwoComponentConstantInitializer import TwoComponentConstantInitializer
from lpf.initializers.liawinitializer import LiawInitializer

class InitializerFactory:
    
    @staticmethod
    def create(name, *args, **kwargs):
        _name = name.lower()

        if "twocomponentconstant" in _name:
            return TwoComponentConstantInitializer(*args, **kwargs)

        if "liaw" in _name:
            return LiawInitializer(*args, **kwargs)
        
        raise ValueError("%s is not a supported initializer."%(name))