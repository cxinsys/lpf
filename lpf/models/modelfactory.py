from lpf.models import LiawModel


class ModelFactory:
    
    @staticmethod
    def create(name, *args, **kwargs):
        _name = name.lower()

        if "liaw" in _name:
            return LiawModel(*args, **kwargs)

        raise ValueError("%s is not a supported model."%(name))