from lpf.array import get_array_module
from lpf.initializers import Initializer

class ReactionDiffusionModel(object):

    def __init__(self, device=None):
        self._am = get_array_module(device)

    @property
    def am(self):  # ArrayModule object
        return self._am

    @property
    def name(self):
        return self._name

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
    
    @property
    def initializer(self):
        return self._initializer

    @initializer.setter
    def initializer(self, obj):
        if not isinstance(obj, Initializer):
            raise TypeError("initializer should be a subclass of lpf.initializers.Initializer.")

        self._initializer = obj

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, obj):
        if not self.am.is_ndarray(obj):
            raise TypeError("params should be a ndarray of %s."%(self.am.__name__))

        self._params = obj

    @property
    def n_states(self):
        return self._n_states

    @property
    def y_linear(self):
        return self._y_linear

    @property
    def y_mesh(self):
        return self._y_mesh

    def initialize(self):
        self._initializer.initialize(self)

    def update(self):
        raise NotImplementedError()
        
    def is_early_stopping(self, rtol):       
        raise NotImplementedError()
        
    def save_image(self, index, fpath):
        raise NotImplementedError()
        
    def save_states(self, index, fpath):
        raise NotImplementedError()
        
    def save_model(self, index, fpath):
        raise NotImplementedError()
        
    def get_param_bounds(self):
        raise NotImplementedError()

    def check_invalid_values(self):
        raise NotImplementedError()
