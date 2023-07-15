from abc import ABC
from lpf.array import get_array_module


class ReactionDiffusionModel(ABC):

    def __init__(self, device=None):
        self._am = get_array_module(device)

    @property
    def am(self):  # ArrayModule object
        return self._am

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def dx(self):
        return self._dx
    
    @property
    def shape_grid(self):
        return self._shape_grid
    
    @property
    def initializer(self):
        return self._initializer

    @initializer.setter
    def initializer(self, obj):
        self._initializer = obj

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, obj):
        with self.am:
            self._params = self.am.array(obj, dtype=obj.dtype)

        if self._params is not None:
            self._batch_size = self._params.shape[0]
        
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_states(self):
        return self._n_states
    @property
    def y_mesh(self):
        return self._y_mesh

    @property
    def thr_color(self):
        return self._thr_color

    def initialize(self):
        self._initializer.initialize(self)
        
    def has_initializer(self):
        return self._initializer is not None

    def laplacian2d(self, a, dx):
        raise NotImplementedError()

    def reactions(self, t, u_c, v_c):
        raise NotImplementedError()

    def pdefunc(self, t, y_mesh=None, y_linear=None):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()
        
    def is_early_stopping(self, rtol):       
        raise NotImplementedError()

    def colorize(self, thr_color=None):
        raise NotImplementedError()

    def create_image(self, index=0, arr_color=None):
        raise NotImplementedError()

    def to_dict(self):
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
