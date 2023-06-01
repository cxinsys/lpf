from lpf.initializers import Initializer


class TwoComponentInitializer(Initializer):

    def __init__(self, name, init_states=None, init_pts=None, dtype=None):
        super().__init__(name=name,
                         init_states=init_states,
                         init_pts=init_pts,
                         dtype=dtype)

    def update(self, model_dicts):
        raise NotImplementedError()

    def initialize(self, model, init_states=None, init_pts=None):
        raise NotImplementedError()

    def to_dict(self, index):
        raise NotImplementedError()
