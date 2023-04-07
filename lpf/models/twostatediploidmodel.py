import json
from collections.abc import Sequence

import numpy as np
import PIL
from PIL import Image
 
from lpf.models import TwoStateModel
from lpf.initializers import Initializer
from lpf.solvers import Solver
from lpf.utils import get_template_fpath
from lpf.utils import get_mask_fpath


def check_model(model, name):
    if model is None:
        raise ValueError(f"{name} must be provided.") 
        
    if not isinstance(model, TwoStateModel):
        raise TypeError(f"{name} must be a derivative of TwoStateModel class.")
        
    if not hasattr(model, "initializer"):
        raise AttributeError("%s must have initializer member variable."%(name))

    if model.initializer is None:
        raise TypeError("%s must have an initializer."%(name))
   
    if not isinstance(model.initializer, Initializer):
        raise TypeError("%s.initializer must be a derivative of Initializer class."%(name))

    if model.n_states != 2:
        raise ValueError("%s.n_states must be two."%(name))


class TwoStateDiploidModel(TwoStateModel):

    def __init__(self,
                 *args,
                 paternal_model=None,
                 maternal_model=None,
                 **kwargs):

        # Call the __init__ of parent class.
        super().__init__(*args, **kwargs)
        
        # Set the name of model.
        self._name = "TwoStateDiploidModel"
        
        # Check types and members of paternal and maternal models.
        check_model(paternal_model, "paternal_model")        
        check_model(maternal_model, "maternal_model")
        
        if id(paternal_model) == id(maternal_model):
            raise ValueError("paternal_model and maternal_model "\
                             "must be different objects.")

        self._paternal_model = paternal_model
        self._maternal_model = maternal_model


    def pdefunc(self, t, y_linear):
        """Equation function for integration.
        """        
        
        # [In LiawInitializer]
        # y_mesh = model.am.zeros(shape_grid, dtype=init_states.dtype)
        
        # model._u = y_mesh[0, :, :, :]
        # model._v = y_mesh[1, :, :, :]

        # model._y_linear = y_mesh.ravel()
        
        # model._dydt_mesh = model.am.zeros(shape_grid, dtype=init_states.dtype)
        # model._dydt_linear = model._dydt_mesh.ravel()
        
                
        # [In TwoStateModel]
        # batch_size = self.params.shape[0]

        # y_mesh = y_linear.reshape(self.n_states, batch_size, self.height, self.width)
        # dydt_mesh = self._dydt_mesh

        # u = y_mesh[0, :, :, :]
        # v = y_mesh[1, :, :, :]

        # # Model must update its states.
        # self._u = u
        # self._v = v
        # dx = self._dx
        
        # Get the kinetic parameters.
        # Du = self.params[:, 0].reshape(batch_size, 1, 1)
        # Dv = self.params[:, 1].reshape(batch_size, 1, 1)

        # u_c = u[:, 1:-1, 1:-1]
        # v_c = v[:, 1:-1, 1:-1]

        # # Reactions
        # f, g = self.reactions(t, u_c, v_c)

        # # Diffusions + Reactions
        # dydt_mesh[0, :, 1:-1, 1:-1] = Du * self.laplacian2d(u, dx) + f
        # dydt_mesh[1, :, 1:-1, 1:-1] = Dv * self.laplacian2d(v, dx) + g

        # # Neumann boundary condition: dydt = 0
        # dydt_mesh[:, :, 0, :] = 0.0
        # dydt_mesh[:, :, -1, :] = 0.0
        # dydt_mesh[:, :, :, 0] = 0.0
        # dydt_mesh[:, :, :, -1] = 0.0
        
        
        # [Update relationship]
        # y_linear -> y_mesh -> u, v -> self._u, self._v
        # dydt_mesh -> self._dydt_mesh -> self._dydt_linear
        
        
        pa_model = self._paternal_model
        ma_model = self._maternal_model
        
        with self.am:
            
            dydt_linear_pa = pa_model.pdefunc(t, y_linear)
            dydt_linear_ma = ma_model.pdefunc(t, y_linear)            
            
            self._u = 0.5 * (pa_model._u + ma_model._u)
            self._v = 0.5 * (pa_model._v + ma_model._v)
            
            self._dydt_linear[:] = 0.5 * (dydt_linear_pa + dydt_linear_ma)            
        
        return self._dydt_linear


    # @staticmethod
    # def parse_params(model_dicts):
    #     """Parse the parameters from the model dictionaries.
    #        A model knows how to parse its parameters.
    #     """
    #     if not isinstance(model_dicts, Sequence):
    #         raise TypeError("model_dicts should be a sequence of model dictionary.")

    #     batch_size = len(model_dicts)
    #     params = np.zeros((batch_size, 8), dtype=np.float64)

    #     for index, n2v in enumerate(model_dicts):
    #         params[index, 0] = n2v["Du"]
    #         params[index, 1] = n2v["Dv"]
    #         params[index, 2] = n2v["ru"]
    #         params[index, 3] = n2v["rv"]
    #         params[index, 4] = n2v["k"]
    #         params[index, 5] = n2v["su"]
    #         params[index, 6] = n2v["sv"]
    #         params[index, 7] = n2v["mu"]

    #     return params

    # @staticmethod
    # def parse_init_states(self, model_dicts):
    #     """Parse the initial states from the model dictionaries.
    #        A model knows how to parse its initial states.
    #     """
    #     if not isinstance(model_dicts, Sequence):
    #         raise TypeError("model_dicts should be a sequence of model dictionary.")

    #     batch_size = len(model_dicts)
    #     init_states = np.zeros((batch_size, 2), dtype=np.float64)

    #     for index, n2v in enumerate(model_dicts):
    #         init_states[index, 0] = n2v["u0"]
    #         init_states[index, 1] = n2v["v0"]
    #     # end of for

    #     return init_states

    # def get_param_bounds(self):
    #     n_init_pts = self._n_init_pts

    #     if not hasattr(self, "bounds_min"):
    #         self.bounds_min = self.am.zeros((10 + 2 * n_init_pts), dtype=np.float64)
            
    #     if not hasattr(self, "bounds_max"):
    #         self.bounds_max = self.am.zeros((10 + 2 * n_init_pts), dtype=np.float64)
        
    #     # Du
    #     self.bounds_min[0] = -4
    #     self.bounds_max[0] = 0
        
    #     # Dv
    #     self.bounds_min[1] = -4
    #     self.bounds_max[1] = 0
        
    #     # ru
    #     self.bounds_min[2] = -2
    #     self.bounds_max[2] = 2
        
    #     # rv
    #     self.bounds_min[3] = -2
    #     self.bounds_max[3] = 2        
        
    #     # k
    #     self.bounds_min[4] = -4
    #     self.bounds_max[4] = 0
        
    #     # su
    #     self.bounds_min[5] = -4
    #     self.bounds_max[5] = 0
        
    #     # sv
    #     self.bounds_min[6] = -4
    #     self.bounds_max[6] = 0
        
    #     # mu
    #     self.bounds_min[7] = -3
    #     self.bounds_max[7] = -1
        
    #     # u0
    #     self.bounds_min[8] = 0
    #     self.bounds_max[8] = 1.5

    #     # v0
    #     self.bounds_min[9] = 0
    #     self.bounds_max[9] = 1.5
        
    #     # init coords (25 points).     
    #     for index in range(10, 2 * n_init_pts, 2):
    #         self.bounds_min[index] = 0
    #         self.bounds_max[index] = self._height - 1
    #     # end of for

    #     for index in range(11, 2 * n_init_pts, 2):
    #         self.bounds_min[index] = 0
    #         self.bounds_max[index] = self._width - 1
    #     # end of for
        
    #     return self.bounds_min, self.bounds_max

    # def len_decision_vector(self):  # length of the decision vector in PyGMO
    #     return 10 + 2 * self._n_init_pts

# end of class LiawModel
