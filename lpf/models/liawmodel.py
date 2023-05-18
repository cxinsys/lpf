import json
from collections.abc import Sequence

import numpy as np

from lpf.models import TwoComponentModel


class LiawModel(TwoComponentModel):

    def __init__(self, *args, **kwargs):

        # Set the device.
        super().__init__(*args, **kwargs)

        # Set constant members.
        self._name = "LiawModel"

    def reactions(self, t, u_c, v_c):
        batch_size = self.params.shape[0]

        ru = self.params[:, 2].reshape(batch_size, 1, 1)
        rv = self.params[:, 3].reshape(batch_size, 1, 1)

        k = self.params[:, 4].reshape(batch_size, 1, 1)

        su = self.params[:, 5].reshape(batch_size, 1, 1)
        sv = self.params[:, 6].reshape(batch_size, 1, 1)
        mu = self.params[:, 7].reshape(batch_size, 1, 1)
                
        try:
            f = ru * ((u_c ** 2 * v_c) / (1 + k * u_c ** 2)) + su - mu * u_c
            g = -rv * ((u_c ** 2 * v_c) / (1 + k * u_c ** 2)) + sv
        except FloatingPointError as err:
            raise err
        
        return f, g
    
    def to_dict(self,
                index=None,
                params=None,
                initializer=None,
                solver=None,
                generation=None,
                fitness=None):

        if params is None:
            if self._params is None:
                raise ValueError("params should be given.")

            params = self._params

        # Get the dict from the parent class.
        n2v = super().to_dict(index=index,
                              initializer=initializer,
                              params=params,
                              solver=solver,
                              generation=generation,
                              fitness=fitness)
       
        # Add the kinetic parameters to the parent dict.
        n2v["Du"] = float(params[index, 0])
        n2v["Dv"] = float(params[index, 1])
        n2v["ru"] = float(params[index, 2])
        n2v["rv"] = float(params[index, 3])
        n2v["k"]  = float(params[index, 4])
        n2v["su"] = float(params[index, 5])
        n2v["sv"] = float(params[index, 6])
        n2v["mu"] = float(params[index, 7])      
                   
        return n2v
   
    @staticmethod
    def parse_params(model_dicts):
        """Parse the parameters from the model dictionaries.
           A model knows how to parse its parameters.
        """
        if not isinstance(model_dicts, Sequence):
            raise TypeError("model_dicts should be a sequence of model dictionary.")

        batch_size = len(model_dicts)
        params = np.zeros((batch_size, 8), dtype=np.float64)

        for index, n2v in enumerate(model_dicts):
            params[index, 0] = n2v["Du"]
            params[index, 1] = n2v["Dv"]
            params[index, 2] = n2v["ru"]
            params[index, 3] = n2v["rv"]
            params[index, 4] = n2v["k"]
            params[index, 5] = n2v["su"]
            params[index, 6] = n2v["sv"]
            params[index, 7] = n2v["mu"]

        return params

    def get_param_bounds(self):
        n_init_pts = self._n_init_pts

        if not hasattr(self, "bounds_min"):
            self.bounds_min = self.am.zeros((10 + 2 * n_init_pts), dtype=np.float64)
            
        if not hasattr(self, "bounds_max"):
            self.bounds_max = self.am.zeros((10 + 2 * n_init_pts), dtype=np.float64)
        
        # Du
        self.bounds_min[0] = -4
        self.bounds_max[0] = 0
        
        # Dv
        self.bounds_min[1] = -4
        self.bounds_max[1] = 0
        
        # ru
        self.bounds_min[2] = -2
        self.bounds_max[2] = 2
        
        # rv
        self.bounds_min[3] = -2
        self.bounds_max[3] = 2        
        
        # k
        self.bounds_min[4] = -4
        self.bounds_max[4] = 0
        
        # su
        self.bounds_min[5] = -4
        self.bounds_max[5] = 0
        
        # sv
        self.bounds_min[6] = -4
        self.bounds_max[6] = 0
        
        # mu
        self.bounds_min[7] = -3
        self.bounds_max[7] = -1
        
        # u0
        self.bounds_min[8] = 0
        self.bounds_max[8] = 1.5

        # v0
        self.bounds_min[9] = 0
        self.bounds_max[9] = 1.5
        
        # init coords (25 points).     
        for index in range(10, 2 * n_init_pts, 2):
            self.bounds_min[index] = 0
            self.bounds_max[index] = self._height - 1
        # end of for

        for index in range(11, 2 * n_init_pts, 2):
            self.bounds_min[index] = 0
            self.bounds_max[index] = self._width - 1
        # end of for
        
        return self.bounds_min, self.bounds_max

    def len_decision_vector(self):  # length of the decision vector in PyGMO
        return 10 + 2 * self._n_init_pts

# end of class LiawModel
