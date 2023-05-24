import numbers
import numpy as np

from lpf.models import TwoComponentModel
from lpf.solvers import Solver


def check_model(model, name):
    if model is None:
        raise ValueError(f"{name} must be provided.") 
        
    if not isinstance(model, TwoComponentModel):
        raise TypeError(f"{name} must be a derivative of TwoComponentModel class.")
        
    if not hasattr(model, "initializer"):
        raise AttributeError("%s must have initializer member variable."%(name))

    if model.initializer is None:
        raise TypeError("%s must have an initializer."%(name))
   
    # if not isinstance(model.initializer, Initializer):
    #     raise TypeError("%s.initializer must be a derivative of Initializer class."%(name))

    if model.n_states != 2:
        raise ValueError("%s.n_states must be two."%(name))


class TwoComponentDiploidModel(TwoComponentModel):

    def __init__(self,
                 *args,
                 paternal_model=None,
                 maternal_model=None,
                 initializer=None,
                 alpha=0.5,
                 beta=0.5,
                 **kwargs):

        # Call the __init__ of parent class.
        super().__init__(*args, **kwargs)
        
        # Set the name of model.
        self._name = "TwoComponentDiploidModel"
        
        # Check types and members of paternal and maternal models.
        check_model(paternal_model, "paternal_model")        
        check_model(maternal_model, "maternal_model")
        
        # Check the compatibilty between paternal and maternal models.
        if id(paternal_model) == id(maternal_model):
            raise ValueError("paternal_model and maternal_model "\
                             "must be different objects.")

        pa_dtype = paternal_model.params.dtype
        ma_dtype = maternal_model.params.dtype
         
        if pa_dtype != ma_dtype:
            raise TypeError("The dtype of paternal_model.params "\
                             "and maternal_model.parms must be the same.")
                
        dtype = pa_dtype
                
        if paternal_model.batch_size != maternal_model.batch_size:
            raise ValueError("The batch size of paternal_model "\
                             "and maternal_model must be the same.")
        
        self._batch_size = paternal_model.batch_size
                
        self._paternal_model = paternal_model
        self._maternal_model = maternal_model
        
        
        if isinstance(alpha, numbers.Number):            
            self._alpha = alpha
        else:
            with self.am:
                alpha = np.expand_dims(alpha, axis=(1, 2))
                self._alpha = self.am.array(alpha, dtype=dtype)
        
        if isinstance(beta, numbers.Number):            
            self._beta = beta
        else: 
            with self.am:
                beta = np.expand_dims(beta, axis=(1, 2))
                self._beta = self.am.array(beta, dtype=dtype)
        
        
    @property
    def paternal_model(self):
        return self._paternal_model
    
    @property
    def maternal_model(self):
        return self._maternal_model    
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def beta(self):
        return self._beta

    def initialize(self):
        pa_model = self._paternal_model
        ma_model = self._maternal_model
        
        alpha = self._alpha
        beta = self._beta
        
        pa_model.initialize()
        ma_model.initialize()
        
        with self.am:

            self._shape_grid = (2 * self.n_states,
                                self.batch_size,
                                self.height,
                                self.width)
            
            self._y_mesh = self.am.zeros(self._shape_grid,
                                         dtype=pa_model.params.dtype)

            self._y_mesh[0, :] = pa_model._y_mesh[0, :]
            self._y_mesh[1, :] = pa_model._y_mesh[1, :]
            self._y_mesh[2, :] = ma_model._y_mesh[0, :]
            self._y_mesh[3, :] = ma_model._y_mesh[1, :]

            # The total u and v are determined by
            # a linear combination of paternal and maternal u and v.
            self._u = alpha * pa_model._u + beta * ma_model._u
            self._v = alpha * pa_model._v + beta * ma_model._v
            
            self._y_linear = self._y_mesh.ravel()
             
            self._dydt_mesh = self.am.zeros(self._shape_grid,
                                            dtype=pa_model.params.dtype)
            self._dydt_linear = self._dydt_mesh.ravel()
            
        
    def has_initializer(self):
        return self._paternal_model.has_initializer() \
               and self._maternal_model.has_initializer()

    def pdefunc(self, t, y_mesh=None, y_linear=None, ):
        """Equation function for integration.
        """
        
        pa_model = self._paternal_model
        ma_model = self._maternal_model

        with self.am:
            
            alpha = self.alpha
            beta = self.beta            
            
            dydt_mesh_pa = pa_model.pdefunc(t, y_mesh=y_mesh[:2, :, : ,:])
            dydt_mesh_ma = ma_model.pdefunc(t, y_mesh=y_mesh[2:, :, : ,:])
        
            self._dydt_mesh[0, :] = dydt_mesh_pa[0, :]
            self._dydt_mesh[1, :] = dydt_mesh_pa[1, :]
            self._dydt_mesh[2, :] = dydt_mesh_ma[0, :]
            self._dydt_mesh[3, :] = dydt_mesh_ma[1, :]

            # The total u and v are determined by
            # a linear combination of paternal and maternal u and v.
            self._u[:] = alpha * pa_model._u + beta * ma_model._u
            self._v[:] = alpha * pa_model._v + beta * ma_model._v
            
        return self._dydt_mesh

    def to_dict(self,      
                index=None,
                initializer=None,
                params=None,
                solver=None,
                generation=None,
                fitness=None):
        
        model_dict = super().to_dict(index=index,
                                     initializer=initializer,
                                     solver=solver,
                                     generation=generation,
                                     fitness=fitness)       
        
        model_dict["model_name"] = self._name
          
            
        pa_model_dict = self.paternal_model.to_dict(
                                  index=index,
                                  generation=generation,
                                  fitness=fitness)
        
        ma_model_dict = self.maternal_model.to_dict(
                                  index=index,
                                  generation=generation,
                                  fitness=fitness)

        model_dict["paternal_model"] = pa_model_dict
        model_dict["maternal_model"] = ma_model_dict        
        
        
        if not isinstance(self._alpha, numbers.Number):
            with self.am:
                alpha = float(self._alpha[index])
        else:
            alpha = self._alpha
            
        if not isinstance(self._beta, numbers.Number):
            with self.am:
                beta = float(self._beta[index])
        else:
            beta = self._beta
            
        model_dict["alpha"] = alpha            
        model_dict["beta"] = beta
    
        return model_dict
    

# end of class TwoComponentDiploidModel
