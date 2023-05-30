from lpf.models import TwoComponentModel
from lpf.models import Diploidy


class TwoComponentDiploidModel(Diploidy, TwoComponentModel):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if not isinstance(self._paternal_model, TwoComponentModel):
            raise TypeError(f"paternal_model must be a derivative of "
                             "TwoComponentModel class.")

        if not isinstance(self._maternal_model, TwoComponentModel):
            raise TypeError(f"maternal_model must be a derivative of "
                             "TwoComponentModel class.")

        # Set the name of model.
        self._name = "TwoComponentDiploidModel"

    def initialize(self):
        super().initialize()

        pa_model = self._paternal_model
        ma_model = self._maternal_model

        alpha = self._alpha
        beta = self._beta

        with self.am:
            self._u = alpha * pa_model._u + beta * ma_model._u
            self._v = alpha * pa_model._v + beta * ma_model._v
        
    def has_initializer(self):
        #return super(Diploidy, self).has_initializer()
        return Diploidy.has_initializer(self)

    def pdefunc(self, t, y_mesh=None, y_linear=None):
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

# end of class TwoComponentDiploidModel
