from lpf.models import TwoComponentDiploidModel


class TwoComponentCrosstalkDiploidModel(TwoComponentDiploidModel):

    def initialize(self):
        pa_model = self._paternal_model
        ma_model = self._maternal_model

        pa_model.initialize()
        ma_model.initialize()
        
        with self.am:
            self._shape_grid = (self.n_states,
                                self.batch_size,
                                self.height,
                                self.width)
            
            self._y_mesh = self.am.zeros(self._shape_grid,
                                         dtype=pa_model.params.dtype)

            alpha = self._alpha
            beta = self._beta

            # The total u and v are determined by a linear combination of paternal and maternal u and v.
            self._y_mesh[0, :] = alpha * pa_model._y_mesh[0, :] + beta * ma_model._y_mesh[0, :]
            self._y_mesh[1, :] = alpha * pa_model._y_mesh[1, :] + beta * ma_model._y_mesh[1, :]
            
            self._u = self._y_mesh[0, :]
            self._v = self._y_mesh[1, :]
            
            # self._y_linear = self._y_mesh.ravel()
             
            self._dydt_mesh = self.am.zeros(self._shape_grid,
                                            dtype=pa_model.params.dtype)

    def pdefunc(self, t, y_mesh=None, y_linear=None):
        """Equation function for integration.
        """
        
        pa_model = self._paternal_model
        ma_model = self._maternal_model

        with self.am:
            
            alpha = self.alpha
            beta = self.beta            
            
            dydt_mesh_pa = pa_model.pdefunc(t, y_mesh=y_mesh)
            dydt_mesh_ma = ma_model.pdefunc(t, y_mesh=y_mesh)
        
            self._dydt_mesh[0, :] = alpha * pa_model._dydt_mesh[0, :] + beta * ma_model._dydt_mesh[0, :]
            self._dydt_mesh[1, :] = alpha * pa_model._dydt_mesh[1, :] + beta * ma_model._dydt_mesh[1, :]
            
            # The total u and v are determined by a linear combination of paternal and maternal u and v.
            # self._u[:] = alpha * pa_model._u + beta * ma_model._u
            # self._v[:] = alpha * pa_model._v + beta * ma_model._v
            
        return self._dydt_mesh

# end of class TwoComponentCrosstalkDiploidModel
