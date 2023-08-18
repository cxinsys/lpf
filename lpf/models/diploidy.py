from abc import ABC
import numbers

import numpy as np


def check_model(model, name):
    if model is None:
        raise ValueError(f"{name} must be provided.")

    if not hasattr(model, "initializer"):
        raise AttributeError("%s must have initializer member variable." % (name))

    if model.initializer is None:
        raise TypeError("%s must have an initializer." % (name))

    if model.n_states != 2:
        raise ValueError("%s.n_states must be two." % (name))


class Diploidy(ABC):

    def __init__(self,
                 *args,
                 paternal_model=None,
                 maternal_model=None,
                 alpha=0.5,
                 beta=0.5,
                 **kwargs):

        # Call the __init__ of parent class.
        super().__init__(*args, **kwargs)

        # Set the name of model.
        self._name = "DiploidModel"

        # Check types and members of paternal and maternal models.
        check_model(paternal_model, "paternal_model")
        check_model(maternal_model, "maternal_model")

        # Check the compatibilty between paternal and maternal models.
        if id(paternal_model) == id(maternal_model):
            raise ValueError("paternal_model and maternal_model " \
                             "must be different objects.")

        pa_dtype = paternal_model.dtype
        ma_dtype = maternal_model.dtype

        if pa_dtype != ma_dtype:
            raise TypeError("The dtype of paternal_model.params " \
                            "and maternal_model.parms must be the same.")

        self._dtype = pa_dtype

        if paternal_model.batch_size != maternal_model.batch_size:
            raise ValueError("The batch size of paternal_model " \
                             "and maternal_model must be the same.")

        self._batch_size = paternal_model.batch_size

        self._paternal_model = paternal_model
        self._maternal_model = maternal_model

        if isinstance(alpha, numbers.Number):
            self._alpha = alpha
        else:
            with self.am:
                alpha = np.expand_dims(alpha, axis=(1, 2))
                self._alpha = self.am.array(alpha, dtype=self._dtype)

        if isinstance(beta, numbers.Number):
            self._beta = beta
        else:
            with self.am:
                beta = np.expand_dims(beta, axis=(1, 2))
                self._beta = self.am.array(beta, dtype=self._dtype)

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

        # alpha = self._alpha
        # beta = self._beta

        pa_model.initialize()
        ma_model.initialize()

        with self.am:
            self._shape_grid = (2 * self.n_states,
                                self.batch_size,
                                self.height,
                                self.width)

            self._y_mesh = self.am.zeros(self._shape_grid,
                                         dtype=pa_model.dtype)

            self._y_mesh[0, :] = pa_model._y_mesh[0, :]
            self._y_mesh[1, :] = pa_model._y_mesh[1, :]
            self._y_mesh[2, :] = ma_model._y_mesh[0, :]
            self._y_mesh[3, :] = ma_model._y_mesh[1, :]

            # # The total u and v are determined by
            # # a linear combination of paternal and maternal u and v.
            # self._u = alpha * pa_model._u + beta * ma_model._u
            # self._v = alpha * pa_model._v + beta * ma_model._v

            # self._y_linear = self._y_mesh.ravel()

            self._dydt_mesh = self.am.zeros(self._shape_grid,
                                            dtype=pa_model.dtype)

    def has_initializer(self):
        return self._paternal_model.has_initializer() \
               and self._maternal_model.has_initializer()

    def pdefunc(self, t=None, y_mesh=None, y_linear=None):
        raise NotImplementedError()

    def to_dict(self,
                index=0,
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

        model_dict["model"] = self._name

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
