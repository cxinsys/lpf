from lpf.initializers import InitializerFactory
from lpf.models import ReactionDiffusionModel
from lpf.models import Diploidy
from lpf.models import ModelFactory

import numpy as np


class RandomDiploidReproducer(object):

    def generate_gametes(self, model, n_gametes, prob_crossover=0.5):

        if not isinstance(model, Diploidy):
            raise TypeError("model must be a subclass of Diploidy.")

        n_crossover = int(n_gametes * prob_crossover)
        n_haploid = n_gametes // 2

        pa_model = model.paternal_model
        ma_model = model.maternal_model

        pa_params = pa_model.params.copy()
        ma_params = ma_model.params.copy()

        pa_params = np.repeat(pa_params, repeats=n_haploid, axis=0)
        ma_params = np.repeat(ma_params, repeats=n_haploid, axis=0)

        pa_init_states = pa_model.initializer.init_states.copy()
        ma_init_states = ma_model.initializer.init_states.copy()

        pa_init_states = np.repeat(pa_init_states, repeats=n_haploid, axis=0)
        ma_init_states = np.repeat(ma_init_states, repeats=n_haploid, axis=0)

        pa_init_pts = pa_model.initializer.init_pts.copy()
        ma_init_pts = ma_model.initializer.init_pts.copy()

        pa_init_pts = np.repeat(pa_init_pts, repeats=n_haploid, axis=0)
        ma_init_pts = np.repeat(ma_init_pts, repeats=n_haploid, axis=0)

        # Apply crossover.
        if prob_crossover > 0.0 and type(pa_model) == type(ma_model):
            n_haploid_crossover = n_crossover // 2
            i_beg = n_haploid - n_haploid_crossover

            # Crossover between initial states.
            shape = [n_haploid_crossover, *pa_init_states.shape[1:]]
            ind = np.random.choice([True, False], size=shape)
            pa_init_states[i_beg:][ind], ma_init_states[i_beg:][ind] \
                = ma_init_states[i_beg:][ind], pa_init_states[i_beg:][ind]

            # Crossover between initial points.
            shape = [n_haploid_crossover, *pa_init_pts.shape[1:]]
            ind = np.random.choice([True, False], size=shape)
            pa_init_pts[i_beg:][ind], ma_init_pts[i_beg:][ind] \
                = ma_init_pts[i_beg:][ind], pa_init_pts[i_beg:][ind]

            # Crossover between parameters
            shape = [n_haploid_crossover, *pa_params.shape[1:]]
            ind = np.random.choice([True, False], size=shape)
            pa_params[i_beg:][ind], ma_params[i_beg:][ind] \
                = ma_params[i_beg:][ind], pa_params[i_beg:][ind]
        # end of if

        init_states = np.vstack([pa_init_states, ma_init_states])
        init_pts = np.vstack([pa_init_pts, ma_init_pts])
        params = np.vstack([pa_params, ma_params])

        return init_states, init_pts, params

    def cross(self,
              male_model,
              female_model,
              n_progenies,
              n_gametes,
              prob_crossover=0.5,
              autosomal=True,
              device=None):

        if not isinstance(male_model, ReactionDiffusionModel):
            raise TypeError("male_model model must be a ReactionDiffusion model.")

        if not isinstance(female_model, ReactionDiffusionModel):
            raise TypeError("female_model model must be a ReactionDiffusion model.")

        if type(male_model.paternal_model) != type(male_model.maternal_model):
            raise TypeError("paternal and maternal models of male must be the same.")
            
        if type(female_model.paternal_model) != type(female_model.maternal_model):
            raise TypeError("paternal and maternal models of female must be the same.")

        male_init_states, male_init_pts, male_params \
            = self.generate_gametes(male_model, n_gametes, prob_crossover)
        female_init_states, female_init_pts, female_params \
            = self.generate_gametes(female_model, n_gametes, prob_crossover)

        ind = np.random.randint(low=0, high=n_gametes, size=n_progenies)

        male_init_states, male_init_pts, male_params \
            = male_init_states[ind, :], male_init_pts[ind, :], male_params[ind, :]
        female_init_states, female_init_pts, female_params \
            = female_init_states[ind, :], female_init_pts[ind, :], female_params[ind, :]

        name = male_model.paternal_model.initializer.name
        male_initializer = InitializerFactory.create(name,
                                                     init_states=male_init_states,
                                                     init_pts=male_init_pts)

        name = female_model.paternal_model.initializer.name
        female_initializer = InitializerFactory.create(name,
                                                       init_states=female_init_states,
                                                       init_pts=female_init_pts)

        name = male_model.paternal_model.name
        paternal_model = ModelFactory.create(name,
                                             initializer=male_initializer,
                                             params=male_params,
                                             device=device)

        name = female_model.paternal_model.name
        maternal_model = ModelFactory.create(name,
                                             initializer=female_initializer,
                                             params=female_params,
                                             device=device)

        return paternal_model, maternal_model
