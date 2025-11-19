import os
import os.path as osp
from os.path import join as pjoin
import random
import time

import numpy as np
from numpy.random import default_rng

import lpf

from lpf.initializers import TwoComponentInitializer
from lpf.initializers import LiawInitializer
from lpf.initializers import InitializerFactory

from lpf.models import ReactionDiffusionModel
from lpf.models import Diploidy
from lpf.models import ModelFactory
from lpf.models import TwoComponentModel
from lpf.models import LiawModel
from lpf.models import TwoComponentDiploidModel


class RandomTwoComponentDiploidReproducer(object):

    def __init__(self,
                 population=None,
                 solver=None,
                 n_generations=100,
                 pop_size=32,
                 n_cross=4,
                 n_gametes=32,
                 prob_crossover=0.3,
                 autosomal=True,
                 alpha=0.5,
                 beta=0.5,
                 diploid_model_class=None,
                 haploid_model_class=None,
                 haploid_initializer_class=None,
                 dpath_output=None,
                 device="cpu",
                 verbose=0):
        """Evolve a population of diploid models.

        Args:
            population (list, Sequence): Initial population of models.
            solver (lpf.solvers.Solver): LPF solver object.
            n_generations (int, optional): Number of generations.
            pop_size (int, optional): Size of population (number of organisms).
            n_cross (int, optional): Number of crossing experiments.
            n_gametes (int, optional): Number of gametes (number of daughter cells).
            prob_crossover (float, optional): Probability that crossover occurs.
            autosomal (bool, optional): Whether sex affects inheritance.
            alpha (float): Paternal coefficient of the linear combination in two-component system.
            beta (float): Maternal coefficient of the linear combination in two-component system.
            diploid_model_class (lpf.models.TwoComponentDiploidModel):
            haploid_model_class (lpf.models.TwoComponentModel):
            haploid_initializer_class (lpf.models.TwoComponentInitializer):
            dpath_output (str, optional): Directory path of output.
            device (str, optional): Computing device.
            verbose (int, optional): Verbosity of messages for progress information.
        """

        if not population:
            raise ValueError("population must be given.")

        self._population = [population[:]]
        self._population_dicts = [[]]

        if not solver:
            raise ValueError("solver must be given.")

        if not isinstance(solver, lpf.solvers.Solver):
            raise TypeError("The type of solver must be a subclass of lpf.solvers.Solver.")

        self._solver = solver

        if diploid_model_class is None:
           diploid_model_class = TwoComponentDiploidModel
        elif not issubclass(diploid_model_class, lpf.models.TwoComponentDiploidModel):
            raise TypeError("diploid_model_class must be a subclass of lpf.models.TwoComponentDiploidModel.")

        if haploid_model_class is None:
            haploid_model_class = LiawModel
        elif not issubclass(haploid_model_class, lpf.models.TwoComponentModel):
            raise TypeError("diploid_model_class must be a subclass of lpf.models.TwoComponentModel.")

        if haploid_initializer_class is None:
            haploid_initializer_class = LiawInitializer
        elif not issubclass(haploid_initializer_class, lpf.initializers.TwoComponentInitializer):
            raise TypeError("diploid_model_class must be a subclass of lpf.initializers.TwoComponentInitializer.")

        
        self._diploid_model_class = diploid_model_class
        self._haploid_model_class = haploid_model_class
        self._haploid_initializer_class = haploid_initializer_class
        self._n_generations = n_generations
        self._pop_size = pop_size
        self._n_cross = n_cross
        self._n_gametes = n_gametes
        self._prob_crossover = prob_crossover
        self._autosomal = autosomal

        if alpha < 0:
            raise ValueError("alpha should be greater than 0.")

        if beta < 0:
            raise ValueError("beta should be greater than 0.")

        if (alpha + beta) > 1.0:
            raise ValueError("Summation of alpha and beta must be equal to or less than 1.0.")

        self._alpha = alpha
        self._beta = beta

        self._dpath_output = dpath_output
        self._device = device
        self._verbose = verbose

    @property
    def population(self):
        return self._population
    
    @property
    def population_dicts(self):
        return self._population_dicts

    @property
    def solver(self):
        return self._sovler

    @property
    def n_generations(self):
        return self._n_generations

    @property
    def pop_size(self):
        return self._pop_size

    @property
    def n_cross(self):
        return self._n_cross

    @property
    def n_gametes(self):
        return self._n_gametes

    @property
    def prob_crossover(self):
        return self._prob_crossover

    @property
    def autosomal(self):
        return self._autosomal

    @property
    def dpath_output(self):
        return self._dpath_output

    @property
    def device(self):
        return self._device

    @property
    def verbose(self):
        return self._verbose

    def generate_gametes(self, model, n_gametes, prob_crossover=0.5):

        if not isinstance(model, Diploidy):
            raise TypeError("model must be a subclass of Diploidy.")

        n_crossover = int(n_gametes * prob_crossover)
        n_haploid = n_gametes // 2

        pa_model = model.paternal_model
        ma_model = model.maternal_model

        pa_params = pa_model.am.copy(pa_model.params)
        ma_params = ma_model.am.copy(ma_model.params)

        pa_params = pa_model.am.repeat(pa_params, repeats=n_haploid, axis=0)
        ma_params = ma_model.am.repeat(ma_params, repeats=n_haploid, axis=0)

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
            
            # Number of individuals that undergo crossover.
            n_haploid_crossover = n_crossover // 2
            
            # Number of individuals that do NOT undergo crossover.
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
        params = pa_model.am.vstack([pa_params, ma_params])
        
        # Shuffle the values along the batch axis.
        np.random.shuffle(init_states)
        np.random.shuffle(init_pts)
        np.random.shuffle(params)

        return init_states, init_pts, params

    def cross(self,
              male_model,
              female_model,
              n_progenies,
              n_gametes,
              prob_crossover=0.5,
              autosomal=True,
              device="cpu"):

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

        # Randomly select the gametes of male and female.
        #ind_male = np.random.randint(low=0, high=n_gametes, size=n_progenies)
        #ind_female = np.random.randint(low=0, high=n_gametes, size=n_progenies)
        rng = default_rng()
        ind_male = rng.choice(n_gametes, size=n_progenies, replace=False)
        ind_female = rng.choice(n_gametes, size=n_progenies, replace=False)


        male_init_states, male_init_pts, male_params \
            = male_init_states[ind_male, :], male_init_pts[ind_male, :], male_params[ind_male, :]
        female_init_states, female_init_pts, female_params \
            = female_init_states[ind_female, :], female_init_pts[ind_female, :], female_params[ind_female, :]

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

    def evolve(self, n_generations=None, verbose=None):
        """Evolve a population of diploid models.

        Args:
            n_generations (int, optional): Number of generations.
            verbose (int, optional): Verbosity of messages for progress information.
        """

        if not verbose:
            verbose = self._verbose

        if not n_generations:
            n_generations = self._n_generations

        n_progenies_per_cross = self._pop_size // self._n_cross

        fstr_gen = "generation-%0{}d".format(int(np.floor(np.log10(n_generations))) + 1)
        fstr_duration = "[Generation #%d] Elapsed time: %f sec."

        str_gen = fstr_gen % (0)

        if self._dpath_output:
            dpath_gen = pjoin(self._dpath_output, str_gen)
            os.makedirs(dpath_gen, exist_ok=True)
        
        for i, model in enumerate(self._population[0]):
            str_id = "%s_model-%d" % (str_gen, i + 1)
            self._population_dicts[0].append({
                "ID": str_id,
                "MODEL": model,
                "PATERNAL": None,
                "MATERNAL": None,
                "MORPH": None,
                "MODEL_DICT": model.to_dict()
            })

            if self._dpath_output:
                fpath_model = pjoin(dpath_gen, "model_%s.json" % (str_id))
                model.save_model(index=0, fpath=fpath_model)
        # end of for

        for i in range(1, n_generations):
            
            t_beg = time.time()

            # Add a list that contains the organisms of this generation.
            self._population.append([])
            self._population_dicts.append([])

            # Create a directory that contains the files of this generation.
            str_gen = fstr_gen % (i)

            # [DEBUG]
            print(i, str_gen)
            
            if self._dpath_output:
                dpath_gen = pjoin(self._dpath_output, str_gen)
                os.makedirs(dpath_gen, exist_ok=True)

            cnt_progenies = 0

            # Repeat crossing experiments.
            for j in range(self._n_cross):
                male_model_dict = random.choice(self._population_dicts[i - 1])
                female_model_dict = random.choice(self._population_dicts[i - 1])

                id_male = male_model_dict["ID"]
                male_model = male_model_dict["MODEL"]

                id_female = female_model_dict["ID"]
                female_model = female_model_dict["MODEL"]

                pa_model, ma_model = self.cross(male_model=male_model,
                                                female_model=female_model,
                                                n_progenies=n_progenies_per_cross,
                                                n_gametes=self._n_gametes,
                                                prob_crossover=self._prob_crossover,
                                                autosomal=self._autosomal,
                                                device=self._device)

                # Create the model.
                model = self._diploid_model_class(
                    paternal_model=pa_model,
                    maternal_model=ma_model,
                    alpha=self._alpha,
                    beta=self._beta,
                    device=self._device
                )

                # Perform a numerical simulation.
                # self._solver.solve(model=model)
                self._solver.solve(model=model, verbose=1)  # [DEBUG]

                # Colorize the morphs from the states.
                arr_color = model.colorize()

                # Update and save the progenies.
                for k in range(n_progenies_per_cross):
                    img_ladybird, img_pattern = model.create_image(k, arr_color)

                    # Paternal model
                    initializer = self._haploid_initializer_class(
                        init_states=pa_model.initializer.init_states[None, k, :],
                        init_pts=pa_model.initializer.init_pts[None, k, :, :]
                    )

                    progeny_pa_model = self._haploid_model_class(
                        initializer=initializer,
                        params=pa_model.params[None, k, :],
                        width=model.width,
                        height=model.height,
                        dx=model.dx,
                        device=self._device
                    )

                    # Maternal model
                    initializer = LiawInitializer(
                        init_states=ma_model.initializer.init_states[None, k, :],
                        init_pts=ma_model.initializer.init_pts[None, k, :, :]
                    )

                    progeny_ma_model = self._haploid_model_class(
                        initializer=initializer,
                        params=ma_model.params[None, k, :],
                        width=model.width,
                        height=model.height,
                        dx=model.dx,
                        device=self._device
                    )

                    progeny_model = TwoComponentDiploidModel(
                        paternal_model=progeny_pa_model,
                        maternal_model=progeny_ma_model,
                        alpha=self._alpha,
                        beta=self._beta,
                        device=self._device
                    )

                    cnt_progenies += 1
                    str_id = "%s_model-%d" % (str_gen, cnt_progenies)
                    
                    self._population[i].append(model)                        
                        
                    self._population_dicts[i].append({
                        "ID": str_id,
                        "MODEL": progeny_model,
                        "PATERNAL": id_male,
                        "MATERNAL": id_female,
                        "MORPH": img_ladybird,
                        "MODEL_DICT": model.to_dict()
                    })

                    if self._dpath_output:
                        fpath_model = pjoin(dpath_gen, "model_%s.json" % (str_id))
                        fpath_morph = pjoin(dpath_gen, "ladybird_%s.png" % (str_id))
                        fpath_pattern = pjoin(dpath_gen, "pattern_%s.png" % (str_id))

                        progeny_model.save_model(index=0, fpath=fpath_model)
                        img_ladybird.save(fpath_morph)
                        img_pattern.save(fpath_pattern)
                    # end of if

                # end of for k in range(n_progenies_per_cross)
            # end of for j in range(n_cross)

            # [!] pop_size = n_cross * n_progenies_per_cross

            t_end = time.time()
            if verbose > 0:
                print(fstr_duration % (i, t_end - t_beg))
        # end of for i in range(1, n_generations)

        return self._population, self._population_dicts
