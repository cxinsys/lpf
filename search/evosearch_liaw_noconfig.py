import os
import os.path as osp
import time

import numpy as np
import pygmo as pg
from PIL import Image

from lpf.data import load_model_dicts
from lpf.data import load_targets
from lpf.solvers import SolverFactory
from lpf.search import EvoSearch
from lpf.objectives import ObjectiveFactory
from lpf.models import ModelFactory
from lpf.converters import ConverterFactory

np.seterr(all='raise')



if __name__ == "__main__":
    
    LPF_REPO_HOME = osp.abspath("..")
    LPF_REPO_HOME

   # Create a model.
    dx = 0.1
    width = 128
    height = 128
    n_init_pts = 25
    
    model = ModelFactory.create(
        name="Liaw",
        n_init_pts=n_init_pts,
        width=width,
        height=height,                 
        dx=dx
    )

    # Create a solver.
    dt = 0.01
    n_iters = 500000
    solver = SolverFactory.create(name="Euler", dt=dt, n_iters=n_iters)
    
    # Create a converter.
    converter = ConverterFactory.create("LiawInitializer")
    
    # Create objectives.
    obj_config = [
        ['MeanMeanSquareError', '1e-1', 'cpu'],
        ['MeanColorProportion', '1e0', 'cpu'],
        ['MeanVgg16PerceptualLoss', '1e-4', 'cuda:0'],
        ['MeanLearnedPerceptualImagePatchSimilarity:vgg', '1.5e1', 'cuda:0'],
        ['MeanLearnedPerceptualImagePatchSimilarity:alex', '4e0', 'cuda:0']
    ]
    
    objectives = ObjectiveFactory.create(obj_config)


    # Load the target laybirds.
    targets = []
    
    dpath_photos = osp.join(LPF_REPO_HOME, "lpf/data/haxyridis/photo")
    print("[DPATH PHOTOS]", dpath_photos)
    for entity in os.listdir(dpath_photos):
        fpath_photo = osp.join(dpath_photos, entity)        
        if osp.isfile(fpath_photo) and entity.startswith("spectabilis") and entity.endswith("png"):
            print(" - ", fpath_photo)
            img = Image.open(fpath_photo)
            targets.append(img)
    
    for img in targets:    
        print(img)


    # Create an evolutionary search problem.
    droot_output = osp.join("./output")
    
    search = EvoSearch(model=model,
                       solver=solver,
                       converter=converter,
                       targets=targets,
                       objectives=objectives,
                       droot_output=droot_output)
    
    prob = pg.problem(search)
    
    
    dpath_init_pop = osp.join(LPF_REPO_HOME, "population", "init_pop_axyridis")  
    model_dicts = load_model_dicts(dpath_init_pop)
    
    
    t_beg = time.time()

    # Create the initial population.
    pop_size = 1  # We set population size = 16.
    pop = pg.population(prob)
    dvs = []
    
    # Initialize the population with axyridis subtype.
    for i, param_dict in enumerate(model_dicts):
        if i >= pop_size:
            break
    
        dv = converter.to_dv(param_dict, n_init_pts)
        dvs.append(dv)
    
    # Adding decision vectors incorporates evaluating the fitness score.
    for i, dv in enumerate(dvs):
        print(f"[DECISION VECTOR #{i+1}]\n", dv)
    
        # pop.set_x(i, dv)      
        pop.push_back(dv) 
    
    # end of for
    
    t_end = time.time()
    
    print("[POPULATION INITIALIZATION COMPLETED]")
    print("- DURATION OF INITIALIZING POPULATION: %.3f sec."%(t_end - t_beg))
    print(pop)
    
    
    # Create an evolutionary algorithm.
    n_procs = 8
    n_gen = 10000
    
    udi = pg.mp_island()
    udi.resize_pool(n_procs)
    
    algo = pg.algorithm(pg.sade(gen=1))
    isl = pg.island(algo=algo, pop=pop, udi=udi)
    print(isl)
    
    
    # Start seraching.
    try:
        for i in range(n_gen):
            t_beg = time.time()
            isl.evolve()
            isl.wait_check()
            t_end = time.time()
    
            print("[EVOLUTION #%d] Best objective: %f (%.3f sec.)"%(i + 1, pop.champion_f[0], t_end - t_beg))
    
            # Save the best.
            pop = isl.get_population()
            search.save("best", pop.champion_x, generation=i+1, fitness=pop.champion_f[0])
    
            # Save the population.
            arr_x = pop.get_x()
            arr_f = pop.get_f()
            for j in range(arr_x.shape[0]):
                x = arr_x[j]
                fitness = arr_f[j, 0]
                search.save("pop", x, generation=i+1, fitness=fitness)
        # end of for
    except Exception as err:
        print(err)
        udi.shutdown_pool()
        raise err
    
    
    print("[EVOLUTIONARY SEARCH COMPLETED]")
    udi.shutdown_pool()
