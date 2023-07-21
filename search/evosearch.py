import os
import os.path as osp
from os.path import abspath as apath
import time
import argparse
from datetime import datetime

import yaml
import numpy as np
import pygmo as pg
from PIL import ImageColor

from lpf.data import load_model_dicts
from lpf.data import load_targets
from lpf.solvers import SolverFactory
from lpf.search import EvoSearch
from lpf.objectives import ObjectiveFactory
from lpf.models import ModelFactory
from lpf.converters import ConverterFactory

np.seterr(all='raise')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse configurations for searching.')
    parser.add_argument('--config',
                        dest='config',
                        action='store',
                        type=str,
                        help='Designate the file path of configuration file in YAML')

    parser.add_argument('--gpu',
                        dest='gpu',
                        action='store',
                        type=int,
                        default=-1,
                        help='Designate the gpu device id')

    args = parser.parse_args()

    fpath_config = osp.abspath(args.config)        
            
    with open(fpath_config, "rt") as fin:
        config = yaml.safe_load(fin)
    
    if args.gpu >= 0:
        print("[CUDA DEVICE ID]", args.gpu)
        for cfg in config["OBJECTIVES"]:
            if "cuda" not in cfg[2]:
                 continue

            if ":" in cfg[2]:
                _, device_id = cfg[2].split(":")
                device_id = int(device_id) 
            else:
                cfg[2] = "cuda:%d"%(args.gpu)

            print("[OBJECTIVE DEVICE] %s"%(cfg))
        # end of for
    # end of if

    # Create a model.
    dx = float(config["DX"])
    width = int(config["WIDTH"])
    height = int(config["HEIGHT"])
    n_init_pts = int(config["N_INIT_PTS"])

    color_u = None
    color_v = None
    thr_color = None

    if "COLOR_U" in config:
        color_u = ImageColor.getcolor(config["COLOR_U"], "RGB")

    if "COLOR_V" in config:
        color_v = ImageColor.getcolor(config["COLOR_V"], "RGB")

    if "THR_COLOR" in config:
        thr_color = float(config["THR_COLOR"])

    model = ModelFactory.create(
        name=config["MODEL"],
        n_init_pts=n_init_pts,
        width=width,
        height=height,                 
        dx=dx,
        color_u=color_u,
        color_v=color_v,
        thr_color=thr_color
    )

    # Create a solver.
    solver = SolverFactory.create(name=config["SOLVER"],
                                  dt=float(config["DT"]),
                                  n_iters=int(config["N_ITERS"]))

    # Create a converter.
    converter = ConverterFactory.create(config["INITIALIZER"])

    # Create objectives.
    objectives = ObjectiveFactory.create(config["OBJECTIVES"])

    # Load the target laybirds.
    targets = load_targets(config["LADYBIRD_TYPE"], config["LADYBIRD_SUBTYPES"])
    
    
    # Write the config file.
    droot_output = apath(config["DPATH_OUTPUT"])
    

    # Create directories and record the config file.
    str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    dpath_output = osp.join(droot_output, "search_%s"%(str_now))
    os.makedirs(dpath_output, exist_ok=True)
    
    fpath_config = osp.join(dpath_output, "config.yaml")
    with open(fpath_config, 'wt') as fout:
        yaml.dump(config, fout, default_flow_style=False)

    # Create an evolutionary search problem
    search = EvoSearch(model=model,
                       solver=solver,
                       converter=converter,
                       targets=targets,
                       objectives=objectives,
                       droot_output=droot_output)

    prob = pg.problem(search)
    print(prob) 

    # Create the initial population.
    t_beg = time.time()
    pop_size = int(config["POP_SIZE"])
    pop = pg.population(prob, size=pop_size)
    # pop = pg.population(prob)

    print("[POPULATION INITIALIZATION COMPLETED]")

    dpath_init_pop = osp.abspath(config["INIT_POP"])
    if dpath_init_pop:
        # To test the batch processing, add model JSON files.
        model_dicts = load_model_dicts(dpath_init_pop)

        eval_init_fitness = int(config["EVAL_INIT_FITNESS"])
        for i, model_dict in enumerate(model_dicts):
            if i >= pop_size:
                break

            dv = converter.to_dv(model_dict, n_init_pts)
            if eval_init_fitness:
                pop.set_x(i, dv)
            elif "fitness" in model_dict:
                fitness = float(model_dict["fitness"])
                pop.set_xf(i, dv, [fitness])
            else:
                pop.set_xf(i, dv, [np.inf])

        # end of for
    # end of if
    t_end = time.time()
    print("[DURATION OF INITIALIZING POPULATION] %.3f sec."%(t_end - t_beg))
    print(pop)

    # Create an evolutionary algorithm.
    n_procs = int(config["N_PROCS"])
    n_gen = int(config["N_GEN"])

    udi = pg.mp_island()
    udi.resize_pool(n_procs)

    algo = pg.algorithm(pg.sade(gen=1))
    isl = pg.island(algo=algo, pop=pop, udi=udi)

    print(isl)

    # Start searching.
    try:
        for i in range(n_gen):
            t_beg = time.time()
            isl.evolve()
            isl.wait_check()
            t_end = time.time()
            
            print("[EVOLUTION #%d] Best objective: %f (%.3f sec.)"%(i + 1, pop.champion_f[0], t_end - t_beg))

            # Save the best.
            pop = isl.get_population()
            search.save("best", pop.champion_x, max_generation=n_gen, generation=i+1, fitness=pop.champion_f[0])

            # Save the population.
            arr_x = pop.get_x()
            arr_f = pop.get_f()
            for j in range(arr_x.shape[0]):
                x = arr_x[j]
                fitness = arr_f[j, 0]
                search.save("pop", x, max_generation=n_gen, generation=i+1, fitness=fitness)
        # end of for
    except Exception as err:
        print(err)
        udi.shutdown_pool()
        raise err


    print("[EVOLUTIONARY SEARCH COMPLETED]")
    udi.shutdown_pool()
