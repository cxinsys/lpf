import time
import os.path as osp
from os.path import abspath as apath
import argparse

import yaml
import numpy as np
import pygmo as pg

from lpf.models import LiawModel
from lpf.objectives import ObjectiveFactory as ObjFac
from lpf.search import EvoSearch
from lpf.converters import LiawConverter
from lpf.data import load_model_dicts
from lpf.data import load_targets


np.seterr(all='raise')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse configruations for searching.')
    parser.add_argument('--config',
                        dest='config',
                        action='store',
                        type=str,
                        help='Designate the file path of configuration file in YAML')

    parser.add_argument('--gpu',
                        dest='gpu',
                        action='store',
                        type=int,
                        default=0,
                        help='Designate the gpu device id')

    args = parser.parse_args()
    fpath_config = osp.abspath(args.config)        
            
    with open(fpath_config, "rt") as fin:
        config = yaml.safe_load(fin)

    # Create a search object.
    num_init_pts = config["NUM_INIT_PTS"]

    # Create the model.
    dx = float(config["DX"])
    dt = float(config["DT"])
    width = int(config["WIDTH"])
    height = int(config["HEIGHT"])
    thr = float(config["THR"])
    n_iters = int(config["N_ITERS"])
    rtol_early_stop = float(config["RTOL_EARLY_STOP"])

    shape = (height, width)
   
    # Create the objectives.
    objectives = ObjFac.create(config["OBJECTIVES"])

    # objectives = []
    # for cfg in config["OBJECTIVES"]:
    #     obj = cfg[0]
    #     coeff = float(cfg[1])
    #
    #     device = "cpu"
    #     if "cuda" in cfg[2]:
    #         device = "cuda:%d"%(args.gpu)
    #
    #     print("[OBJECTIVE DEVICE] %s: %s"%(obj, device))
    #
    #     objectives.append(ObjFac.create(obj, coeff=coeff, device=device))
    # # end of for

    # Load ladybird type and the corresponding data.
    # ladybird_type = config["LADYBIRD_TYPE"].lower()
    # dpath_data = pjoin(get_module_dpath("data"), ladybird_type)
    # dpath_template = pjoin(dpath_data, "template")
    # dpath_target = pjoin(dpath_data, "target")
    #
    # fpath_template = pjoin(dpath_template, "ladybird.png")
    # fpath_mask = pjoin(dpath_template, "mask.png")

    #device = config["DEVICE"]

    model = LiawModel(
        width=width,
        height=height,                 
        dx=dx,
        dt=dt,
        n_iters=n_iters,
        num_init_pts=num_init_pts,
        rtol_early_stop=rtol_early_stop,
        # device=device
    )
    
    # Load targets.
    targets = load_targets(config["LADYBIRD_TYPE"], config["LADYBIRD_SUBTYPES"])
    # ladybird_subtypes = [elem.lower() for elem in ladybird_subtypes]
    # targets = load_targets(dpath_target,
    #                        ladybird_subtypes)

    dpath_output = apath(config["DPATH_OUTPUT"])

    converter = LiawConverter()

    search = EvoSearch(config,
                       model,
                       converter,
                       targets,
                       objectives,
                       dpath_output)

    prob = pg.problem(search)
    print(prob) 
   
    
    # Create the initial population.
    t_beg = time.time()
    pop_size = int(config["POP_SIZE"])
    pop = pg.population(prob, size=pop_size)
    
    dpath_init_pop = osp.abspath(config["INIT_POP"])
    #dpath_init_pop = None  # FOR DEBUGGING
    if dpath_init_pop:

        # To test the batch processing, add model JSON files.
        model_dicts = load_model_dicts("../population/init_pop_axyridis/")
        # dvs = converter.to_dvs(model_dicts)

        eval_init_fitness = int(config["EVAL_INIT_FITNESS"])
        for i, model_dict in enumerate(model_dicts):
            if i >= pop_size:
                break

            dv = converter.to_dv(model_dict)
            if eval_init_fitness:
                pop.set_x(i, dv)
            elif "fitness" in model_dict:
                fitness = float(model_dict["fitness"])
                pop.set_xf(i, dv, [fitness])
            else:
                raise ValueError("'fitness' should be defined in the JSON file if EVAL_INIT_FITNESS is False.")

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

    try:
        for i in range(n_gen):
            print(isl)
            t_beg = time.time()
            isl.evolve()
            isl.wait_check()
            t_end = time.time()
            
            print("[EVOLUTION #%d] Best objective: %f (%.3f sec.)"%(i + 1, pop.champion_f[0], t_end - t_beg))

            # # Save the best.
            # pop = isl.get_population()
            # search.save("best", pop.champion_x, generation=i+1, fitness=pop.champion_f[0])
            #
            # # Save the population.
            # arr_x = pop.get_x()
            # arr_f = pop.get_f()
            # for j in range(arr_x.shape[0]):
            #     x = arr_x[j]
            #     fitness = arr_f[j, 0]
            #     search.save("pop", x, generation=i+1, fitness=fitness)
        # end of for
    except Exception as err:
        #print(err)
        udi.shutdown_pool()
        raise err


    print("[EVOLUTIONARY SEARH COMPLETED]")
    udi.shutdown_pool()
