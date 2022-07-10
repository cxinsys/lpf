import time
import json
import os.path as osp
from os.path import join as pjoin
from os.path import abspath as apath
from datetime import datetime
import argparse
from itertools import product

import yaml
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt

from lpf.models import LiawModel
from lpf.initializers import LiawInitializer
from lpf.initializers import InitializerFactory as InitFac
from lpf.objectives import ObjectiveFactory as ObjFac
from lpf.search import RdPde2dSearch
from lpf.utils import get_module_dpath
from lpf.data import load_targets

plt.ioff()
np.seterr(all='raise')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse configruations for searching.')
    parser.add_argument('--config',
                        dest='config',
                        action='store',
                        type=str,
                        help='Designate the file path of configuration file in YAML')

    args = parser.parse_args()
    fpath_config = osp.abspath(args.config)        
            
    with open(fpath_config, "rt") as fin:
        config = yaml.safe_load(fin)

    # Create a search object.
    num_init_pts = config["NUM_INIT_PTS"]

    class LiawModelConverter:
                                            
        def to_params(self, x, params=None):
            """
            Args:
                x: Decision vector of PyGMO
            """
            if params is None:
                params = np.zeros((10,), dtype=np.float64)
            Du = 10 ** x[0]
            Dv = 10 ** x[1]
            ru = 10 ** x[2]
            rv = 10 ** x[3]
            k  = 10 ** x[4]
            su = 10 ** x[5]
            sv = 10 ** x[6]
            mu = 10 ** x[7]
                
            params[0] = Du
            params[1] = Dv
            params[2] = ru
            params[3] = rv
            params[4] = k
            params[5] = su
            params[6] = sv
            params[7] = mu
            
            return params
            
            
        def to_init_states(self, x, init_states=None):    
            if init_states is None:
                init_states = np.zeros((2,), dtype=np.float64)
                
            init_states[0] =  10 ** x[8]  # u0
            init_states[1] = 10 ** x[9]  # v0
            return init_states
        
        def to_init_pts(self, x):            
            ir = np.zeros(num_init_pts, dtype=np.int32)
            ic = np.zeros(num_init_pts, dtype=np.int32)
            
            for i, coord in enumerate(zip(x[10::2], x[11::2])): 
                ir[i] = int(coord[0])
                ic[i] = int(coord[1])
                
            return ir, ic
        
        def to_initializer(self, x):            
            ir_init, ic_init = self.to_init_pts(x)            
            init = LiawInitializer(ir_init=ir_init, ic_init=ic_init)
            return init
                   
            
        
    converter = LiawModelConverter()
        
    # Create the model.
    dx = float(config["DX"])
    dt = float(config["DT"])
    width = int(config["WIDTH"])
    height = int(config["HEIGHT"])
    thr = float(config["THR"])
    n_iters = int(config["N_ITERS"])
    shape = (height, width)
   
    # Create the objectives.
    objectives = []
    for cfg in config["OBJECTIVES"]:
        obj = cfg[0]
        coeff = float(cfg[1])
        device = cfg[2]        
        
        objectives.append(ObjFac.create(obj, coeff=coeff, device=device))
    # end of for

    # Load ladybird type and the corresponding data.
    ladybird_type = config["LADYBIRD_TYPE"].lower()
    dpath_data = pjoin(get_module_dpath("data"), ladybird_type)
    dpath_template = pjoin(dpath_data, "template")
    dpath_target = pjoin(dpath_data, "target")

    fpath_template = pjoin(dpath_template, "ladybird.png")    
    fpath_mask = pjoin(dpath_template, "mask.png")
    
    model = LiawModel(
        width=width,
        height=height,                 
        dx=dx,
        dt=dt,
        n_iters=n_iters,
        num_init_pts=num_init_pts,
        fpath_template=fpath_template,
        fpath_mask=fpath_mask
    )
    
    # Load targets.
    ladybird_subtypes = config["LADYBIRD_SUBTYPES"]
    ladybird_subtypes = [elem.lower() for elem in ladybird_subtypes]
    targets = load_targets(dpath_target,
                           ladybird_subtypes)

    droot_output = apath(config["DPATH_OUTPUT"])
    search = RdPde2dSearch(config,
                           model,
                           converter,
                           targets,
                           objectives,
                           droot_output)
    prob = pg.problem(search)
    print(prob) 
   
    
    # Create an initial population.
    pop_size = int(config["POP_SIZE"])
    pop = pg.population(prob, size=pop_size)
    
    fpath_init_pop = config["INIT_POP"]

    if fpath_init_pop:
        fpath_init_pop = osp.abspath(fpath_init_pop)
        
        if osp.isfile(fpath_init_pop):
            with open(fpath_init_pop, "rt") as fin:
                print("[INITIAL POPULATION]", fpath_init_pop)
                
                init_pop = json.load(fin)
                
            x = np.zeros((10 + 2*num_init_pts,), dtype=np.float64)
            # for i, n2v in enumerate(init_pop):

            num_init_pop = len(init_pop)

            for i in range(pop_size):
                n2v = init_pop[i % num_init_pop]

                x[0] = np.log10(n2v["Du"])
                x[1] = np.log10(n2v["Dv"])
                x[2] = np.log10(n2v["ru"])
                x[3] = np.log10(n2v["rv"])
                x[4] = np.log10(n2v["k"])
                x[5] = np.log10(n2v["su"])
                x[6] = np.log10(n2v["sv"])
                x[7] = np.log10(n2v["mu"])
                x[8] = np.log10(n2v["u0"])
                x[9] = np.log10(n2v["v0"])
               
               
                j = 0
                for name, val in n2v.items():
                    if "init-pts" in name:
                        x[10 + 2*j] = int(val[0])
                        x[11 + 2*j] = int(val[1])
                        j += 1
                        # print(name, (10 + 2*j, 11 + 2*j), val)
                # end of for

                if j == 0:  # if the number of init pts equals to 0.
                    # rc_product: Production of rows and columns
                    rc_product = product(np.arange(40, 90, 10),
                                         np.arange(10, 110, 20))

                    for j, (ir, ic) in enumerate(rc_product):
                        x[10 + 2*j] = ir
                        x[11 + 2*j] = ic
                    # end of for
                
                pop.set_x(i, x)
            # end of for


    # Create an algorithm.
    
#    isl = pg.island(algo=pg.sade(gen=1),
#                    pop=pop,                    
#                    udi=pg.mp_island())
    
   
    n_isl = int(config["N_ISL"])
    archi = pg.archipelago(n=n_isl,
                           algo=pg.sade(gen=1),
                           pop=pop,                    
                           udi=pg.mp_island())

    # algo = pg.algorithm(pg.sade(gen=1))    
    #algo = pg.algorithm(pg.sga(gen=1))    
    #algo = pg.algorithm(pg.sea(gen=1))    
    
    
    num_gen = int(config["N_GEN"])
    for i in range(num_gen):
        print(archi)

        t_beg = time.time()
        
        # pop = algo.evolve(pop)
        # isl.evolve()
        archi.evolve()
        # pop = isl.get_population()
        #pop = archi.get_population()
       
        archi.wait_check()

        t_end = time.time()        
        dur = t_end - t_beg
        # print("[Evolution #%d] Best objective: %f (%.3f sec.)"%(i + 1, pop.champion_f[0], dur))        
        
        print("[Evolution #%d] (%.3f sec.)"%(i+1, dur))
        
        str_now = datetime.now().strftime('%Y%m%d-%H%M%S')

        list_champ_f = archi.get_champions_f()
        list_champ_x = archi.get_champions_x()

        for j, (champ_f, champ_x) in enumerate(zip(list_champ_f, list_champ_x)):
            print("- Champion: %f"%(champ_f[0]))        

            fpath_model = pjoin(search.dpath_best, "model_%s_%d.json"%(str_now, j+1))
            fpath_image = pjoin(search.dpath_best, "image_%s_%d.png"%(str_now, j+1))        
            search.save(fpath_model, fpath_image, champ_f[0], champ_x)
    # end of for
