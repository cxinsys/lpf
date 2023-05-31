import os
import os.path as osp
from os.path import join as pjoin
import time
from datetime import datetime
import shutil
import random
import argparse

import numpy as np
np.seterr(all='raise')

from lpf.initializers import LiawInitializer
from lpf.converters import LiawConverter
from lpf.models import LiawModel
from lpf.models import TwoComponentDiploidModel
from lpf.models import TwoComponentCrosstalkDiploidModel
from lpf.solvers import EulerSolver
from lpf.reproducers import RandomDiploidReproducer


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Configruations for simulation experiment.')

    parser.add_argument('--gpu',
                        dest='gpu',
                        action='store',
                        type=int,
                        default=-1,
                        help='Designate the gpu device id')

    args = parser.parse_args()

    device = "cpu"
    if args.gpu >= 0:
        device = "cuda:%d"%(int(args.gpu))


    dx = 0.1
    dt = 0.01
    width = 128
    height = 128
    thr_color = 0.5
    n_iters = 500000
    shape = (height, width)

    # Create the output directory.
    str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    dpath_output = pjoin("./output", "evopop_%s" % (str_now))
    dpath_output = osp.abspath(dpath_output)
    os.makedirs(dpath_output, exist_ok=True)

    # Copy this source file to the output directory for recording purpose.
    fpath_code_src = pjoin(osp.dirname(__file__), osp.basename(__file__))
    fpath_code_dst = pjoin(dpath_output, osp.basename(__file__))
    shutil.copyfile(fpath_code_src, fpath_code_dst)

    # Create a paternal model.
    model_dict = [
        {"u0": 1.5635296830073362, 
         "v0": 1.6325853074866885, 
         "Du": 0.0004980801662982812, 
         "Dv": 0.07500000000000001, 
         "ru": 0.17999999999999997, 
         "rv": 0.0979337206999831, 
         "k": 0.20000000000000004, 
         "su": 0.0008394576268270522, 
         "sv": 0.025000000000000005, 
         "mu": 0.07999999999999999, 
         "init_pts_0": ["33", "60"], 
         "init_pts_1": ["46", "3"], 
         "init_pts_2": ["53", "50"], 
         "init_pts_3": ["39", "70"], 
         "init_pts_4": ["40", "90"], 
         "init_pts_5": ["64", "4"], 
         "init_pts_6": ["60", "117"], 
         "init_pts_7": ["66", "7"], 
         "init_pts_8": ["50", "70"],
         "init_pts_9": ["50", "90"],
         "init_pts_10": ["58", "48"],
         "init_pts_11": ["60", "30"],
         "init_pts_12": ["63", "47"],
         "init_pts_13": ["87", "30"],
         "init_pts_14": ["77", "4"],
         "init_pts_15": ["87", "29"],
         "init_pts_16": ["72", "111"],
         "init_pts_17": ["57", "61"],
         "init_pts_18": ["110", "42"],
         "init_pts_19": ["78", "59"]}
    ]
    
    # Create the initializer.
    initializer = LiawInitializer()
    initializer.update(model_dict)

    params = LiawModel.parse_params(model_dict)
    pa_model = LiawModel(
        initializer=initializer,
        params=params,
        width=width,
        height=height,
        dx=dx,
        device=device
    )

    # Create a maternal model.
    model_dict = [
        {"u0": 4.125239430041862, 
         "v0": 18.18479114252238, 
         "Du": 0.0011041379940844168,
         "Dv": 0.14596920545639724,
         "ru": 0.08897465243231621,
         "rv": 0.11404666216196024,
         "k": 0.5720189763213703,
         "su": 0.0007382506069868803,
         "sv": 0.021257567063086704,
         "mu": 0.028948565299659442,
         "init_pts_0": ["89", "93"],
         "init_pts_1": ["25", "65"],
         "init_pts_2": ["77", "52"],
         "init_pts_3": ["62", "96"],
         "init_pts_4": ["12", "88"],
         "init_pts_5": ["27", "67"],
         "init_pts_6": ["26", "28"],
         "init_pts_7": ["44", "81"],
         "init_pts_8": ["86", "44"],
         "init_pts_9": ["80", "69"],
         "init_pts_10": ["42", "72"],
         "init_pts_11": ["90", "50"],
         "init_pts_12": ["61", "30"],
         "init_pts_13": ["63", "71"],
         "init_pts_14": ["16", "23"],
         "init_pts_15": ["91", "88"],
         "init_pts_16": ["35", "33"],
         "init_pts_17": ["81", "2"],
         "init_pts_18": ["70", "89"],
         "init_pts_19": ["83", "75"]}
    ]    
    
    # Create the initializer.
    initializer = LiawInitializer()
    initializer.update(model_dict)
    
    params = LiawModel.parse_params(model_dict)
    ma_model = LiawModel(
        initializer=initializer,
        params=params,
        width=width,
        height=height,
        dx=dx,
        device=device
    )
    
    
    # Create a diploid model.
    model = TwoComponentDiploidModel(
        paternal_model=pa_model,
        maternal_model=ma_model,
        alpha=0.5,
        beta=0.5,
        device=device
    )
        
    
    # Start population evolution.      
    n_generations = 1000
    pop_size = 32  # The size of population (the number of organisms)
    n_cross = 4  # The number of crossing experiments
    n_gametes = 32  # The number of gametes (daughter celss)
    prob_crossover = 0.3
    n_progenies_per_cross = pop_size // n_cross
    
    fstr_gen = "generation-%0{}d".format(int(np.floor(np.log10(n_generations))) + 1)
    fstr_duration = "[Generation #%d] Elapsed time: %f sec."
        
    pops = [[]]  # population per generation.
    
    str_gen = fstr_gen%(0)
    dpath_gen = pjoin(dpath_output, str_gen)
    os.makedirs(dpath_gen, exist_ok=True)
    
    for i in range(pop_size):
        str_id = "%s_model-%d"%(str_gen, i + 1)
        pops[0].append({
            "ID": str_id,
            "MODEL": model,
            "PATERNAL": None,
            "MATERNAL": None,
            "MORPH": None,
            "MODEL_DICT": model.to_dict()            
        })
        fpath_model = pjoin(dpath_gen, "model_%s.json"%(str_id))                
        model.save_model(index=0, fpath=fpath_model)
    # end of for
        
    reproducer = RandomDiploidReproducer()
    
    for i in range(1, n_generations):
        t_beg = time.time()
        
        # Add a list that contains the organisms of this generation.
        pops.append([])
        
        # Create a directory that contains the files of this generation.
        str_gen = fstr_gen%(i)
        dpath_gen = pjoin(dpath_output, str_gen)
        os.makedirs(dpath_gen, exist_ok=True)
            
        cnt_progenies = 0
        # Repeat crossing experiments.
        for j in range(n_cross):
            male_model_dict = random.choice(pops[i - 1])
            female_model_dict = random.choice(pops[i - 1])
            
            id_male = male_model_dict["ID"]
            male_model = male_model_dict["MODEL"] 
            
            id_female = female_model_dict["ID"]
            female_model = female_model_dict["MODEL"] 
            
            pa_model, ma_model = reproducer.cross(male_model=male_model,
                                                  female_model=female_model,
                                                  n_progenies=n_progenies_per_cross,
                                                  n_gametes=n_gametes,
                                                  prob_crossover=prob_crossover,
                                                  device=device)
            
            # Create the model.
            model = TwoComponentDiploidModel(
                paternal_model=pa_model,
                maternal_model=ma_model,
                alpha=0.5,
                beta=0.5,
                device=device
            )
                
            
            # Perform a numerical simulation.
            solver = EulerSolver()        
        
            solver.solve(
                model=model,
                dt=dt,
                n_iters=n_iters,
                period_output=n_iters,
                verbose=0
            )
                    
            arr_color = model.colorize(thr_color=thr_color)

    
            # Update and save the progenies.                
            for k in range(n_progenies_per_cross):
                img_ladybird, img_pattern = model.create_image(k, arr_color)
                            
                # Paternal model
                initializer = LiawInitializer(
                    init_states=pa_model.initializer.init_states[None, k, :],
                    init_pts=pa_model.initializer.init_pts[None, k, :, :]
                )
                
                progeny_pa_model = LiawModel(
                    initializer=initializer,
                    params=pa_model.params[None, k, :],
                    width=width,
                    height=height,
                    dx=dx,
                    device=device
                )
                
                # Maternal model
                initializer = LiawInitializer(
                    init_states=ma_model.initializer.init_states[None, k, :],
                    init_pts=ma_model.initializer.init_pts[None, k, :, :]
                )
                
                progeny_ma_model = LiawModel(
                    initializer=initializer,
                    params=ma_model.params[None, k, :],
                    width=width,
                    height=height,
                    dx=dx,
                    device=device
                )
                
                progeny_model = TwoComponentDiploidModel(
                    paternal_model=progeny_pa_model,
                    maternal_model=progeny_ma_model,
                    alpha=0.5,
                    beta=0.5,
                    device=device
                )
                
                cnt_progenies += 1
                str_id = "%s_model-%d"%(str_gen, cnt_progenies)
                pops[i].append({
                    "ID": str_id,
                    "MODEL": progeny_model,
                    "PATERNAL": id_male,
                    "MATERNAL": id_female,
                    "MORPH": img_ladybird,
                    "MODEL_DICT": model.to_dict()            
                })
                
                fpath_model = pjoin(dpath_gen, "model_%s.json"%(str_id))
                fpath_ladybird = pjoin(dpath_gen, "ladybird_%s.png"%(str_id))
                fpath_pattern = pjoin(dpath_gen, "pattern_%s.png"%(str_id))
                
                progeny_model.save_model(index=0, fpath=fpath_model)
                img_ladybird.save(fpath_ladybird)
                img_pattern.save(fpath_pattern)            
            # end of for k in range(pop_size)
        # end of for j in range(n_cross)
        
        t_end = time.time()    
        print(fstr_duration % (i, t_end - t_beg))        
    # end of for i in range(1, n_generations)

        
