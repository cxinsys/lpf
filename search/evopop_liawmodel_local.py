import os
import os.path as osp
from os.path import join as pjoin
from datetime import datetime
import shutil
import argparse

import numpy as np
np.seterr(all='raise')


from lpf.data import load_model_dicts
from lpf.initializers import LiawInitializer
from lpf.models import LiawModel
from lpf.models import TwoComponentDiploidModel
from lpf.solvers import EulerSolver
from lpf.reproducers import RandomTwoComponentDiploidReproducer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Configurations for simulation experiment.'
    )

    parser.add_argument('--gpu',
                        dest='gpu',
                        action='store',
                        type=int,
                        default=-1,
                        help='Designate the gpu device id')

    args = parser.parse_args()

    device = "torch"
    # device = "cpu"
    # if args.gpu >= 0:
    #     device = "cuda:%d"%(int(args.gpu))

    # Define spatiotemporal parameters.
    dx = 0.1
    dt = 0.01
    width = 128
    height = 128
    thr_color = 0.5
    n_iters = 500000
    
    # Define hyper-parameters for population evolution.      
    n_generations = 1000
    pop_size = 32  # The size of population (the number of organisms)
    n_cross = 8  # The number of crossing experiments
    n_gametes = 32  # The number of gametes (the number of daughter cells)
    prob_crossover = 0.5
    alpha = 0.5
    beta = 0.5

    # Create the output directory.
    str_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    dpath_output = pjoin("./output", "evopop_%s" % (str_now))
    dpath_output = osp.abspath(dpath_output)
    os.makedirs(dpath_output, exist_ok=True)

    # Copy this source file to the output directory for recording purpose.
    fpath_code_src = pjoin(osp.dirname(__file__), osp.basename(__file__))
    fpath_code_dst = pjoin(dpath_output, osp.basename(__file__))
    shutil.copyfile(fpath_code_src, fpath_code_dst)

    # Load the dicts of paternal models.
    dpath_init_pop = osp.abspath(r"../population/init_pop_axyridis")
    pa_model_dicts = load_model_dicts(dpath_init_pop)
        

    # Load the dicts of maternal models.
    dpath_init_pop = osp.abspath(r"../population/init_pop_succinea")
    ma_model_dicts = load_model_dicts(dpath_init_pop)
    
    assert(len(pa_model_dicts) == len(ma_model_dicts))
    
    # Shuffle the model dicts.
    np.random.shuffle(pa_model_dicts)
    np.random.shuffle(ma_model_dicts)
    
    # Create a population.
    init_pop = []
    
    for i in range(pop_size):        
    
        i = i % len(pa_model_dicts)
        
        # Create paternal model.  
        model_dict = pa_model_dicts[i:i+1]
        params = LiawModel.parse_params(model_dict)
        
        initializer = LiawInitializer()
        initializer.update(model_dict)
        
        pa_model = LiawModel(
            initializer=initializer,
            params=params,
            width=width,
            height=height,
            dx=dx,
            device=device
        )
    
        # Create maternal model.                
        model_dict = ma_model_dicts[i:i+1]
        params = LiawModel.parse_params(model_dict)
        
        initializer = LiawInitializer()
        initializer.update(model_dict)
        
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
            alpha=alpha,
            beta=beta,
            device=device
        )
        init_pop.append(model)
    # end of for
    
    solver = EulerSolver(
        dt=dt,
        n_iters=n_iters,
        period_output=1000,
        verbose=1
    )
    
    reproducer = RandomTwoComponentDiploidReproducer(
        population=init_pop,
        solver=solver,
        pop_size=pop_size,
        n_cross=n_cross,
        n_gametes=n_gametes,
        prob_crossover=prob_crossover,
        alpha=alpha,
        beta=beta,
        diploid_model_class=TwoComponentDiploidModel,
        haploid_model_class=LiawModel,
        haploid_initializer_class=LiawInitializer,
        dpath_output=dpath_output,
        device=device,
        verbose=1        
    )
    
    reproducer.evolve(n_generations=n_generations)
    

        
