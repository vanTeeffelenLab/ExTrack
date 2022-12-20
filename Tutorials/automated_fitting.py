import extrack
import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob

dt = 0.3

datafolder = '/mnt/c/Users/username/path/dataset'
SAVEDIR = '/mnt/c/Users/username/path/Res'
workers = 5

if not os.path.exists(SAVEDIR):
    os.mkdir(SAVEDIR)

exps = glob(datafolder + '/*')
print(exps)

# performs a fitting for each replicate represented by the folder name (Exp1 or Exp2 for instance).
for exp in exps:
    paths = glob(exp + '/*.xml') # collect all paths from the replicate
    
    if len(paths) ==0:
        raise ValueError('Wrong path, no xml file were found')
    
    lengths = np.arange(2,20)
    
    all_tracks, frames, opt_metrics = extrack.readers.read_trackmate_xml(paths,
                                           lengths=lengths,
                                           dist_th = 0.4,
                                           frames_boundaries = [0, 10000],
                                           remove_no_disp = True,
                                           opt_metrics_names = [], # Name of the optional metrics to catch
                                           opt_metrics_types = None)
    
    for l in list(all_tracks.keys()):
        if len(all_tracks[l])==0:
            del all_tracks[l]
    
    for l in all_tracks:
        print(all_tracks[l].shape)
    
    params = extrack.tracking.generate_params(nb_states = 2,
                            LocErr_type = 1,
                            nb_dims = 2, # only matters if LocErr_type == 2.
                            LocErr_bounds = [0.01, 0.05], # the initial guess on LocErr will be the geometric mean of the boundaries.
                            D_max = 1, # maximal diffusion coefficient allowed.
                            Fractions_bounds = [0.001, 0.99],
                            estimated_LocErr = [0.022],
                            estimated_Ds = [0.0001, 0.03], # D will be arbitrary spaced from 0 to D_max if None, otherwise input a list of Ds for each state from state 0 to nb_states - 1.
                            estimated_Fs = [0.3,0.7], # fractions will be equal if None, otherwise input a list of fractions for each state from state 0 to nb_states - 1.
                            estimated_transition_rates = 0.1, # transition rate per step. example [0.1,0.05,0.03,0.07,0.2,0.2] for a 3-state model.
                            )
        
    res = {}
    for param in params:
        res[param] = []
    res['residual'] = []
    
    for k in range(3):
        # We run multiple iterations to make sure about the convergence.
        model_fit = extrack.tracking.param_fitting(all_tracks = all_tracks,
                                                    dt = dt,
                                                    params = params,
                                                    nb_states = 2,
                                                    nb_substeps = 1,
                                                    cell_dims = [0.3],
                                                    frame_len = 9,
                                                    verbose = 0,
                                                    workers = workers, # increase the number of CPU workers for faster computing, do not work on windows or mac (keep to 1)
                                                    input_LocErr = None,
                                                    steady_state = False,
                                                    threshold = 0.1,
                                                    max_nb_states = 200,
                                                    method = 'bfgs')
    
        params = model_fit.params
        print(model_fit.residual[0])
    
        for param in params:
            res[param].append(params[param].value)
        res['residual'].append(model_fit.residual[0])
    
        np.save(SAVEDIR + exp.split('/')[-1] + '.npy', res, allow_pickle=True)
    
        print(res)

