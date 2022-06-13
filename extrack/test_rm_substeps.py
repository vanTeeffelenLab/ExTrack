
from tracking import get_2DSPT_params
from simulate_tracks import sim_FOV
import numpy as np
from matplotlib import pyplot as plt

tracks, True_Bs = sim_FOV(nb_tracks=10000,
                    max_track_len=30, 
                    LocErr=0.02,
                    Ds = np.array([0,0.2]),
                    initial_fractions = np.array([0.6,0.4]),
                    TrMat = np.array([[0.7,0.3],[0.3,0.7]]),
                    dt = 0.02,
                    pBL = 0.1, 
                    cell_dims = [1000,None,None], # dimension limits in x, y and z respectively
                    min_len = 3)

dt = 0.02
LocErr = 0.02
Fs =  np.array([0.6,0.4])
TrMat = np.array([[0.9,0.1],[0.1,0.9]])
ds = (2* np.array([0,0.2])*dt)**0.5
pBL = 0.1
cell_dims = [0.5]
min_len = 5

Cs = tracks['5'][:1]

model_fit = get_2DSPT_params(tracks,
                            dt,
                            nb_substeps = 2,
                            nb_states = 2,
                            frame_len = 6,
                            verbose = 1,
                            workers = 1,
                            method = 'powell',
                            steady_state = False,
                            cell_dims = [1000], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                            vary_params = {'LocErr' : True, 'D0' : True, 'D1' : True, 'F0' : True, 'p01' : True, 'p10' : True, 'pBL' : True},
                            estimated_vals =  {'LocErr' : 0.02, 'D0' : 1e-20, 'D1' : 0.4, 'F0' : 0.45, 'p01' : 0.4, 'p10' : 0.4, 'pBL' : 0.1},
                            min_values = {'LocErr' : 0.007, 'D0' : 1e-12, 'D1' : 0.00001, 'F0' : 0.001, 'p01' : 0.001, 'p10' : 0.001, 'pBL' : 0.001},
                            max_values = {'LocErr' : 0.6, 'D0' : 1, 'D1' : 10, 'F0' : 0.999, 'p01' : 1., 'p10' : 1., 'pBL' : 0.99})

model_fit = get_2DSPT_params(tracks,
                            dt,
                            nb_substeps = 3,
                            nb_states = 2,
                            frame_len = 4,
                            verbose = 1,
                            workers = 1,
                            method = 'powell',
                            steady_state = False,
                            cell_dims = [1000], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                            vary_params = {'LocErr' : False, 'D0' : False, 'D1' : True, 'F0' : False, 'p01' : False, 'p10' : False, 'pBL' : False},
                            estimated_vals =  {'LocErr' : 0.02, 'D0' : 1e-20, 'D1' : 0.4, 'F0' : 0.45, 'p01' : 0.3, 'p10' : 0.3, 'pBL' : 0.1},
                            min_values = {'LocErr' : 0.007, 'D0' : 1e-12, 'D1' : 0.00001, 'F0' : 0.001, 'p01' : 0.001, 'p10' : 0.001, 'pBL' : 0.001},
                            max_values = {'LocErr' : 0.6, 'D0' : 1, 'D1' : 10, 'F0' : 0.999, 'p01' : 1., 'p10' : 1., 'pBL' : 0.99})

all_tracks = tracks
nb_substeps = 3
frame_len = 7
verbose = 1
workers = 1
method = 'powell'
steady_state = False
cell_dims = [1000] # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
vary_params = {'LocErr' : True, 'D0' : True, 'D1' : True, 'F0' : True, 'p01' : True, 'p10' : True, 'pBL' : True}
estimated_vals =  {'LocErr' : 0.025, 'D0' : 1e-20, 'D1' : 0.2, 'F0' : 0.45, 'p01' : 0.2, 'p10' : 0.2, 'pBL' : 0.1}
min_values = {'LocErr' : 0.007, 'D0' : 1e-12, 'D1' : 0.00001, 'F0' : 0.001, 'p01' : 0.001, 'p10' : 0.001, 'pBL' : 0.001}
max_values = {'LocErr' : 0.6, 'D0' : 1, 'D1' : 10, 'F0' : 0.999, 'p01' : 1., 'p10' : 1., 'pBL' : 0.99}

model_fit = get_2DSPT_params(tracks,
                            dt,
                            nb_substeps = 1,
                            nb_states = 2,
                            frame_len = 7,
                            verbose = 1,
                            workers = 1,
                            method = 'powell',
                            steady_state = False,
                            cell_dims = [1000], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                            vary_params = {'LocErr' : False, 'D0' : False, 'D1' : False, 'F0' : False, 'p01' : False, 'p10' : False, 'pBL' : False},
                            estimated_vals =  {'LocErr' : 0.02, 'D0' : 1e-20, 'D1' : 0.05, 'F0' : 0.4, 'p01' : 0.1, 'p10' : 0.1, 'pBL' : 0.1},
                            min_values = {'LocErr' : 0.007, 'D0' : 1e-12, 'D1' : 0.00001, 'F0' : 0.001, 'p01' : 0.001, 'p10' : 0.001, 'pBL' : 0.001},
                            max_values = {'LocErr' : 0.6, 'D0' : 1, 'D1' : 10, 'F0' : 0.999, 'p01' : 1., 'p10' : 1., 'pBL' : 0.99})


pred_Bs = predict_Bs(tracks,
                   dt,
                   model_fit.params,
                   cell_dims=[1],
                   nb_states=2,
                   frame_len=5)



for track in tracks['20']:
    plt.plot(track[:,0], track[:,1])


