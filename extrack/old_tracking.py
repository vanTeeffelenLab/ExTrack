# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 18:31:26 2022

@author: Franc
"""

import numpy as np

GPU_computing = False

if GPU_computing :
    import cupy as cp
    from cupy import asnumpy
else :
    import numpy as cp
    def asnumpy(x):
        return np.array(x)

import itertools
import scipy
from lmfit import minimize, Parameters

import multiprocessing
from itertools import product

'''
Maximum likelihood to determine transition rates :
We compute the probability of observing the tracks knowing the parameters :
For this, we assum a given set of consecutive states ruling the displacement terms,
we express the probability of having a given set of tracks and real positions.
Next, we integrate reccursively over the possible real positions to find the probability
of the track knowing the consecutive states. This recurrance can be performed as the
integral of the product of normal laws is a constant time a normal law.
In the end of the reccurance process we have a constant time normal law of which 
all the terms are known and then a value of the probability.
We finally use the conditional probability principle over the possible set of states
to compute the probability of having the tracks. 
'''

def ds_froms_states(ds, cur_states):
    cur_ds = ds[cur_states]
    cur_ds = (cur_ds[:,:,1:]**2 + cur_ds[:,:,:-1]**2)**0.5 / 2**0.5 # assuming a transition at the middle of the substeps
    # we can average the variances of displacements per step to get the actual std of displacements per step
    cur_ds = cp.mean(cur_ds**2, axis = 2)**0.5
    cur_ds = cur_ds[:,:,None]
    cur_ds = cp.array(cur_ds)
    return cur_ds
    
def log_integrale_dif(Ci, l, cur_ds, Km, Ks):
    '''
    integral of the 3 exponetional terms (localization error, diffusion, previous term)
    the integral over r1 of f_l(r1-c1)f_d(r1-r0)f_Ks(r1-Km) equals :
    np.exp(-((l**2+Ks**2)*r0**2+(-2*Km*l**2-2*Ks**2*c1)*r0+Km**2*l**2+(Km**2-2*c1*Km+c1**2)*d**2+Ks**2*c1**2)/((2*d**2+2*Ks**2)*l**2+2*Ks**2*d**2))/(2*np.pi*Ks*d*l*np.sqrt((d**2+Ks**2)*l**2+Ks**2*d**2))
    which can be turned into the form Constant*fKs(r0 - newKm) where fKs is a normal law of std newKs
    the idea is to create a function of integral of integral of integral etc
    dim 0 : tracks
    dim 1 : possible sequences of states
    dim 2 : x,y (z)
    '''
    newKm = (Km*l**2 + Ci*Ks**2)/(l**2+Ks**2)
    newKs = ((cur_ds**2*l**2 + cur_ds**2*Ks**2 + l**2*Ks**2)/(l**2 + Ks**2))**0.5
    logConstant = Km.shape[2]*cp.log(1/((l**2 + Ks[:,:,0]**2)**0.5*np.sqrt(2*np.pi))) + cp.sum((newKm**2/(2*newKs**2) - (Km**2*l**2 + Ks**2*Ci**2 + (Km-Ci)**2*cur_ds**2)/(2*newKs**2*(l**2 + Ks**2))), axis = 2)
    return newKm, newKs, logConstant

def first_log_integrale_dif(Ci, LocErr, cur_ds):
    '''
    convolution of 2 normal laws = normal law (mean = sum of means and variance = sum of variances)
    '''
    Ks = (LocErr**2+cur_ds**2)**0.5
    Km = Ci
    return Km, Ks

def P_Cs_inter_bound_stats(Cs, LocErr, ds, Fs, TrMat, pBL=0.1, isBL = 1, cell_dims = [0.5], nb_substeps=1, frame_len = 6, do_preds = 0, min_len = 3) :
    '''
    compute the product of the integrals over Ri as previousily described
    work in log space to avoid overflow and underflow
    
    Cs : dim 0 = track ID, dim 1 : states, dim 2 : peaks postions through time,
    dim 3 : x, y position
    
    we process by steps, at each step we account for 1 more localization, we compute
    the canstant (LC), the mean (Km) and std (Ks) of of the normal distribution 
    resulting from integration.
    
    each step is made of substeps if nb_substeps > 1, and we increase the matrix
    of possible Bs : cur_Bs accordingly
    
    to be able to process long tracks with good accuracy, for each track we fuse Km and Ks
    of sequences of states equal exept for the state 'frame_len' steps ago.
    '''
    nb_Tracks = Cs.shape[0]
    nb_locs = Cs.shape[1] # number of localization per track
    nb_dims = Cs.shape[2] # number of spatial dimentions (x, y) or (x, y, z)
    Cs = Cs.reshape((nb_Tracks,1,nb_locs, nb_dims))
    Cs = cp.array(Cs)
    nb_states = TrMat.shape[0]
    Cs = Cs[:,:,::-1]
    
    if do_preds:
        preds = np.zeros((nb_Tracks, nb_locs, nb_states))-1
    else :
        preds = []
    
    if nb_locs < 2:
        raise ValueError('minimal track length = 2, here track length = %s'%nb_locs)
    
    all_Bs = get_all_Bs(np.min([(nb_locs-1)*nb_substeps+1, frame_len + nb_substeps - 1]), nb_states)[None]
    
    TrMat = cp.array(TrMat.T)
    current_step = 1
    
    #cur_Bs = get_all_Bs(nb_substeps + 1, nb_states)[None] # get initial sequences of states
    cur_Bs = all_Bs[:,:nb_states**(nb_substeps + 1),:nb_substeps + 1]

    cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int) #states of interest for the current displacement
    cur_nb_Bs = cur_Bs.shape[1]
    # compute the vector of diffusion stds knowing the current states
    ds = cp.array(ds)
    Fs = cp.array(Fs)
    
    LT = get_Ts_from_Bs(cur_states, TrMat) # Log proba of transitions per step
    LF = cp.log(Fs[cur_states[:,:,-1]]) # Log proba of finishing/starting in a given state (fractions)
    
    LP = LT + LF #+ compensate_leaving
    # current log proba of seeing the track
    LP = cp.repeat(LP, nb_Tracks, axis = 0)
    cur_ds = ds[cur_states]
    cur_ds = (cur_ds[:,:,1:]**2 + cur_ds[:,:,:-1]**2)**0.5 / 2**0.5 # assuming a transition at the middle of the substeps
    # we can average the variances of displacements per step to get the actual std of displacements per step
    cur_ds = cp.mean(cur_ds**2, axis = 2)**0.5
    cur_ds = cur_ds[:,:,None]
    cur_ds = cp.array(cur_ds)
    
    sub_Bs = cur_Bs.copy()[:,:cur_Bs.shape[1]//nb_states,:nb_substeps] # list of possible current states we can meet to compute the proba of staying in the FOV
    sub_ds = cp.mean(ds[sub_Bs]**2, axis = 2)**0.5 # corresponding list of d
    sub_ds = asnumpy(sub_ds)
    
    p_stay = np.ones(sub_ds.shape[-1])
    for cell_len in cell_dims:
        xs = np.linspace(0+cell_len/2000,cell_len-cell_len/2000,1000)
        cur_p_stay = ((cp.mean(scipy.stats.norm.cdf((cell_len-xs[:,None])/(sub_ds+1e-200)) - scipy.stats.norm.cdf(-xs[:,None]/(sub_ds+1e-200)),0))) # proba to stay in the FOV for each of the possible cur Bs
        p_stay = p_stay*cur_p_stay
    p_stay = cp.array(p_stay)
    Lp_stay = cp.log(p_stay * (1-pBL)) # proba for the track to survive = both stay in the FOV and not bleach
    
    # inject the first position to get the associated Km and Ks :
    Km, Ks = first_log_integrale_dif(Cs[:,:, nb_locs-current_step], LocErr, cur_ds)
    current_step += 1
    Km = cp.repeat(Km, cur_nb_Bs, axis = 1)
    removed_steps = 0
    #TrMat = np.array([[0.9,0.1],[0.2,0.8]])
    while current_step <= nb_locs-1:
        # update cur_Bs to describe the states at the next step :
        #cur_Bs = get_all_Bs(current_step*nb_substeps+1 - removed_steps, nb_states)[None]
        cur_Bs = all_Bs[:,:nb_states**(current_step*nb_substeps+1 - removed_steps),:current_step*nb_substeps+1 - removed_steps]
        
        cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int)
        # compute the vector of diffusion stds knowing the states at the current step
        cur_ds = ds[cur_states]
        cur_ds = (cur_ds[:,:,1:]**2 + cur_ds[:,:,:-1]**2)**0.5 / 2**0.5 # assuming a transition at the middle of the substeps
        cur_ds = cp.mean(cur_ds**2, axis = 2)**0.5
        cur_ds = cur_ds[:,:,None]
        LT = get_Ts_from_Bs(cur_states, TrMat)

        #np.arange(32)[None][:,TT]
        #np.arange(32)[None][:,1:][:,TT[:-1]]
        # repeat the previous matrix to account for the states variations due to the new position
        Km = cp.repeat(Km, nb_states**nb_substeps , axis = 1)
        Ks = cp.repeat(Ks, nb_states**nb_substeps, axis = 1)
        LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)
        # inject the next position to get the associated Km, Ks and Constant describing the integral of 3 normal laws :
        Km, Ks, LC = log_integrale_dif(Cs[:,:,nb_locs-current_step], LocErr, cur_ds, Km, Ks)
        #print('integral',time.time() - t0); t0 = time.time()
        if current_step >= min_len :
            LL = Lp_stay[np.argmax(np.all(cur_states[:,None,:,:-1] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states
        else:
            LL = 0
        
        LP += LT + LC + LL # current (log) constants associated with each track and sequences of states
        del LT, LC
        cur_nb_Bs = len(cur_Bs[0]) # current number of sequences of states
        
        ''''idea : the position and the state 6 steps ago should not impact too much the 
        probability of the next position so the Km and Ks of tracks with the same 6 last 
        states must be very similar, we can then fuse the parameters of the pairs of Bs
        which vary only for the last step (7) and sum their probas'''
        if current_step < nb_locs-1:
            while cur_nb_Bs >= nb_states**frame_len:
                if do_preds :
                    newKs = cp.array((Ks**2 + LocErr**2)**0.5)[:,:,0]
                    log_integrated_term = -cp.log(2*np.pi*newKs**2) - cp.sum((Cs[:,:,nb_locs-current_step] - Km)**2,axis=2)/(2*newKs**2)
                    LF = 0 #cp.log(Fs[cur_Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
                    
                    test_LP = LP + log_integrated_term + LF
                    
                    if np.max(test_LP)>600: # avoid overflow of exponentials, mechanically also reduces the weight of longest tracks
                        test_LP = test_LP - (np.max(test_LP)-600)

                    P = np.exp(test_LP)
                    
                    for state in range(nb_states):
                        B_is_state = cur_Bs[:,:,-1] == state
                        preds[:,nb_locs-current_step+frame_len-2, state] = asnumpy(np.sum(B_is_state*P,axis = 1)/np.sum(P,axis = 1))
                cur_Bs = cur_Bs[:,:cur_nb_Bs//nb_states, :-1]
                Km, Ks, LP = fuse_tracks(Km, Ks, LP, cur_nb_Bs, nb_states)
                cur_nb_Bs = len(cur_Bs[0])
                removed_steps += 1
        #print('frame',time.time() - t0)
        #print(current_step,time.time() - t0)
        current_step += 1

    all_Bs.shape
    cur_Bs.shape
    
    if not isBL:
        LL = 0
    else:
        cur_Bs = get_all_Bs(np.round(np.log(cur_nb_Bs)/np.log(nb_states)+nb_substeps).astype(int), nb_states)[None]
        cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int)
        len(cur_Bs[0])
        LT = get_Ts_from_Bs(cur_states, TrMat)
        #cur_states = cur_states[:,:,0]
        # repeat the previous matrix to account for the states variations due to the new position
        Km = cp.repeat(Km, nb_states**nb_substeps , axis = 1)
        Ks = cp.repeat(Ks, nb_states**nb_substeps, axis = 1)
        LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)
        
        #LL = Lp_stay[np.argmax(np.all(cur_states[:,None] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states
        #end_p_stay = p_stay[np.argmax(np.all(cur_states[:,None:,:-1] == sub_Bs[:,:,None],-1),1)]
        end_p_stay = p_stay[cur_states[:,None:,:-1]][:,:,0]
        end_p_stay.shape
        LL = cp.log(pBL + (1-end_p_stay) - pBL * (1-end_p_stay)) + LT
        
    newKs = cp.array((Ks**2 + LocErr**2)**0.5)[:,:,0]
    log_integrated_term = -cp.log(2*np.pi*newKs**2) - cp.sum((Cs[:,:,0] - Km)**2,axis=2)/(2*newKs**2)
    #LF = cp.log(Fs[cur_Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
    #LF = cp.log(0.5)
    # cp.mean(cp.log(Fs[cur_Bs[:,:,:].astype(int)]), 2) # Log proba of starting in a given state (fractions)
    LP += log_integrated_term + LL
    
    pred_LP = LP
    if np.max(LP)>600: # avoid overflow of exponentials, mechanically also reduces the weight of longest tracks
        pred_LP = LP - (np.max(LP)-600)
    
    P = np.exp(pred_LP)
    if do_preds :
        for state in range(nb_states):
            B_is_state = cur_Bs[:,:] == state
            preds[:,0:frame_len, state] = asnumpy(np.sum(B_is_state[:,:,isBL:]*P[:,:,None],axis = 1)/np.sum(P[:,:,None],axis = 1)) # index isBL is here to remove the additional position infered to take leaving the FOV into account when isBL (when the track stops)
        preds = preds[:,::-1]
    return LP, cur_Bs, preds

def fuse_tracks(Km, Ks, LP, cur_nb_Bs, nb_states = 2):
    '''
    The probabilities of the pairs of tracks must be added
    I chose to define the updated Km and Ks as the weighted average (of the variance for Ks)
    but other methods may be better
    As I must divid by a sum of exponentials which can be equal to zero because of underflow
    I correct the values in the exponetial to keep the maximal exp value at 0
    '''
    # cut the matrixes so the resulting matrices only vary for their last state
    I = cur_nb_Bs//nb_states
    LPk = []
    Kmk = []
    Ksk = []
    for k in range(nb_states):
        LPk.append(LP[:, k*I:(k+1)*I])# LP of which the last state is k
        Kmk.append(Km[:, k*I:(k+1)*I])# Km of which the last state is k
        Ksk.append(Ks[:, k*I:(k+1)*I])# Ks of which the last state is k
        
    LPk = cp.array(LPk)
    Kmk = cp.array(Kmk)
    Ksk = cp.array(Ksk)
    
    maxLP = cp.max(LPk, axis = 0, keepdims = True)
    Pk = cp.exp(LPk - maxLP)
    
    #sum of the probas of the 2 corresponding matrices :
    SP = cp.sum(Pk, axis = 0, keepdims = True)
    ak = Pk/SP
    
    # update the parameters, this step is tricky as an approximation of a gaussian mixture by a simple gaussian
    Km = cp.sum(ak[:,:,:,None] * Kmk, axis=0)
    Ks = cp.sum((ak[:,:,:,None] * Ksk**2), axis=0)**0.5
    del ak
    LP = maxLP + np.log(SP)
    LP = LP[0]
    # cur_Bs = cur_Bs[:,:I, :-1]
    # np.mean(np.abs(Km0-Km1)) # to verify how far they are, I found a difference of 0.2nm for D = 0.1um2/s, LocErr=0.02um and 6 frames
    # np.mean(np.abs(Ks0-Ks1))
    return Km, Ks, LP

def get_all_Bs(nb_Cs, nb_states):
    '''
    produces a matrix of the possible sequences of states
    '''
    Bs_ID = np.arange(nb_states**nb_Cs)
    all_Bs = np.zeros((nb_states**nb_Cs, nb_Cs), int)
    
    for k in range(all_Bs.shape[1]):
        cur_row = np.mod(Bs_ID,nb_states**(k+1))
        Bs_ID = (Bs_ID - cur_row)
        all_Bs[:,k] = cur_row//nb_states**k
    return all_Bs

def get_Ts_from_Bs(all_Bs, TrMat):
    '''
    compute the probability of the sequences of states according to the markov transition model
    '''
    LT = cp.zeros((all_Bs.shape[:2]))
    # change from binary base 10 numbers to identify the consecutive states (from ternary if 3 states) 
    for k in range(len(all_Bs[0,0])-1):
        LT += cp.log(TrMat[all_Bs[:,:,k], all_Bs[:,:,k+1]])
    return LT

def Proba_Cs(Cs, LocErr, ds, Fs, TrMat,pBL,isBL, cell_dims, nb_substeps, frame_len, min_len):
    '''
    inputs the observed localizations and determine the probability of 
    observing these data knowing the localization error, D the diffusion coef,
    pu the proba of unbinding per step and pb the proba of binding per step
    sum the proba of Cs inter Bs (calculated with P_Cs_inter_bound_stats)
    over all Bs to get the proba of Cs (knowing the initial position c0)
    '''

    LP_CB, _, _  = P_Cs_inter_bound_stats(Cs, LocErr, ds, Fs, TrMat, pBL,isBL,cell_dims, nb_substeps, frame_len, do_preds = 0,  min_len = min_len)
    # calculates P(C) the sum of P(C inter B) for each track
    max_LP = np.max(LP_CB, axis = 1, keepdims = True)
    LP_CB = LP_CB - max_LP
    max_LP = max_LP[:,0]
    P_CB = np.exp(LP_CB)
    P_C = cp.sum(P_CB, axis = 1) # sum over B
    LP_C = np.log(P_C) + max_LP # back to log proba of C without overflow due to exponential
    return LP_C

def predict_Bs(all_tracks,
               dt,
               params,
               cell_dims=[1],
               nb_states=2,
               frame_len=12):
    '''
    inputs the observed localizations and parameters and determines the proba
    of each localization to be in a given state.
    
    arguments:
    all_tracks: dict describing the tracks with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = x, y position.
    params: lmfit parameters used for the model.
    dt: time in between frames.
    cell_dims: dimension limits (um). estimated_vals, min_values, max_values should be changed accordingly to describe all states and transitions.
    nb_states: number of states. estimated_vals, min_values, max_values should be changed accordingly to describe all states and transitions.
    frame_len: number of frames for which the probability is perfectly computed. See method of the paper for more details.
    
    outputs:
    pred_Bs: dict describing the state probability of each track for each time position with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = state.
    extrack.visualization.visualize_states_durations
    '''
    nb_substeps=1 # substeps should not impact the step labelling
    LocErr, ds, Fs, TrMat, pBL = extract_params(params, dt, nb_states, nb_substeps)
    all_pred_Bs = []

    l_list = np.sort(np.array(list(all_tracks.keys())).astype(int)).astype(str)
    min_len = int(l_list[0])
    max_len = int(l_list[-1])
    
    for l in l_list:
        if len(all_tracks[l]) > 0:
            if int(l) == max_len:
                isBL = 1
            else:
                isBL = 0
            LP_Cs, trunkated_Bs, pred_Bs = P_Cs_inter_bound_stats(all_tracks[l], LocErr, ds, Fs, TrMat, pBL,isBL, cell_dims, nb_substeps = 1, frame_len = frame_len, do_preds = 1, min_len = min_len)
            all_pred_Bs.append(pred_Bs)
    
    all_pred_Bs_dict = {}
    for pred_Bs in all_pred_Bs:
        all_pred_Bs_dict[str(pred_Bs.shape[1])] = pred_Bs

    return all_pred_Bs_dict

def extract_params(params, dt, nb_states, nb_substeps):
    '''
    turn the parameters which differ deppending on the number of states into lists
    ds (diffusion lengths), Fs (fractions), TrMat (substep transiton matrix)
    '''
    LocErr = params['LocErr'].value
    
    param_names = np.sort(list(params.keys()))

    Ds = []
    Fs = []
    for param in param_names:
        if param.startswith('D') and len(param)<3:
            Ds.append(params[param].value)
        elif param.startswith('F'):
            Fs.append(params[param].value)
    Ds = np.array(Ds)
    Fs = np.array(Fs)
    TrMat = np.zeros((len(Ds),len(Ds)))
    for param in params:
        if param == 'pBL':
            pBL = params[param].value
        elif param.startswith('p'):
            i = int(param[1])
            j = int(param[2])
            TrMat[i,j] = params[param].value
    
    TrMat = TrMat/nb_substeps
    

    TrMat = 1 - np.exp(-TrMat)
    TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 1-np.sum(TrMat,1)
    
    #print(TrMat)
    ds = np.sqrt(2*Ds*dt)
    return LocErr, ds, Fs, TrMat, pBL

def pool_star_proba(args):
    return Proba_Cs(*args)

def cum_Proba_Cs(params, all_tracks, dt, cell_dims, nb_states, nb_substeps, frame_len, verbose = 1, workers = 1):
    '''
    each probability can be multiplied to get a likelihood of the model knowing
    the parameters LocErr, D0 the diff coefficient of state 0 and F0 fraction of
    state 0, D1 the D coef at state 1, p01 the probability of transition from
    state 0 to 1 and p10 the proba of transition from state 1 to 0.
    here sum the logs(likelihood) to avoid too big numbers
    '''
    LocErr, ds, Fs, TrMat, pBL = extract_params(params, dt, nb_states, nb_substeps)
    min_len = all_tracks[0].shape[1]
    
    if np.all(TrMat>0) and np.all(Fs>0):
        Cum_P = 0
        Csss = []
        isBLs = []
        for k in range(len(all_tracks)):
            if k == len(all_tracks)-1:
                isBL = 0 # last position correspond to tracks which didn't disapear within maximum track length
            else:
                isBL = 1
            Css = all_tracks[k]
            nb_max = 50
            for n in range(int(np.ceil(len(Css)/nb_max))):
                Csss.append(Css[n*nb_max:(n+1)*nb_max])
                if k == len(all_tracks)-1:
                    isBLs.append(0) # last position correspond to tracks which didn't disapear within maximum track length
                else:
                    isBLs.append(1)
        Csss.reverse()
        args_prod = np.array(list(product(Csss, [LocErr], [ds], [Fs], [TrMat],[pBL], [0],[cell_dims], [nb_substeps], [frame_len], [min_len])))
        args_prod[:, 6] = isBLs

        if workers >= 2:
            with multiprocessing.Pool(workers) as pool:
                LP = pool.map(pool_star_proba, args_prod)
        else:
            LP = []
            for args in args_prod:
                LP.append(pool_star_proba(args))
        
        Cum_P += cp.sum(cp.concatenate(LP))
        Cum_P = asnumpy(Cum_P)
        
        if verbose == 1:
            q = [param + ' = ' + str(np.round(params[param].value, 4)) for param in params]
            print(Cum_P, q)
        else:
            print('.', end='')
        out = - Cum_P # normalize by the number of tracks and number of displacements
    else:
        out = 1e100
        print('x',end='')
    if out == np.nan:
        out = 1e100
        print('inputs give nans')
    return out

def get_params(nb_states = 2,
               steady_state = False,
               vary_params = {'LocErr' : True, 'D0' : True, 'D1' : True, 'F0' : True, 'p01' : True, 'p10' : True, 'pBL' : True},
               estimated_vals = {'LocErr' : 0.025, 'D0' : 1e-20, 'D1' : 0.05, 'F0' : 0.45, 'p01' : 0.05, 'p10' : 0.05, 'pBL' : 0.1},
               min_values = {'LocErr' : 0.007, 'D0' : 1e-12, 'D1' : 0.00001, 'F0' : 0.001, 'p01' : 0.01, 'p10' : 0.01, 'pBL' : 0.01},
               max_values = {'LocErr' : 0.6, 'D0' : 1, 'D1' : 10, 'F0' : 0.999, 'p01' : 1., 'p10' : 1., 'pBL' : 0.99}):
    
    if  nb_states == 2:
        if steady_state:
            print(estimated_vals)
            param_kwargs = [{'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : max_values['D0'], 'vary' : vary_params['D0']},
                            {'name' : 'D1_minus_D0', 'value' : estimated_vals['D1'] - estimated_vals['D0'], 'min' : min_values['D1']-min_values['D0'], 'max' : max_values['D1'], 'vary' : vary_params['D1']},
                            {'name' : 'D1', 'expr' : 'D0 + D1_minus_D0'},
                            {'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' :  min_values['LocErr'],'max' :  max_values['LocErr'], 'vary' : vary_params['LocErr']},
                            {'name' : 'F0', 'value' : estimated_vals['F0'], 'min' :  min_values['F0'], 'max' :  max_values['F0'], 'vary' :  vary_params['F0']},
                            {'name' : 'F1', 'expr' : '1 - F0'},
                            {'name' : 'p01', 'value' : estimated_vals['p01'], 'min' :  min_values['p01'], 'max' :  max_values['p01'], 'vary' :  vary_params['p01']},
                            {'name' : 'p10', 'expr' : 'p01/(1/F0-1)'},
                            {'name' : 'pBL', 'value' : estimated_vals['pBL'], 'min' :  min_values['pBL'], 'max' :  max_values['pBL'], 'vary' : vary_params['pBL']}]
        else :
            param_kwargs = [{'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : max_values['D0'], 'vary' : vary_params['D0']},
                            {'name' : 'D1_minus_D0', 'value' : estimated_vals['D1'] - estimated_vals['D0'], 'min' : min_values['D1']-min_values['D0'], 'max' : max_values['D1'], 'vary' : vary_params['D1']},
                            {'name' : 'D1', 'expr' : 'D0 + D1_minus_D0' },
                            {'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' :  min_values['LocErr'],'max' :  max_values['LocErr'], 'vary' : vary_params['LocErr']},
                            {'name' : 'F0', 'value' : estimated_vals['F0'], 'min' :  min_values['F0'], 'max' :  max_values['F0'], 'vary' :  vary_params['F0']},
                            {'name' : 'F1', 'expr' : '1 - F0'},
                            {'name' : 'p01', 'value' : estimated_vals['p01'], 'min' :  min_values['p01'], 'max' :  max_values['p01'], 'vary' :  vary_params['p01']},
                            {'name' : 'p10', 'value' : estimated_vals['p10'], 'min' :  min_values['p10'], 'max' :  max_values['p10'], 'vary' : vary_params['p10']},
                            {'name' : 'pBL', 'value' : estimated_vals['pBL'], 'min' :  min_values['pBL'], 'max' :  max_values['pBL'], 'vary' : vary_params['pBL']}]

    elif nb_states == 3:
        if not (len(min_values) == 13 and len(max_values) == 13 and len(estimated_vals) == 13 and len(vary_params) == 13):
            raise ValueError('estimated_vals, min_values, max_values and vary_params should all containing 13 parameters for a 3 states model')

        if steady_state:
            param_kwargs = [{'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' : min_values['LocErr'], 'max' : max_values['LocErr'] , 'vary' : vary_params['LocErr']},
                            {'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : 0.3, 'brute_step' : 0.04, 'vary' : vary_params['D0']},
                            {'name' : 'D1_minus_D0', 'value' : estimated_vals['D1'] - estimated_vals['D0'], 'min' : 0, 'max' : max_values['D1'], 'brute_step' : 0.04, 'vary' : vary_params['D1']},
                            {'name' : 'D1', 'expr' : 'D0+D1_minus_D0'},
                            {'name' : 'D2_minus_D1', 'value' : estimated_vals['D2'] - estimated_vals['D1'], 'min' : 0, 'max' : max_values['D2'], 'vary' : vary_params['D2']},
                            {'name' : 'D2', 'expr' : 'D1+D2_minus_D1'},
                            {'name' : 'p01', 'value' : estimated_vals['p01'], 'min' : min_values['p01'], 'max' : max_values['p01'], 'vary' : vary_params['p01']},
                            {'name' : 'p02', 'value' : estimated_vals['p02'], 'min' : min_values['p02'], 'max' : max_values['p02'], 'vary' : vary_params['p02']},
                            {'name' : 'p10', 'value' : estimated_vals['p10'], 'min' : min_values['p10'], 'max' : max_values['p10'], 'vary' : vary_params['p10']},
                            {'name' : 'p12', 'value' : estimated_vals['p12'], 'min' : min_values['p12'], 'max' : max_values['p12'], 'vary' : vary_params['p12']},
                            {'name' : 'p20', 'value' : estimated_vals['p20'], 'min' : min_values['p20'], 'max' : max_values['p20'], 'vary' : vary_params['p20']},
                            {'name' : 'p21', 'value' : estimated_vals['p21'], 'min' : min_values['p21'], 'max' : max_values['p21'], 'vary' : vary_params['p21']},
                            {'name' : 'F0', 'expr' : '(p10*(p21+p20)+p20*p12)/((p01)*(p12 + p21) + p02*(p10 + p12 + p21) + p01*p20 + p21*p10 + p20*(p10+p12))'},
                            {'name' : 'F1', 'expr' : '(F0*p01 + (1-F0)*p21)/(p10 + p12 + p21)'},
                            {'name' : 'F2', 'expr' : '1-F0-F1'},
                            {'name' : 'pBL', 'value' : estimated_vals['pBL'], 'min' :  min_values['pBL'], 'max' :  max_values['pBL'], 'vary' : vary_params['pBL']}]
        else:
            param_kwargs = [{'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' : min_values['LocErr'], 'max' : max_values['LocErr'] , 'vary' : vary_params['LocErr']},
                            {'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : 0.3, 'brute_step' : 0.04, 'vary' : vary_params['D0']},
                            {'name' : 'D1_minus_D0', 'value' : estimated_vals['D1'] - estimated_vals['D0'], 'min' : 0, 'max' : max_values['D1'], 'brute_step' : 0.04, 'vary' : vary_params['D1']},
                            {'name' : 'D1', 'expr' : 'D0+D1_minus_D0'},
                            {'name' : 'D2_minus_D1', 'value' : estimated_vals['D2'] - estimated_vals['D1'], 'min' : 0, 'max' : max_values['D2'], 'vary' : vary_params['D2']},
                            {'name' : 'D2', 'expr' : 'D1+D2_minus_D1'},
                            {'name' : 'p01', 'value' : estimated_vals['p01'], 'min' : min_values['p01'], 'max' : max_values['p01'], 'vary' : vary_params['p01']},
                            {'name' : 'p02', 'value' : estimated_vals['p02'], 'min' : min_values['p02'], 'max' : max_values['p02'], 'vary' : vary_params['p02']},
                            {'name' : 'p10', 'value' : estimated_vals['p10'], 'min' : min_values['p10'], 'max' : max_values['p10'], 'vary' : vary_params['p10']},
                            {'name' : 'p12', 'value' : estimated_vals['p12'], 'min' : min_values['p12'], 'max' : max_values['p12'], 'vary' : vary_params['p12']},
                            {'name' : 'p20', 'value' : estimated_vals['p20'], 'min' : min_values['p20'], 'max' : max_values['p20'], 'vary' : vary_params['p20']},
                            {'name' : 'p21', 'value' : estimated_vals['p21'], 'min' : min_values['p21'], 'max' : max_values['p21'], 'vary' : vary_params['p21']},
                            {'name' : 'F0', 'value' : estimated_vals['F0'], 'min' : min_values['F0'], 'max' : max_values['F0'], 'vary' : vary_params['F0']},
                            {'name' : 'F1_minus_F0', 'value' : (estimated_vals['F1'])/(1-estimated_vals['F0']), 'min' : min_values['F1'], 'max' : max_values['F1'], 'vary' : vary_params['F1']},
                            {'name' : 'F1', 'expr' : 'F1_minus_F0*(1-F0)'},
                            {'name' : 'F2', 'expr' : '1-F0-F1'},
                            {'name' : 'pBL', 'value' : estimated_vals['pBL'], 'min' :  min_values['pBL'], 'max' :  max_values['pBL'], 'vary' : vary_params['pBL']}]
    else :
        param_kwargs = [{'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' : min_values['LocErr'], 'max' : max_values['LocErr'] , 'vary' : vary_params['LocErr']}]
        Ds = []
        Fs = []
        for param in list(vary_params.keys()):
            if param.startswith('D'):
                Ds.append(param)
            if param.startswith('F'):
                Fs.append(param)
        param_kwargs.append({'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : 0.3, 'brute_step' : 0.04, 'vary' : vary_params['D0']})
        last_D = 'D0'
        sum_Ds =  estimated_vals['D0']
        expr = 'D0'
        for D in Ds[1:]:
            param_kwargs.append({'name' : D + '_minus_' + last_D, 'value' : estimated_vals[D] - sum_Ds, 'min' : 0, 'max' : max_values[D] , 'vary' : vary_params[D]})
            expr = expr + '+' + D + '_minus_' + last_D
            param_kwargs.append({'name' : D, 'expr' : expr})
            last_D = D
            sum_Ds += estimated_vals[D]
        
        param_kwargs.append({'name' : 'F0', 'value' : estimated_vals['F0'], 'min' : min_values['F0'], 'max' : 0.3, 'brute_step' : 0.04, 'vary' : vary_params['F0']})
        frac = 1-estimated_vals['F0']
        expr = '1-F0'        
        
        for F in Fs[1:len(Ds)-1]:
            param_kwargs.append({'name' : F , 'value' : estimated_vals[F], 'min' : 0.001, 'max' : 0.99 , 'vary' : vary_params[F]})
            frac = frac - 1
            expr = expr + '-' + F
        param_kwargs.append({'name' : 'F'+str(len(Ds)-1), 'expr' : expr})
        
        for param in list(vary_params.keys()):
            if param.startswith('p'):
                param_kwargs.append({'name' : param, 'value' : estimated_vals[param], 'min' : min_values[param], 'max' : max_values[param] , 'vary' : vary_params[param]})
    
    params = Parameters()
    [params.add(**param_kwargs[k]) for k in range(len(param_kwargs))]
    return params

def get_2DSPT_params(all_tracks,
                     dt,
                     nb_substeps = 1,
                     nb_states = 2,
                     frame_len = 8,
                     verbose = 1,
                     workers = 1,
                     method = 'powell',
                     steady_state = False,
                     cell_dims = [1], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                     vary_params = {'LocErr' : True, 'D0' : True, 'D1' : True, 'F0' : True, 'p01' : True, 'p10' : True, 'pBL' : True},
                     estimated_vals =  {'LocErr' : 0.025, 'D0' : 1e-20, 'D1' : 0.05, 'F0' : 0.45, 'p01' : 0.05, 'p10' : 0.05, 'pBL' : 0.1},
                     min_values = {'LocErr' : 0.007, 'D0' : 1e-12, 'D1' : 0.00001, 'F0' : 0.001, 'p01' : 0.001, 'p10' : 0.001, 'pBL' : 0.001},
                     max_values = {'LocErr' : 0.6, 'D0' : 1, 'D1' : 10, 'F0' : 0.999, 'p01' : 1., 'p10' : 1., 'pBL' : 0.99}):
    '''
    fitting the parameters to the data set

    arguments:
    all_tracks: dict describing the tracks with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = x, y position.
    dt: time in between frames.
    cell_dims: dimension limits (um).
    nb_substeps: number of virtual transition steps in between consecutive 2 positions.
    nb_states: number of states. estimated_vals, min_values, max_values should be changed accordingly to describe all states and transitions.
    frame_len: number of frames for which the probability is perfectly computed. See method of the paper for more details.
    verbose: if 1, print the intermediate values for each iteration of the fit.
    steady_state: True if tracks are considered at steady state (fractions independent of time), this is most likely not true as tracks join and leave the FOV.
    vary_params: dict specifying if each parameters should be changed (True) or not (False).
    estimated_vals: initial values of the fit. (stay constant if parameter fixed by vary_params). estimated_vals must be in between min_values and max_values even if fixed.
    min_values: minimal values for the fit.
    max_values: maximal values for the fit.

    outputs:
    model_fit: lmfit model
    
    in case of 3 states models vary_params, estimated_vals, min_values and max_values can be replaced :
    
    vary_params = {'LocErr' : True, 'D0' : False, 'D1' :  True, 'D2' : True, 'F0' : True, 'F1' : True, 'p01' : True, 'p02' : True, 'p10' : True,'p12' :  True,'p20' :  True, 'p21' : True, 'pBL' : True},
    estimated_vals = {'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'F0' : 0.33,  'F1' : 0.33, 'p01' : 0.1, 'p02' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p20' :0.1, 'p21' :0.1, 'pBL' : 0.1},
    min_values = {'LocErr' : 0.007, 'D0' : 1e-20, 'D1' : 0.0000001, 'D2' :  0.000001, 'F0' : 0.001,  'F1' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p20' :0.001, 'p21' :0.001, 'pBL' : 0.001},
    max_values = {'LocErr' : 0.6, 'D0' : 1e-20, 'D1' : 1, 'D2' :  10, 'F0' : 0.999,  'F1' : 0.999, 'p01' : 1, 'p02' : 1, 'p10' : 1, 'p12' : 1, 'p20' : 1, 'p21' : 1, 'pBL' : 0.99}
    
    in case of 4 states models :
    
    vary_params = {'LocErr' : True, 'D0' : True, 'D1' : True, 'D2' :  True, 'D3' : True, 'F0' : True,  'F1' : True, 'F2' : True, 'p01' : True, 'p02' : True, 'p03' : True, 'p10' : True, 'p12' : True, 'p13' : True, 'p20' :True, 'p21' :True, 'p23' : True, 'p30' :True, 'p31' :True, 'p32' : True, 'pBL' : True}
    estimated_vals = {'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'D3' : 0.5, 'F0' : 0.1,  'F1' : 0.2, 'F2' : 0.3, 'p01' : 0.1, 'p02' : 0.1, 'p03' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p13' : 0.1, 'p20' :0.1, 'p21' :0.1, 'p23' : 0.1, 'p30' :0.1, 'p31' :0.1, 'p32' : 0.1, 'pBL' : 0.1}
    min_values = {'LocErr' : 0.005, 'D0' : 0, 'D1' : 0, 'D2' :  0.001, 'D3' : 0.001, 'F0' : 0.001,  'F1' : 0.001, 'F2' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p03' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p13' : 0.001, 'p20' :0.001, 'p21' :0.001, 'p23' : 0.001, 'p30' :0.001, 'p31' :0.001, 'p32' : 0.001, 'pBL' : 0.001}
    max_values = {'LocErr' : 0.023, 'D0' : 1, 'D1' : 1, 'D2' :  10, 'D3' : 100, 'F0' : 0.999,  'F1' : 0.999, 'F2' : 0.999, 'p01' : 1, 'p02' : 1, 'p03' : 1, 'p10' :1, 'p12' : 1, 'p13' : 1, 'p20' : 1, 'p21' : 1, 'p23' : 1, 'p30' : 1, 'p31' : 1, 'p32' : 1, 'pBL' : 0.99}
    '''
    if nb_states == 2:
        if not (len(min_values) == 7 and len(max_values) == 7 and len(estimated_vals) == 7 and len(vary_params) == 7):
            raise ValueError('estimated_vals, min_values, max_values and vary_params should all containing 7 parameters')
    elif nb_states == 3:
        if len(vary_params) != 13:
            vary_params = {'LocErr' : True, 'D0' : True, 'D1' :  True, 'D2' : True, 'F0' : True, 'F1' : True, 'p01' : True, 'p02' : True, 'p10' : True,'p12' :  True,'p20' :  True, 'p21' : True, 'pBL' : True},
        if len(estimated_vals) != 13:
            estimated_vals = {'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'F0' : 0.33,  'F1' : 0.33, 'p01' : 0.1, 'p02' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p20' :0.1, 'p21' :0.1, 'pBL' : 0.1},
        if len(min_values) != 13:
            min_values = {'LocErr' : 0.007, 'D0' : 1e-20, 'D1' : 0.0000001, 'D2' :  0.000001, 'F0' : 0.001,  'F1' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p20' :0.001, 'p21' :0.001, 'pBL' : 0.001},
        if len(max_values) != 13:
            max_values = {'LocErr' : 0.023, 'D0' : 1, 'D1' : 1, 'D2' :  10, 'F0' : 0.999,  'F1' : 0.999, 'p01' : 1, 'p02' : 1, 'p10' :1, 'p12' : 1, 'p20' : 1, 'p21' : 1, 'pBL' : 0.99}
    elif nb_states == 4:
        if len(vary_params) != 21:
            vary_params = {'LocErr' : True, 'D0' : True, 'D1' : True, 'D2' :  True, 'D3' : True, 'F0' : True,  'F1' : True, 'F2' : True, 'p01' : True, 'p02' : True, 'p03' : True, 'p10' : True, 'p12' : True, 'p13' : True, 'p20' :True, 'p21' :True, 'p23' : True, 'p30' :True, 'p31' :True, 'p32' : True, 'pBL' : True}
        if len(estimated_vals) != 21:
            estimated_vals = {'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'D3' : 0.5, 'F0' : 0.1,  'F1' : 0.2, 'F2' : 0.3, 'p01' : 0.1, 'p02' : 0.1, 'p03' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p13' : 0.1, 'p20' :0.1, 'p21' :0.1, 'p23' : 0.1, 'p30' :0.1, 'p31' :0.1, 'p32' : 0.1, 'pBL' : 0.1}
        if len(min_values) != 21:
            min_values = {'LocErr' : 0.005, 'D0' : 0, 'D1' : 0, 'D2' :  0.001, 'D3' : 0.001, 'F0' : 0.001,  'F1' : 0.001, 'F2' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p03' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p13' : 0.001, 'p20' :0.001, 'p21' :0.001, 'p23' : 0.001, 'p30' :0.001, 'p31' :0.001, 'p32' : 0.001, 'pBL' : 0.001}
        if len(max_values) != 21:
            max_values = {'LocErr' : 0.023, 'D0' : 1, 'D1' : 1, 'D2' :  10, 'D3' : 100, 'F0' : 0.999,  'F1' : 0.999, 'F2' : 0.999, 'p01' : 1, 'p02' : 1, 'p03' : 1, 'p10' :1, 'p12' : 1, 'p13' : 1, 'p20' : 1, 'p21' : 1, 'p23' : 1, 'p30' : 1, 'p31' : 1, 'p32' : 1, 'pBL' : 0.99}
    else:
        if len(vary_params) == 7 or len(estimated_vals) == 7 or len(min_values) == 7 or len(max_values) == 7:
            raise ValueError('vary_params, estimated_vals, min_values and max_values have to be correctly specified if more than 4 states')
    
    if not str(all_tracks.__class__) == "<class 'dict'>":
        raise ValueError('all_tracks should be a dictionary of arrays with n there number of steps as keys')
    
    params = get_params(nb_states, steady_state, vary_params, estimated_vals, min_values, max_values)
    
    print(params)
    
    l_list = np.sort(np.array(list(all_tracks.keys())).astype(int)).astype(str)
    sorted_tracks = []
    for l in l_list:
        if len(all_tracks[l]) > 0 :
            sorted_tracks.append(all_tracks[l])
    all_tracks = sorted_tracks
    
    fit = minimize(cum_Proba_Cs, params, args=(all_tracks, dt, cell_dims, nb_states, nb_substeps, frame_len, verbose, workers), method = method, nan_policy = 'propagate')
    
    return fit