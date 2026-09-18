# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 11:00:24 2025

@author: Franc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:23:30 2022

@author: francois
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

from scipy import linalg
import itertools
import scipy
from lmfit import minimize, Parameters

import multiprocessing
try:
    multiprocessing.set_start_method('fork')
    start_method = 'fork'
except:
    print('multiprocessing is only supported on Linux and MacOS')
    start_method = 'notfork'

from itertools import product

from time import time

try:
    from extrack import numba_kernels as _numba_kernels
except Exception: # numba is optional, the numpy path stays the reference
    _numba_kernels = None

USE_NUMBA = 'auto'

def set_numba(mode = 'auto', threads = None):
    '''
    Turn the numba kernels on or off.

    mode: 'auto' (default) uses them whenever numba is installed and the inputs
        fall inside what they cover, True is the same but raises if numba is
        missing, False always runs the numpy implementation.

    threads: number of threads for the kernels, None (default) to leave numba's
        own setting alone. The recursion is a short parallel region called once
        per time step, so the useful range is narrow: measured on a 24 core
        i9-14900K the best point is around 8 (11-26x over numpy) and 16 or more
        is slower again, the thread pool costing more than the step.

    The kernels compute the same likelihood as the numpy code they replace; they
    exist because that code spends most of its time on per-call dispatch and on
    temporaries rather than on arithmetic. `numba_status()` reports what is in
    use, and the first call to a kernel pays a few seconds of compilation which
    is then cached on disk.
    '''
    global USE_NUMBA
    if mode is True and (_numba_kernels is None or not _numba_kernels.NUMBA_AVAILABLE):
        raise ImportError('numba is not installed: pip install numba, or use set_numba(False)')
    USE_NUMBA = mode
    if threads is not None:
        import numba
        numba.set_num_threads(int(threads))

def numba_active():
    ''' True when the numba kernels will be used for the next likelihood call '''
    if USE_NUMBA is False or GPU_computing:
        return False
    return _numba_kernels is not None and _numba_kernels.NUMBA_AVAILABLE

def numba_status():
    if _numba_kernels is None or not _numba_kernels.NUMBA_AVAILABLE:
        return 'numba is not installed, running the numpy implementation'
    if not numba_active():
        return 'numba is installed but switched off (USE_NUMBA = %s)'%repr(USE_NUMBA)
    import numba
    return 'numba kernels active, %d threads'%numba.get_num_threads()


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
    cur_d2s = ds[cur_states]**2
    cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps
    # we can average the variances of displacements per step to get the actual std of displacements per step
    cur_d2s = cp.mean(cur_d2s, axis = 2)
    cur_d2s = cur_d2s[:,:,None]
    cur_d2s = cp.array(cur_d2s)
    return cur_d2s

'''
l2 = LocErr2[:,:,min(LocErr_index,nb_locs-current_step)]
Ci = Cs[:,:,nb_locs-current_step]
l2.shape
Ci.shape
m_arr.shape
s2_arr.shape
'''

def log_integrale_dif(Ci, l2, cur_d2s, m_arr, s2_arr):
    '''
    integral of the 3 exponetional terms (localization error, diffusion, previous term)
    the integral over r1 of f_l(r1-c1)f_d(r1-r0)f_Ks(r1-m_arr) equals :
    np.exp(-((l**2+Ks**2)*r0**2+(-2*m_arr*l**2-2*Ks**2*c1)*r0+m_arr**2*l**2+(m_arr**2-2*c1*m_arr+c1**2)*d**2+Ks**2*c1**2)/((2*d**2+2*Ks**2)*l**2+2*Ks**2*d**2))/(2*np.pi*Ks*d*l*np.sqrt((d**2+Ks**2)*l**2+Ks**2*d**2))
    which can be turned into the form Constant*fKs(r0 - newm_arr) where fKs is a normal law of std newKs
    the idea is to create a function of integral of integral of integral etc
    dim 0 : tracks
    dim 1 : possible sequences of states
    dim 2 : x,y (z)
    '''
    l2_plus_s2_arr = l2+s2_arr
    new_m = (m_arr*l2 + Ci*s2_arr)/(l2+s2_arr)
    if len(cur_d2s.shape)==1:
        new_s2 = ((cur_d2s*l2 + cur_d2s*s2_arr + l2*s2_arr)/l2_plus_s2_arr)
    else:
        new_s2 = ((cur_d2s*l2 + cur_d2s*s2_arr + l2*s2_arr)/l2_plus_s2_arr)

    if l2_plus_s2_arr.shape[2] == 1: # the variance is shared by all the dimensions, the log term is then simply repeated nb_dims times
        new_K = m_arr.shape[2] * -0.5*cp.log(2*np.pi*(l2_plus_s2_arr[:,:,0])) - cp.sum(((Ci-m_arr).astype(float))**2/(2*l2_plus_s2_arr),axis = 2)
    else:
        new_K = np.sum(-0.5*cp.log(2*np.pi*(l2_plus_s2_arr)), 2) - cp.sum(((Ci-m_arr).astype(float))**2/(2*l2_plus_s2_arr),axis = 2)
    return new_m, new_s2, new_K

#Ci, l2 = Cs[:,:,nb_locs-current_step], LocErr2[:,:,min(LocErr_index,nb_locs-current_step)]
def first_log_integrale_dif(Ci, l2, cur_d2s):
    '''
    convolution of 2 normal laws = normal law (mean = sum of means and variance = sum of variances)
    '''
    s2_arr = l2+cur_d2s
    m_arr = Ci
    return m_arr, s2_arr

def P_Cs_inter_bound_stats(Cs, LocErr, ds, Fs, TrMat, pBL=0.1, isBL = 1, cell_dims = [0.5], nb_substeps=1, frame_len = 4, do_preds = 0, min_len = 3) :
    '''
    compute the product of the integrals over Ri as previousily described
    work in log space to avoid overflow and underflow
    
    Cs : dim 0 = track ID, dim 1 : states, dim 2 : peaks postions through time,
    dim 3 : x, y position
    
    we process by steps, at each step we account for 1 more localization, we compute
    the canstant (LC), the mean (m_arr) and std (Ks) of of the normal distribution 
    resulting from integration.
    
    each step is made of substeps if nb_substeps > 1, and we increase the matrix
    of possible Bs : cur_Bs accordingly
    
    to be able to process long tracks with good accuracy, for each track we fuse m_arr and Ks
    of sequences of states equal exept for the state 'frame_len' steps ago.
    '''
    nb_Tracks = Cs.shape[0]
    nb_locs = Cs.shape[1] # number of localization per track
    
    nb_dims = Cs.shape[2] # number of spatial dimensions (x, y) or (x, y, z)
    Cs = Cs[:,None]
    Cs = cp.array(Cs)
    nb_states = TrMat.shape[0]
    Cs = Cs[:,:,::-1] # I built the model going from the last index to the first index of the reversed positions. Which is equivalent to an iteration from the first position to the last one. 
    LocErr = LocErr[:,None]
    LocErr = LocErr[:,:,::-1] # useful when single peak localization error is inputed,
    LocErr2 = LocErr**2
    if LocErr.shape[2] == 1: # combined to min(LocErr_index, nb_locs-current_step) it will select the right index
        LocErr_index = -1
    elif LocErr.shape[2] == nb_locs:
        LocErr_index = nb_locs
    else:
        raise ValueError("Localization error is not specified correctly, in case of unique localization error specify a float number in estimated_vals['LocErr'].\n If one localization error per dimension, specify a list or 1D array of elements the localization error for each dimension.\n If localization error is predetermined by another method for each position the argument input_LocErr should be a dict for each track length of the 3D arrays corresponding to all_tracks (can be obtained from the reader functions using the opt_colname argument)")
    if do_preds:
        preds = np.zeros((nb_Tracks, nb_locs, nb_states))-1
    else :
        preds = []
    
    if nb_locs < 2:
        raise ValueError('minimal track length = 2, here track length = %s'%nb_locs)
    
    all_Bs = get_all_Bs(frame_len + nb_substeps, nb_states)[None]
    
    sub_Bs = get_all_Bs(nb_substeps, nb_states)[None]
    TrMat = cp.array(TrMat.T)
    current_step = 1
    
    #cur_Bs = get_all_Bs(nb_substeps + 1, nb_states)[None] # get initial sequences of states
    cur_Bs = all_Bs[:,:nb_states**(nb_substeps + 1),:nb_substeps + 1]

    cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int) #states of interest for the current displacement
    cur_nb_Bs = cur_Bs.shape[1]
    # compute the vector of diffusion stds knowing the current states
    ds = cp.array(ds)
    ds2 = ds**2
    Fs = cp.array(Fs)
    
    LT = get_Ts_from_Bs(cur_states, TrMat) # Log proba of transitions per step
    LF = cp.log(Fs[cur_states[:,:,-1]]) # Log proba of finishing/starting in a given state (fractions)
    
    LP = LT + LF #+ compensate_leaving
    # current log proba of seeing the track
    LP = cp.repeat(LP, nb_Tracks, axis = 0)
    cur_d2s = ds2[cur_states]
    cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps

    # we can average the variances of displacements per step to get the actual std of displacements per step
    cur_d2s = cp.mean(cur_d2s, axis = 2)
    cur_d2s = cur_d2s[:,:,None]
    cur_d2s = cp.array(cur_d2s)
    
    sub_Bs = cur_Bs.copy()[:,:cur_Bs.shape[1]//nb_states,:nb_substeps] # list of possible current states we can meet to compute the proba of staying in the FOV
    sub_ds = (cp.mean(ds[sub_Bs]**2, axis = 2)**0.5) # corresponding list of d
    sub_ds = asnumpy(sub_ds)
    
    p_stay = np.ones(sub_ds.shape[-1])
    for cell_len in cell_dims:
        xs = np.linspace(0+cell_len/2000,cell_len-cell_len/2000,1000)
        cur_p_stay = ((cp.mean(scipy.stats.norm.cdf((cell_len-xs[:,None])/(sub_ds+1e-200)) - scipy.stats.norm.cdf(-xs[:,None]/(sub_ds+1e-200)),0))) # proba to stay in the FOV for each of the possible cur Bs
        p_stay = p_stay*cur_p_stay
    p_stay = cp.array(p_stay)
    Lp_stay = cp.log(p_stay * (1-pBL)) # proba for the track to survive = both stay in the FOV and not bleach
    
    # inject the first position to get the associated m_arr and Ks :
    m_arr, s2_arr = first_log_integrale_dif(Cs[:,:, nb_locs-current_step], LocErr2[:,:, min(LocErr_index, nb_locs-current_step)], cur_d2s)
    s2_arr**0.5
    m_arr = cp.repeat(m_arr, cur_nb_Bs, axis = 1)
    removed_steps = 0
    
    if nb_substeps > 1 and 0:
        cur_len = nb_substeps + 1
        fuse_pos = np.arange(1,nb_substeps)
        m_arr, s2_arr, LP, cur_Bs = fuse_tracks_general(m_arr, s2_arr, LP, cur_Bs, cur_len, nb_Tracks, fuse_pos = fuse_pos, nb_states = nb_states, nb_dims = nb_dims)
    
    current_step += 1
    
    np.sum(np.exp(LP), 1)
    
    #TrMat = np.array([[0.9,0.1],[0.2,0.8]])
    while current_step <= nb_locs-1:
        # update cur_Bs to describe the states at the next step :
        #cur_Bs = get_all_Bs(current_step*nb_substeps+1 - removed_steps, nb_states)[None]
        
        #cur_Bs = all_Bs[:,:nb_states**(current_step + nb_substeps - removed_steps),:current_step + nb_substeps - removed_steps]
        cur_Bs = all_Bs[:,:nb_states**(cur_Bs.shape[-1] + nb_substeps),:cur_Bs.shape[-1] + nb_substeps]
        
        cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int)
        # compute the vector of diffusion stds knowing the states at the current step
        cur_d2s = ds[cur_states]**2
        cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps
        cur_d2s = cp.mean(cur_d2s, axis = 2)
        cur_d2s = cur_d2s[:,:,None]
        LT = get_Ts_from_Bs(cur_states, TrMat)

        # repeat the previous matrix to account for the states variations due to the new position
        m_arr = cp.repeat(m_arr, nb_states**nb_substeps , axis = 1)
        s2_arr = cp.repeat(s2_arr, nb_states**nb_substeps, axis = 1)
        LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)
        # inject the next position to get the associated m_arr, s2_arr and Constant describing the integral of 3 normal laws :
        m_arr, s2_arr, LC = log_integrale_dif(Cs[:,:,nb_locs-current_step], LocErr2[:,:,min(LocErr_index,nb_locs-current_step)], cur_d2s, m_arr, s2_arr)
        #print('integral',time.time() - t0); t0 = time.time()
        
        if current_step >= min_len :
            LL = Lp_stay[np.argmax(np.all(cur_states[:,None,:,:-1] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states
        else:
            LL = 0
        
        LP += LT + LC + LL # current (log) constants associated with each track and sequences of states
        del LT, LC
        
        if nb_substeps > 1 and 0:
            cur_len = cur_Bs.shape[-1]
            fuse_pos = np.arange(1,nb_substeps)+nb_substeps
            m_arr, s2_arr, LP, cur_Bs = fuse_tracks_general(m_arr, s2_arr, LP, cur_Bs, cur_len, nb_Tracks, fuse_pos, nb_states = nb_states, nb_dims = nb_dims)

        cur_nb_Bs = len(cur_Bs[0]) # current number of sequences of states
        
        ''''idea : the position and the state 6 steps ago should not impact too much the 
        probability of the next position so the m_arr and s2_arr of tracks with the same 6 last 
        states must be very similar, we can then fuse the parameters of the pairs of Bs
        which vary only for the last step (7) and sum their probas'''
        
        if current_step < nb_locs-1: # do not fuse sequences at the last step as it doesn't improves speed.
            while cur_nb_Bs > nb_states**frame_len:
                if do_preds :
                    #new_s2_arr = cp.array((s2_arr + LocErr2[:,:,min(LocErr_index,nb_locs-current_step)]))[:,:,0]
                    #log_integrated_term = -cp.log(2*np.pi*new_s2_arr) - cp.sum((Cs[:,:,nb_locs-current_step-1] - m_arr)**2,axis=2)/(2*new_s2_arr)
                    new_s2_arr = cp.array((s2_arr + LocErr2[:,:,min(LocErr_index,nb_locs-current_step-1)]))
                    log_integrated_term = cp.sum(-cp.log(2*np.pi*new_s2_arr) - (Cs[:,:,nb_locs-current_step-1] - m_arr)**2 / (2*new_s2_arr), axis=2)
                    LF = 0 #cp.log(Fs[cur_Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
                    
                    test_LP = LP + log_integrated_term + LF
                    
                    if np.max(test_LP)>600: # avoid overflow of exponentials, mechanically also reduces the weight of longest tracks
                        test_LP = test_LP - (np.max(test_LP)-600)

                    P = np.exp(test_LP)
                    
                    for state in range(nb_states):
                        B_is_state = cur_Bs[:,:,-1] == state # get the value of the state at the further time point still considered to compute the proba and remove it.
                        preds[:,nb_locs-current_step+frame_len-1, state] = asnumpy(np.sum(B_is_state*P,axis = 1)/np.sum(P,axis = 1))

                cur_len = cur_Bs.shape[-1]
                fuse_pos = np.arange(cur_len-1,cur_len)
                m_arr, s2_arr, LP, cur_Bs = fuse_tracks_general(m_arr, s2_arr, LP, cur_Bs, cur_len, nb_Tracks, fuse_pos, nb_states = nb_states, nb_dims = nb_dims)
                cur_nb_Bs = len(cur_Bs[0])
                removed_steps += 1
        #print('frame',time.time() - t0)
        #print(current_step,time.time() - t0)
        current_step += 1
    
    if not isBL:
        LL = 0
    else:
        cur_Bs = get_all_Bs(np.round(np.log(cur_nb_Bs)/np.log(nb_states)+nb_substeps).astype(int), nb_states)[None]
        cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int)
        len(cur_Bs[0])
        LT = get_Ts_from_Bs(cur_states, TrMat)
        #cur_states = cur_states[:,:,0]
        # repeat the previous matrix to account for the states variations due to the new position
        m_arr = cp.repeat(m_arr, nb_states**nb_substeps , axis = 1)
        s2_arr = cp.repeat(s2_arr, nb_states**nb_substeps, axis = 1)
        LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)
        
        #LL = Lp_stay[np.argmax(np.all(cur_states[:,None] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states
        #end_p_stay = p_stay[np.argmax(np.all(cur_states[:,None:,:-1] == sub_Bs[:,:,None],-1),1)]
        end_p_stay = p_stay[cur_states[:,None:,:-1]][:,:,0]
        end_p_stay.shape
        LL = cp.log(pBL + (1-end_p_stay) - pBL * (1-end_p_stay)) + LT

    new_s2_arr = cp.array((s2_arr + LocErr2[:,:, min(LocErr_index, nb_locs-current_step)]))
    log_integrated_term = cp.sum(-0.5*cp.log(2*np.pi*new_s2_arr) - (Cs[:,:,0] - m_arr)**2/(2*new_s2_arr),axis=2)
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
            preds[:,0:frame_len+1, state] = asnumpy(np.sum(B_is_state[:,:,isBL:]*P[:,:,None],axis = 1)/np.sum(P[:,:,None],axis = 1)) # index isBL is here to remove the additional position infered to take leaving the FOV into account when isBL (when the track stops)
        preds = preds[:,::-1]
    return LP, cur_Bs, preds


"""
Fusion of the branches of the tree of states : Gaussian mixture reduction.

At each step the recursion carries, for every track and every sequence of states
b, a log weight LP_b and a Gaussian N(r ; m_b, s2_b) over the current position r.
Keeping one branch per sequence of states is impossible (their number grows as
nb_states**track_len), so branches that became nearly equivalent are fused. Fusing
a group of branches means replacing the mixture

    sum_b exp(LP_b) * N(r ; m_b, s2_b)

by a single weighted Gaussian exp(LP) * N(r ; mu, sig2). The best such
approximation, the one that minimizes the Kullback-Leibler divergence to the
mixture, keeps the total weight and matches the two first moments of the mixture,
hence the name moment matching.
"""

FUSION_MOMENT_MATCHING = True # include the spread of the means of the branches in the fused variance (the exact second moment). Set to False to recover the historical behavior.
FUSION_ISOTROPIC_VARIANCE = False # if True the spread is averaged over the dimensions so the fused Gaussian keeps a single variance for all dimensions

def set_fusion_method(moment_matching = True, isotropic_variance = False):
    '''
    select how the Gaussians of the fused branches are combined.

    moment_matching: if True (default) the variance of the fused Gaussian is the
        exact second central moment of the mixture, i.e. the weighted average of
        the variances of the branches plus the weighted spread of their means. If
        False only the first term is kept, which is what ExTrack used to do and
        which underestimates the variance.
    isotropic_variance: if True the spread is averaged over the spatial dimensions
        so the fused Gaussian keeps one variance for all dimensions (matching the
        trace of the second moment instead of its diagonal). If False (default) one
        variance per dimension is kept, which is a strictly better approximation.
    '''
    global FUSION_MOMENT_MATCHING, FUSION_ISOTROPIC_VARIANCE
    FUSION_MOMENT_MATCHING = bool(moment_matching)
    FUSION_ISOTROPIC_VARIANCE = bool(isotropic_variance)

def fused_variance_shape(m_arr, s2_arr):
    '''
    length of the last axis of the variances produced by fuse_gaussians : the spread
    of the means is specific to each dimension, so unless it is averaged out the
    fused Gaussian carries one variance per dimension even when the branches did not.
    '''
    if FUSION_MOMENT_MATCHING and not FUSION_ISOTROPIC_VARIANCE:
        return max(s2_arr.shape[-1], m_arr.shape[-1])
    return s2_arr.shape[-1]

def fuse_gaussians(m_arr, s2_arr, LP, axis):
    '''
    reduce a mixture of Gaussians to a single Gaussian by matching its 2 first moments.

    With w_b = exp(LP_b) / sum_b exp(LP_b) the normalized weight of branch b, the
    fused quantities are, for each spatial dimension j taken separately :

        LP     = log( sum_b exp(LP_b) )
        mu_j   = sum_b w_b m_bj
        sig2_j = sum_b w_b s2_b  +  sum_b w_b (m_bj - mu_j)**2

    The first variance term is the average width of the branches, the second one is
    the spread of their means. Their sum is the law of total variance : it is the
    exact variance of the mixture. Only the first term used to be kept, which made
    the fused Gaussian systematically too narrow. The recursion then behaved as if
    the position of the particle was known more precisely than it actually is, which
    biases the fitted diffusion coefficients, localization error and transition
    rates and makes the state probabilities overconfident.

    Everything is written with scalar formulas, one per dimension, so the result
    stays in the family of distributions the recursion propagates : a mean and a
    variance per dimension, no covariance matrix (log_integrale_dif accepts a
    variance per dimension natively). The only part of the exact second moment that
    is dropped is the off diagonal sum_b w_b (m_bi - mu_i)(m_bj - mu_j), which would
    require a matrix formulation.

    arguments:
    m_arr: means of the branches, the last axis being the spatial dimension.
    s2_arr: variances of the branches, same layout as m_arr except that its last
        axis may be of length 1 when the variance is shared by all the dimensions.
    LP: log weights of the branches, the layout of m_arr without its last axis.
    axis: int or tuple of ints, axes of LP along which the branches are fused.

    outputs:
    new_m_arr, new_s2_arr, new_LP: the moment matched Gaussian and its log weight,
        the fused axes being removed. new_s2_arr holds one variance per dimension
        unless the fusion is set to be isotropic (see set_fusion_method).
    '''
    max_LP = cp.max(LP, axis = axis, keepdims = True) # subtracted to avoid underflow of the exponentials
    weights = cp.exp(LP - max_LP)
    sum_weights = cp.sum(weights, axis = axis, keepdims = True)
    weights = (weights / sum_weights)[..., None] # normalized weights, with an extra axis to match the shape of m_arr

    new_m_arr = cp.sum(weights * m_arr, axis = axis, keepdims = True)
    new_s2_arr = cp.sum(weights * s2_arr, axis = axis, keepdims = True) # average width of the branches

    if FUSION_MOMENT_MATCHING:
        spread = cp.sum(weights * (m_arr - new_m_arr)**2, axis = axis, keepdims = True) # spread of the means of the branches
        if FUSION_ISOTROPIC_VARIANCE:
            spread = cp.mean(spread, axis = -1, keepdims = True)
        new_s2_arr = new_s2_arr + spread

    new_LP = cp.log(sum_weights) + max_LP

    new_m_arr = np.squeeze(new_m_arr, axis = axis)
    new_s2_arr = np.squeeze(new_s2_arr, axis = axis)
    new_LP = np.squeeze(new_LP, axis = axis)

    return new_m_arr, new_s2_arr, new_LP

def fuse_tracks(m_arr, s2_arr, LP, cur_nb_Bs, nb_states = 2):
    '''
    The probabilities of the fused branches are summed and their Gaussians are
    reduced to a single Gaussian carrying the 2 first moments of their mixture
    (see fuse_gaussians).
    As I must divid by a sum of exponentials which can be equal to zero because of underflow
    I correct the values in the exponetial to keep the maximal exp value at 0
    '''
    # cut the matrixes so the resulting matrices only vary for their last state
    I = cur_nb_Bs//nb_states
    LPk = []
    m_arr_k = []
    s2_arr_k = []
    for k in range(nb_states):
        LPk.append(LP[:, k*I:(k+1)*I])# LP of which the last state is k
        m_arr_k.append(m_arr[:, k*I:(k+1)*I])# m_arr of which the last state is k
        s2_arr_k.append(s2_arr[:, k*I:(k+1)*I])# s2_arr of which the last state is k

    LPk = cp.array(LPk)
    m_arr_k = cp.array(m_arr_k)
    s2_arr_k = cp.array(s2_arr_k)

    # approximate the mixture of Gaussians by a single Gaussian with the same 2 first moments
    m_arr, s2_arr, LP = fuse_gaussians(m_arr_k, s2_arr_k, LPk, 0)
    # cur_Bs = cur_Bs[:,:I, :-1]
    # np.mean(np.abs(m_arr0-m_arr1)) # to verify how far they are, I found a difference of 0.2nm for D = 0.1um2/s, LocErr=0.02um and 6 frames
    # np.mean(np.abs(s2_arr0-s2_arr1))
    return m_arr, s2_arr, LP, 

def fuse_tracks_general(m_arr, s2_arr, LP, cur_Bs, cur_len, nb_Tracks, fuse_pos, nb_states = 2, nb_dims = 2):
    '''
    The probabilities of the fused branches are summed and their Gaussians are
    reduced to a single Gaussian carrying the 2 first moments of their mixture
    (see fuse_gaussians).
    As I must divid by a sum of exponentials which can be equal to zero because of underflow
    I correct the values in the exponetial to keep the maximal exp value at 0
    '''
    # cut the matrixes so the resulting matrices only vary for their last state
    
    fuse_idx = np.zeros(cur_len)
    fuse_idx[fuse_pos] = 1
        
    dims = [nb_states]
    if fuse_idx[0] == 0:
        remove_axis = [0]
    else:
        remove_axis = [1]
    previous_idx = fuse_idx[0]
    for idx in fuse_idx[1:]:
        if idx == 1:
            dims.append(nb_states)
            remove_axis.append(1)
        else:
            if previous_idx == 1:
                dims.append(nb_states)
                remove_axis.append(0)
            else:
                dims[-1] = dims[-1] * nb_states
        previous_idx = idx
    
    dims.reverse() # we need to reverse the list to get the good order because of the way the array indexes are working
    remove_axis.reverse()
        
    rm_axis = np.where(np.array(remove_axis))
    rm_axis = tuple(rm_axis[0]+1) # we add 1 as the first dim of our arrays is for the track ID
    
    LP = LP.reshape([nb_Tracks] + dims)
    m_arr = m_arr.reshape([nb_Tracks] + dims + [nb_dims])
    s2_arr = s2_arr.reshape([s2_arr.shape[0]] + dims + [s2_arr.shape[-1]])

    # approximate the mixture of Gaussians by a single Gaussian with the same 2 first moments
    new_m_arr, new_s2_arr, new_LP = fuse_gaussians(m_arr, s2_arr, LP, rm_axis)

    new_cur_Bs = cur_Bs.reshape([1] + dims + [cur_len])
    for i, axis in enumerate(rm_axis):
        new_cur_Bs = np.take(new_cur_Bs, indices = 0, axis = axis -  i)
    new_cur_Bs = np.delete(new_cur_Bs , tuple(fuse_pos), axis =  -1)
    new_cur_Bs = new_cur_Bs.reshape((1, np.prod(new_cur_Bs.shape[1:-1]), cur_len - len(fuse_pos)))

    new_m_arr = new_m_arr.reshape((nb_Tracks, np.prod(new_m_arr.shape[1:-1]), nb_dims))
    new_s2_arr = new_s2_arr.reshape((new_s2_arr.shape[0], np.prod(new_s2_arr.shape[1:-1]), new_s2_arr.shape[-1]))
    new_LP = new_LP.reshape((nb_Tracks, np.prod(new_LP.shape[1:])))
    
    return new_m_arr, new_s2_arr, new_LP, new_cur_Bs

#Cs, LocErr, ds, Fs, TrMat,pBL,isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states = args_prod[0]

def recurrence_step(Ci, l2, cur_d2s, m_arr, s2_arr, LP, LT, LL, rep):
    '''
    One step of the sequences recursion: replicate the carried Gaussians over the
    `rep` possible new states, fold the observation in (log_integrale_dif) and
    accumulate the transition, integration and survival log terms.

    Branch b of the output descends from branch b // rep of the input, which is
    what np.repeat(..., rep, axis = 1) produces.
    '''
    if numba_active():
        nb_Tracks = m_arr.shape[0]
        nb_out = m_arr.shape[1] * rep
        nb_dims = m_arr.shape[2]
        new_m = np.empty((nb_Tracks, nb_out, nb_dims))
        new_s2 = np.empty((nb_Tracks, nb_out, nb_dims))
        new_LP = np.empty((nb_Tracks, nb_out))
        LT_flat = np.ascontiguousarray(np.broadcast_to(LT, (1, nb_out))[0], dtype = float)
        LL_flat = np.ascontiguousarray(np.broadcast_to(LL, (1, nb_out))[0], dtype = float) if np.ndim(LL) else np.zeros(nb_out)
        _numba_kernels.step_kernel(np.ascontiguousarray(Ci[:,0], dtype = float),
                                   np.ascontiguousarray(l2[:,0], dtype = float),
                                   np.ascontiguousarray(cur_d2s, dtype = float),
                                   np.ascontiguousarray(m_arr, dtype = float),
                                   np.ascontiguousarray(s2_arr, dtype = float),
                                   np.ascontiguousarray(LP, dtype = float),
                                   LT_flat, LL_flat, rep, new_m, new_s2, new_LP)
        return new_m, new_s2, new_LP

    m_arr = cp.repeat(m_arr, rep, axis = 1)
    s2_arr = cp.repeat(s2_arr, rep, axis = 1)
    LP = cp.repeat(LP, rep, axis = 1)
    m_arr, s2_arr, LC = log_integrale_dif(Ci, l2, cur_d2s, m_arr, s2_arr)
    return m_arr, s2_arr, LP + LT + LC + LL

def final_integration(C0, l2, m_arr, s2_arr, LP, LL):
    '''
    The last observation of a track: it is folded in without a further
    prediction step, and the end of track term is added.
    '''
    if numba_active():
        LP = np.ascontiguousarray(LP, dtype = float)
        LL_arr = np.ascontiguousarray(np.atleast_2d(LL) + np.zeros(LP.shape[1]), dtype = float)
        _numba_kernels.final_kernel(np.ascontiguousarray(C0[:,0], dtype = float),
                                    np.ascontiguousarray(l2[:,0], dtype = float),
                                    np.ascontiguousarray(m_arr, dtype = float),
                                    np.ascontiguousarray(s2_arr, dtype = float),
                                    LP, LL_arr)
        return LP
    new_s2_arr = cp.array(s2_arr + l2)
    log_integrated_term = cp.sum(-0.5*cp.log(2*np.pi*new_s2_arr) - (C0 - m_arr)**2/(2*new_s2_arr), axis = 2)
    return LP + log_integrated_term + LL

def P_Cs_inter_bound_stats_th(Cs, LocErr, ds, Fs, TrMat, pBL=0.1, isBL = 1, cell_dims = [0.5], nb_substeps=1, frame_len = 6, do_preds = 0, min_len = 3, threshold = 0.2, max_nb_states = 120):
    '''
    compute the product of the integrals over Ri as previousily described
    work in log space to avoid overflow and underflow
    
    Cs : dim 0 = track ID, dim 1 : states, dim 2 : peaks postions through time,
    dim 3 : x, y position
    
    we process by steps, at each step we account for 1 more localization, we compute
    the canstant (LC), the mean (m_arr) and std (Ks) of of the normal distribution 
    resulting from integration.
    
    each step is made of substeps if nb_substeps > 1, and we increase the matrix
    of possible Bs : cur_Bs accordingly
    
    to be able to process long tracks with good accuracy, for each track we fuse m_arr and Ks
    of sequences of states equal exept for the state 'frame_len' steps ago.
    '''
    nb_Tracks = Cs.shape[0]
    nb_locs = Cs.shape[1] # number of localization per track
    
    dtype = 'float64'
    
    nb_dims = Cs.shape[2] # number of spatial dimensions (x, y) or (x, y, z)
    Cs = Cs[:,None].astype('float64')
    Cs = cp.array(Cs)
    nb_states = TrMat.shape[0]
    Cs = Cs[:,:,::-1] # I built the model going from the last index to the first index of the reversed positions. Which is equivalent to an iteration from the first position to the last one. 
    LocErr = LocErr[:,None].astype(dtype)
    LocErr = LocErr[:,:,::-1] # useful when single peak localization error is inputed,
    LocErr2 = LocErr**2
    if LocErr.shape[2] == 1: # combined to min(LocErr_index, nb_locs-current_step) it will select the right index
        LocErr_index = -1
    elif LocErr.shape[2] == nb_locs:
        LocErr_index = nb_locs
    else:
        raise ValueError("Localization error is not specified correctly, in case of unique localization error specify a float number in estimated_vals['LocErr'].\n If one localization error per dimension, specify a list or 1D array of elements the localization error for each dimension.\n If localization error is predetermined by another method for each position the argument input_LocErr should be a dict for each track length of the 3D arrays corresponding to all_tracks (can be obtained from the reader functions using the opt_colname argument)")
    if do_preds:
        preds = np.zeros((nb_Tracks, nb_locs, nb_states))-1
    else :
        preds = []
    
    if nb_locs < 2:
        raise ValueError('minimal track length = 2, here track length = %s'%nb_locs)
    
    t0 = time()
    
    sub_Bs = get_all_Bs(nb_substeps, nb_states)[None]
    TrMat = cp.array(TrMat.T)
    current_step = 1
    
    cur_Bs = get_all_Bs(nb_substeps + 1, nb_states)[None] # get initial sequences of states
    cur_Bs_cat = (cur_Bs[:,:,:,None] == np.arange(nb_states)[None,None,None,:]).astype('float64')
    
    cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int) #states of interest for the current displacement
    cur_nb_Bs = cur_Bs.shape[1]
    # compute the vector of diffusion stds knowing the current states
    ds = cp.array(ds).astype(dtype)
    Fs = cp.array(Fs).astype(dtype)
    
    LT = get_Ts_from_Bs(cur_states, TrMat) # Log proba of transitions per step
    LF = cp.log(Fs[cur_states[:,:,-1]]) # Log proba of finishing/starting in a given state (fractions)
        
    LP = LT + LF #+ compensate_leaving
    # current log proba of seeing the track
    LP = cp.repeat(LP, nb_Tracks, axis = 0)
    
    if len(ds.shape) == 1:
        cur_d2s = ds[cur_states]**2
    elif len(ds.shape) == 3:
        cur_d2s = ds[:, -1, cur_states[0]]**2
    else:
        raise ValueError('dt is not informed properly. It must either be a float number or a dictionary of same structure than `all_tracks` with each element being an array of dims (nb_tracks, track_len)')
        #cur_d2s.shape (1,4,1)
    cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps

    # we can average the variances of displacements per step to get the actual std of displacements per step
    cur_d2s = cp.mean(cur_d2s, axis = 2)
    cur_d2s = cur_d2s[:,:,None]
    cur_d2s = cp.array(cur_d2s)
    
    sub_Bs = cur_Bs.copy()[:,:cur_Bs.shape[1]//nb_states,:nb_substeps] # list of possible current states we can meet to compute the proba of staying in the FOV
    if len(ds.shape) == 1:
        simplified_ds = ds
    elif len(ds.shape) == 3:
        simplified_ds = np.median(ds[:, 0], axis = 0)
    else:
        raise ValueError(ds.shape)
    sub_ds = (cp.mean(simplified_ds[sub_Bs]**2, axis = 2)**0.5).astype(float) # corresponding list of d
    sub_ds = asnumpy(sub_ds)
    
    p_stay = np.ones(sub_ds.shape[-1])
    for cell_len in cell_dims:
        xs = np.linspace(0+cell_len/2000,cell_len-cell_len/2000,1000)
        cur_p_stay = ((cp.mean(scipy.stats.norm.cdf((cell_len-xs[:,None])/(sub_ds+1e-200)) - scipy.stats.norm.cdf(-xs[:,None]/(sub_ds+1e-200)),0))) # proba to stay in the FOV for each of the possible cur Bs
        p_stay = p_stay*cur_p_stay
    p_stay = cp.array(p_stay)
    Lp_stay = cp.log(p_stay * (1-pBL)) # proba for the track to survive = both stay in the FOV and not bleach
    
    # inject the first position to get the associated m_arr and Ks :
    m_arr, s2_arr = first_log_integrale_dif(Cs[:,:, nb_locs-current_step], LocErr2[:,:, min(LocErr_index, nb_locs-current_step)], cur_d2s)
    
    m_arr = cp.repeat(m_arr, cur_nb_Bs, axis = 1)
    removed_steps = 0
    
    if nb_substeps > 1 and 0:
        cur_len = nb_substeps + 1
        fuse_pos = np.arange(1,nb_substeps)
        m_arr, s2_arr, LP, cur_Bs = fuse_tracks_general(m_arr, s2_arr, LP, cur_Bs, cur_len, nb_Tracks, fuse_pos = fuse_pos, nb_states = nb_states, nb_dims = nb_dims)
    
    current_step += 1
    
    while current_step <= nb_locs-1:
        for iii in range(nb_substeps):
            #cur_Bs = np.concatenate((np.repeat(np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], nb_Tracks, 0), np.repeat(cur_Bs,nb_states,1)),-1)
            cur_Bs = np.concatenate((np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], np.repeat(cur_Bs,nb_states,1)),-1)
            new_states = np.repeat(np.mod(np.arange(cur_Bs_cat.shape[1]*nb_states, dtype = 'int8'),nb_states)[None,:,None,None] == np.arange(nb_states, dtype = 'int8')[None,None,None], cur_Bs_cat.shape[0], 0).astype('int8')
            cur_Bs_cat = np.concatenate((new_states, np.repeat(cur_Bs_cat,nb_states,1)),-2)
        
        cur_states = cur_Bs[:1,:,0:nb_substeps+1].astype(int)
        # compute the vector of diffusion stds knowing the states at the current step
        if len(ds.shape) == 1:
            cur_d2s = ds[cur_states]**2
        elif len(ds.shape) == 3:
            cur_d2s = ds[:, nb_locs-current_step, cur_states[0]]**2
        cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps
        cur_d2s = cp.mean(cur_d2s, axis = 2)
        cur_d2s = cur_d2s[:,:,None]
        LT = get_Ts_from_Bs(cur_states, TrMat)
        
        if current_step >= min_len:
            LL = Lp_stay[np.argmax(np.all(cur_states[:,None,:,:-1] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states
        else:
            LL = 0

        # replicate over the new states, inject the next position and accumulate
        m_arr, s2_arr, LP = recurrence_step(Cs[:,:,nb_locs-current_step],
                                            LocErr2[:,:,min(LocErr_index,nb_locs-current_step)],
                                            cur_d2s, m_arr, s2_arr, LP, LT, LL,
                                            nb_states**nb_substeps)
        del LT
        
        if nb_substeps > 1 and 0:
            cur_len = cur_Bs.shape[-1]
            fuse_pos = np.arange(1,nb_substeps)+nb_substeps
            m_arr, s2_arr, LP, cur_Bs = fuse_tracks_general(m_arr, s2_arr, LP, cur_Bs, cur_len, nb_Tracks, fuse_pos, nb_states = nb_states, nb_dims = nb_dims)

        cur_nb_Bs = len(cur_Bs[0]) # current number of sequences of states
        #print(current_step, m_arr.shape)
        
        if cur_nb_Bs>max_nb_states:
            threshold = threshold*1.2
            #print('threshold', threshold)
            
        
        '''idea : the position and the state 6 steps ago should not impact too much the 
        probability of the next position so the m_arr and s2_arr of tracks with the same 6 last 
        states must be very similar, we can then fuse the parameters of the pairs of Bs
        which vary only for the last step (7) and sum their probas'''
        
        if current_step < nb_locs-1: # do not fuse sequences at the last step as it doesn't improves speed.            
            m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat = fuse_tracks_th(m_arr,
                                                                   s2_arr,
                                                                   LP,
                                                                   cur_Bs,
                                                                   cur_Bs_cat,
                                                                   nb_Tracks,
                                                                   nb_states = nb_states,
                                                                   nb_dims = nb_dims,
                                                                   do_preds = do_preds,
                                                                   threshold = threshold,
                                                                   frame_len = frame_len) # threshold on values normalized by sigma.
            # threshold on values normalized by sigma.
            cur_nb_Bs = len(cur_Bs[0])
            #print(current_step, m_arr.shape)
            removed_steps += 1
            
        current_step += 1
    
    m_arr.shape
    s2_arr.shape
    
    if not isBL:
        LL = 0
    else:
        for iii in range(nb_substeps):
            #cur_Bs = np.concatenate((np.repeat(np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], nb_Tracks, 0), np.repeat(cur_Bs,nb_states,1)),-1)
            cur_Bs = np.concatenate((np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], np.repeat(cur_Bs,nb_states,1)),-1)
            new_states = np.repeat(np.mod(np.arange(cur_Bs_cat.shape[1]*nb_states, dtype = 'int8'),nb_states)[None,:,None,None] == np.arange(nb_states, dtype = 'int8')[None,None,None], cur_Bs_cat.shape[0], 0).astype('int8')
            cur_Bs_cat = np.concatenate((new_states, np.repeat(cur_Bs_cat,nb_states,1)),-2)
                
        cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int)
        len(cur_Bs[0])
        LT = get_Ts_from_Bs(cur_states, TrMat)
        # repeat the previous matrix to account for the states variations due to the new position
        m_arr = cp.repeat(m_arr, nb_states**nb_substeps , axis = 1)
        s2_arr = cp.repeat(s2_arr, nb_states**nb_substeps, axis = 1)
        LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)
        
        end_p_stay = p_stay[cur_states[:,None:,:-1]][:,:,0]
        LL = cp.log(pBL + (1-end_p_stay) - pBL * (1-end_p_stay)) + LT
        cur_Bs_cat = cur_Bs_cat[:,:,1:]

    LP = final_integration(Cs[:,:,0], LocErr2[:,:, min(LocErr_index, nb_locs-current_step)],
                           m_arr, s2_arr, LP, LL)

    pred_LP = LP
    if np.max(LP)>600: # avoid overflow of exponentials, (drawback: mechanically also reduces the weights of longest tracks)
        pred_LP = LP - (np.max(LP)-600)
    
    P = np.exp(pred_LP)
    sum_P = np.sum(P, axis = 1, keepdims = True)[:,:,None]
    if do_preds :
        preds = np.sum(P[:,:,None,None]*cur_Bs_cat, axis = 1) / sum_P
        preds = preds[:,::-1]
    return LP, cur_Bs_cat, preds

def fuse_tracks_th_numba(m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat, nb_Tracks, nb_states = 2, nb_dims = 2, do_preds = 1, threshold = 0.2, frame_len = 6):
    '''
    The numba twin of fuse_tracks_th: same grouping rule, same moment matched
    fusion, same outputs. The grouping is O(nb_branches**2) and the fusion runs
    over every track, which is why both are worth compiling.
    '''
    test_chunks = 30
    m_arr = np.ascontiguousarray(m_arr, dtype = float)
    s2_arr = np.ascontiguousarray(s2_arr, dtype = float)
    LP = np.ascontiguousarray(LP, dtype = float)
    cur_Bs_cat = np.ascontiguousarray(cur_Bs_cat, dtype = float)

    hist_len = cur_Bs_cat.shape[2]
    nF = min(frame_len, hist_len)
    tc_m = min(test_chunks, m_arr.shape[0])
    tc_s = min(test_chunks, s2_arr.shape[0])
    tc_c = min(test_chunks, cur_Bs_cat.shape[0])

    s_arr = s2_arr**0.5
    cat_arg = np.ascontiguousarray(np.argmax(cur_Bs_cat[:tc_c, :, :nF], -1).astype(np.int64))
    state0 = np.ascontiguousarray(np.argmax(cur_Bs_cat[0, :, 0], 1).astype(np.int64))

    group_of, nb_subgroups = _numba_kernels.group_kernel(
        np.ascontiguousarray(m_arr[:tc_m]), np.ascontiguousarray(s_arr[:tc_s]),
        cat_arg, state0, hist_len, frame_len, threshold)
    if np.any(group_of < 0):
        raise ValueError('problem with grouping: some branches were left out')

    member = np.ascontiguousarray(np.argsort(group_of, kind = 'stable').astype(np.int64))
    gptr = np.zeros(nb_subgroups + 1, dtype = np.int64)
    gptr[1:] = np.cumsum(np.bincount(group_of, minlength = nb_subgroups))

    new_m_arr = np.zeros((nb_Tracks, nb_subgroups, m_arr.shape[2]))
    new_s2_arr = np.zeros((nb_Tracks, nb_subgroups, fused_variance_shape(m_arr, s2_arr)))
    new_LP = np.zeros((nb_Tracks, nb_subgroups))
    _numba_kernels.fuse_kernel(m_arr, s2_arr, LP, member, gptr,
                               FUSION_MOMENT_MATCHING, FUSION_ISOTROPIC_VARIANCE,
                               new_m_arr, new_s2_arr, new_LP)

    if not do_preds:
        cur_Bs_cat = np.ascontiguousarray(cur_Bs_cat[:, :, :frame_len])
    nb_cat_rows = nb_Tracks if do_preds else 1   # see the note in fuse_tracks_th
    new_cur_Bs_cat = np.zeros((nb_cat_rows, nb_subgroups, cur_Bs_cat.shape[2], nb_states))
    if do_preds:
        _numba_kernels.fuse_cat_kernel(cur_Bs_cat, LP, member, gptr, new_cur_Bs_cat)
    else:
        _numba_kernels.mean_cat_kernel(cur_Bs_cat, member, gptr,
                                       min(test_chunks, cur_Bs_cat.shape[0]), new_cur_Bs_cat)

    new_cur_Bs = np.empty((1, nb_subgroups, 1), dtype = int)
    for g in range(nb_subgroups):
        new_cur_Bs[0, g, 0] = cur_Bs[0, member[gptr[g]], 0]

    return new_m_arr, new_s2_arr, new_LP, new_cur_Bs, new_cur_Bs_cat

def fuse_tracks_th(m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat, nb_Tracks, nb_states = 2, nb_dims = 2, do_preds = 1, threshold = 0.2, frame_len = 6):
    '''
    The probabilities of the fused branches are summed and their Gaussians are
    reduced to a single Gaussian carrying the 2 first moments of their mixture
    (see fuse_gaussians).
    As I must divid by a sum of exponentials which can be equal to zero because of underflow
    I correct the values in the exponetial to keep the maximal exp value at 0
    '''
    if numba_active():
        return fuse_tracks_th_numba(m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat, nb_Tracks,
                                    nb_states, nb_dims, do_preds, threshold, frame_len)

    # cut the matrixes so the resulting matrices only vary for their last state
    s_arr = s2_arr**0.5
    
    groups = []
    grouped_IDs = []
    
    for Bs_ID in range(m_arr.shape[1]):
        if not np.isin(Bs_ID, grouped_IDs):
            cur_m_arr = m_arr[:,Bs_ID]
            cur_s_arr = s_arr[:,Bs_ID]
            
            cur_cur_Bs_cat = cur_Bs_cat[:,Bs_ID]
            current_state = np.argmax(cur_cur_Bs_cat[0,0])
            cur_state_mask = np.argmax(cur_Bs_cat[0, :,0],1) == current_state
            
            test_chunks = np.min([np.max([10, int(nb_Tracks**0.4)]), 50])
            test_chunks = 30
            
            if cur_cur_Bs_cat.shape[1] > frame_len:
                #state_mask = np.mean(np.all((cur_cur_Bs_cat[:test_chunks,None,:frame_len] == cur_Bs_cat[:test_chunks,:,:frame_len]), (2, 3)), 0) > 0.999
                state_mask = np.mean(np.all(np.argmax(cur_cur_Bs_cat[:test_chunks,None,:frame_len], -1) == np.argmax(cur_Bs_cat[:test_chunks,:,:frame_len], -1), (2)), 0) > 0.999
            #cur_Bs_cat[:,np.where(state_mask)[0]]
            
            #(np.argmax(cur_cur_Bs_cat[:test_chunks,None,:frame_len], -1) == np.argmax(cur_Bs_cat[:test_chunks,:,:frame_len], -1)).shape
            
            else:
                state_mask = False
            
            m_mask_relative = np.mean((np.mean(np.abs(m_arr[:test_chunks] - cur_m_arr[:test_chunks,None]), 2, keepdims = True)/s_arr[:test_chunks]) < threshold, (0, 2)) > 0.8
            #s_mask_relative = ((cur_s_arr[:,None] > (1 - threshold) * s_arr)*(cur_s_arr[:,None] < (1 + threshold) * s_arr))[0,:,0]
            s_mask_relative = np.mean((np.mean(np.abs(s_arr[:test_chunks] - cur_s_arr[:test_chunks,None]), 2, keepdims = True)/s_arr[:test_chunks]) < threshold, (0, 2)) > 0.8
            
            args = np.where(m_mask_relative * s_mask_relative * cur_state_mask + state_mask)[0]

            args = args[np.isin(args, grouped_IDs) == False] # remove elements that already belongs to a group
            if not np.isin(Bs_ID, args): # a branch always belongs to its own
                # group: the tests above are strict, so at threshold = 0 -- the
                # way to ask for no fusion -- it would fail its own test
                args = np.sort(np.append(args, Bs_ID))
            
            groups.append(args)
            grouped_IDs = grouped_IDs + list(args)
    
    if len(grouped_IDs) != m_arr.shape[1]:
        raise ValueError('problem with grouping: len(grouped_IDs)=' + str(len(grouped_IDs)) + ' and m_arr.shape[1]=' + str( m_arr.shape[1]))

    #subgroups = []
    #for group in groups:
    #    for state in range(nb_states):
    #        subgroup = group[cur_Bs[:, group][0,:,0] == state]
    #        if len(subgroup)>0:
    #            subgroups.append(subgroup)
    
    # part that is too time consuming, find a way to make it faster
    subgroups = groups
    nb_subgroups = len(subgroups)
    new_cur_Bs = np.empty((1, nb_subgroups, 1), dtype=(int))
    if not do_preds:
        cur_Bs_cat = cur_Bs_cat[:,:,:frame_len]

    # when the state predictions are not asked for, the histories only serve to
    # decide which branches share a sequence of states and every track ends up
    # with the same ones, so a single row is kept instead of nb_Tracks copies of
    # it (that array is (nb_Tracks, nb_branches, frame_len, nb_states) and was
    # being rebuilt at every step)
    nb_cat_rows = nb_Tracks if do_preds else 1
    new_cur_Bs_cat = np.zeros((nb_cat_rows, nb_subgroups, cur_Bs_cat.shape[2], nb_states), dtype = cur_Bs_cat.dtype)
    
    new_m_arr = np.zeros((nb_Tracks, nb_subgroups, m_arr.shape[2]),  dtype = m_arr.dtype)
    new_s2_arr = np.zeros((nb_Tracks, nb_subgroups, fused_variance_shape(m_arr, s2_arr)),  dtype = s2_arr.dtype)
    new_LP = np.zeros((nb_Tracks, nb_subgroups),  dtype = LP.dtype)
    
    for Bs_ID, subgroup in enumerate(subgroups):
        
        max_LP = LP[:, subgroup].max(axis = 1, keepdims = True)
        weights = np.exp(LP[:, subgroup] - max_LP)
        sum_weights = np.sum(weights, 1, keepdims = True)
        new_cur_Bs[:, Bs_ID] = cur_Bs[:, subgroup[:1], 0]
        
        if len(subgroup)>1:
            #sum_masks = np.zeros((nb_Tracks, cur_Bs_cat.shape[2], nb_states), dtype= bool)
            if do_preds:
                new_cur_Bs_cat[:, Bs_ID] = (np.sum(weights[:,:,None,None] * cur_Bs_cat[:, subgroup, :], 1) / sum_weights[:,:,None])
            else:
                new_cur_Bs_cat[:, Bs_ID] = np.mean(cur_Bs_cat[:test_chunks, subgroup, :], (0,1))[None] # if not do_preds (for the fitting module) the computed values are only useful to know which tracks to fuse because they are sharing the same sequences of states for frame_len time points.
        else:
            new_cur_Bs_cat[:, Bs_ID] =  cur_Bs_cat[:, subgroup[0]]    
        
        # approximate the mixture of Gaussians of the group by a single Gaussian with the same 2 first moments
        new_m_arr[:, Bs_ID], new_s2_arr[:, Bs_ID], new_LP[:, Bs_ID] = fuse_gaussians(m_arr[:, subgroup, :], s2_arr[:, subgroup, :], LP[:, subgroup], 1)
        
    return new_m_arr, new_s2_arr, new_LP, new_cur_Bs, new_cur_Bs_cat


"""
The (segment age, current state) buffer, transposed from ExaTrack.

The scheme above carries one hypothesis per *sequence of states* over the last
`frame_len` frames: nb_states**frame_len of them, brought back down at every step
by fuse_tracks_th, whose grouping is a python loop of cost O(nb_Bs**2) and whose
outcome depends on the data. ExaTrack keeps instead a buffer of fixed size
`frame_len * nb_states`, indexed by

    (a, s) = (number of steps since the last transition, current state)

the oldest slab meaning "a >= frame_len - 1". One step generates exactly
`frame_len * nb_states**2` branches (every hypothesis times every next state) and
folds them straight back into `frame_len * nb_states` hypotheses:

    (a, s) --> (a+1, s)   staying: the age advances and nothing is fused, except
                          in the oldest slab where a+1 is capped and the two
                          oldest ages merge,
    (a, s) --> (0, j)     transitioning: every source arriving in state j is fused
                          into the single newborn (0, j).

Why (age, state) rather than the sequence of states: what the carried Gaussian
depends on is how much diffusion has accumulated since the particle last changed
state, and that is exactly (a, s). What is given up is the identity of the states
before the current segment, and the exact age once it exceeds frame_len - 1.

Per step and per track, against the scheme above:

    hypotheses  frame_len * nb_states       instead of  nb_states ** frame_len
    branches    frame_len * nb_states**2    instead of  nb_states ** (frame_len+1)
    grouping    none, the layout is static  instead of  a python loop over pairs

so the cost no longer depends on the data and the duration of a fit becomes
predictable. Every fusion here is the moment matched one (fuse_gaussians), which
matters more than in the scheme above because the newborn fusion pools
hypotheses whose means are genuinely far apart.
"""

def kalman_fold(Ci, l2, m_arr, s2_arr):
    '''
    Fold one observation into the carried Gaussian, one scalar dimension at a time.

    The carried message is the predictive N(r ; m_arr, s2_arr) of the current true
    position. Observing Ci with variance l2 contributes the evidence
    N(Ci ; m_arr, s2_arr + l2) and leaves the posterior N(r ; m_post, s2_post) :

        tot     = s2_arr + l2
        K       = sum_dims [ -0.5*log(2 pi tot) - (Ci - m_arr)**2 / (2 tot) ]
        m_post  = (m_arr*l2 + Ci*s2_arr) / tot
        s2_post = s2_arr*l2 / tot

    This is log_integrale_dif split in two: it stops before adding the diffusion
    variance of the coming step, because that part depends on the next state while
    this part does not.
    '''
    tot = s2_arr + l2
    K = cp.sum(-0.5*cp.log(2*np.pi*tot) - (Ci - m_arr)**2 / (2*tot), axis = -1)
    m_post = (m_arr*l2 + Ci*s2_arr) / tot
    s2_post = s2_arr*l2 / tot
    return m_post, s2_post, K

def fuse_with_history(m_arr, s2_arr, LP, hist, axis):
    '''
    fuse_gaussians, plus the same weighted average of the state histories that the
    state predictions are read from.
    '''
    new_m, new_s2, new_LP = fuse_gaussians(m_arr, s2_arr, LP, axis)
    if hist is None:
        return new_m, new_s2, new_LP, None
    max_LP = cp.max(LP, axis = axis, keepdims = True)
    weights = cp.exp(LP - max_LP)
    weights = weights / cp.sum(weights, axis = axis, keepdims = True)
    return new_m, new_s2, new_LP, cp.sum(weights[..., None, None] * hist, axis = axis)

def new_ages_carry(nb_Tracks, frame_len, nb_states, nb_dims, hist_len, want_cat):
    '''
    Fresh carry buffers for the (age, state) recursion: the message a segment
    hands to the next one. `nact` is the number of age slabs opened so far.
    '''
    cap = max(int(frame_len), 2) * nb_states
    return dict(LP = np.zeros((nb_Tracks, cap)),
                m = np.zeros((nb_Tracks, cap * nb_dims)),
                s2 = np.zeros((nb_Tracks, cap * nb_dims)),
                cat = np.zeros((nb_Tracks, cap * (hist_len if want_cat else 1) * nb_states)),
                nact = np.ones(nb_Tracks, dtype = np.int64))

def ages_kernel_call(Cs, LocErr2, log_TrMat, log_Fs, pair_d2, Lp_stay, end_LL,
                     frame_len, min_len, nsteps, isfirst, islast, abs_start,
                     carry, hist_len, want_cat, nb_states):
    '''
    Run one segment of the (age, state) recursion through the numba kernel.

    `carry` is the dict from new_ages_carry, or None for a self-contained call;
    it is updated in place for the tracks that do not end in this segment.
    Returns (out_LP, out_cat); out_LP is only meaningful where islast != 0.
    '''
    Cs = np.ascontiguousarray(Cs, dtype = float)
    nb_Tracks, nb_dims = Cs.shape[0], Cs.shape[2]
    if carry is None:
        carry = new_ages_carry(nb_Tracks, frame_len, nb_states, nb_dims, hist_len, want_cat)
    out_LP = np.zeros(nb_Tracks)
    out_cat = np.zeros((nb_Tracks, hist_len if want_cat else 1, nb_states))
    _numba_kernels.ages_kernel(
        Cs,
        np.ascontiguousarray(LocErr2, dtype = float),
        np.ascontiguousarray(log_TrMat, dtype = float),
        np.ascontiguousarray(log_Fs, dtype = float),
        np.ascontiguousarray(pair_d2, dtype = float),
        np.ascontiguousarray(Lp_stay, dtype = float),
        np.ascontiguousarray(end_LL, dtype = float),
        max(int(frame_len), 2), int(min_len),
        FUSION_MOMENT_MATCHING, FUSION_ISOTROPIC_VARIANCE, bool(want_cat),
        np.ascontiguousarray(nsteps, dtype = np.int64),
        np.ascontiguousarray(isfirst, dtype = np.int64),
        np.ascontiguousarray(islast, dtype = np.int64),
        np.ascontiguousarray(abs_start, dtype = np.int64),
        carry['LP'], carry['m'], carry['s2'], carry['cat'], carry['nact'],
        out_LP, out_cat)
    return out_LP, out_cat

def P_Cs_inter_bound_stats_ages(Cs, LocErr, ds, Fs, TrMat, pBL = 0.1, isBL = 1, cell_dims = [0.5], nb_substeps = 1, frame_len = 6, do_preds = 0, min_len = 3, threshold = 0.2, max_nb_states = 120):
    '''
    Same model and same likelihood as P_Cs_inter_bound_stats_th, with the tree of
    sequences of states replaced by the (segment age, current state) buffer
    described above.

    `threshold` and `max_nb_states` are accepted and ignored: this scheme has no
    adaptive grouping, its buffer always holds frame_len*nb_states hypotheses.
    `nb_substeps` must be 1 and `ds` must be a single diffusion length per state
    (a time and track dependent `ds` is not supported yet).

    Cs : dim 0 = track ID, dim 1 : peak positions through time, dim 2 : x, y (z)
    '''
    if nb_substeps != 1:
        raise NotImplementedError('the (age, state) scheme only supports nb_substeps = 1')

    nb_Tracks, nb_locs, nb_dims = Cs.shape
    nb_states = TrMat.shape[0]
    if nb_states < 2:
        raise ValueError('the (age, state) scheme needs at least 2 states')
    L = max(int(frame_len), 2)
    if nb_locs < 2:
        raise ValueError('minimal track length = 2, here track length = %s'%nb_locs)

    Cs = cp.array(np.asarray(Cs).astype('float64'))
    LocErr2 = cp.array(np.asarray(LocErr).astype('float64'))**2   # (nb_tracks|1, nb_locs|1, nb_dims|1)
    single_LocErr = LocErr2.shape[1] == 1
    if not single_LocErr and LocErr2.shape[1] != nb_locs:
        raise ValueError("Localization error is not specified correctly, see P_Cs_inter_bound_stats_th")

    ds = np.asarray(ds)
    if ds.ndim != 1:
        raise NotImplementedError('the (age, state) scheme only supports one diffusion length per state')
    d2s = cp.array(ds**2)[None, None]                             # (1, 1, nb_states)
    Fs = cp.array(Fs)
    log_TrMat = cp.log(cp.array(TrMat))                           # [current state, next state]

    # probability to stay in the field of view and not to bleach, one per state
    p_stay = np.ones(nb_states)
    for cell_len in cell_dims:
        xs = np.linspace(0+cell_len/2000, cell_len-cell_len/2000, 1000)
        p_stay = p_stay * np.mean(scipy.stats.norm.cdf((cell_len-xs[:,None])/(ds+1e-200)) - scipy.stats.norm.cdf(-xs[:,None]/(ds+1e-200)), 0)
    p_stay = cp.array(p_stay)
    Lp_stay = cp.log(p_stay * (1-pBL))

    stay = np.arange(nb_states)
    sources = np.array([[s for s in range(nb_states) if s != j] for j in range(nb_states)]) # (nb_states, nb_states-1)
    arrival = np.arange(nb_states)[:, None]
    pair_d2 = (d2s[:,:,:,None] + d2s[:,:,None,:]) / 2             # (1, 1, from, to), transition in the middle of the step

    def l2_at(t):
        return LocErr2[:, 0 if single_LocErr else t][:, None]     # (nb_tracks|1, 1, nb_dims|1)

    def advance(m_arr, s2_arr, LP, hist, K, LL, n_act, new_state_time):
        '''
        One transition of the buffer: branch every hypothesis over the nb_states
        possible next states, then fold the branches back onto their (age, state)
        target. K is the evidence of the observation folded at this step, None for
        the very first transition which folds none.
        '''
        S = nb_states
        T = LP.shape[0]
        nd = s2_arr.shape[-1]

        LPb = LP.reshape(T, n_act, S)[:,:,:,None] + log_TrMat[None,None] + LL
        if K is not None:
            LPb = LPb + K.reshape(T, n_act, S)[:,:,:,None]
        s2b = s2_arr.reshape(T, n_act, S, 1, nd) + pair_d2[..., None]
        mb = cp.broadcast_to(m_arr.reshape(T, n_act, S, 1, nd), (T, n_act, S, S, nd))
        histb = hist.reshape((T, n_act, S) + hist.shape[2:]) if hist is not None else None

        n_new = min(n_act + 1, L)
        new_LP = cp.empty((T, n_new, S))
        new_m = cp.empty((T, n_new, S, nd))
        new_s2 = cp.empty((T, n_new, S, nd))
        new_hist = cp.empty((T, n_new, S) + hist.shape[2:]) if hist is not None else None

        # newborns : every source arriving in state j is fused into (age 0, j)
        take = (slice(None), slice(None), sources, arrival)
        f_LP = LPb[take].transpose(0,2,1,3).reshape(T, S, n_act*(S-1))
        f_m = mb[take].transpose(0,2,1,3,4).reshape(T, S, n_act*(S-1), nd)
        f_s2 = s2b[take].transpose(0,2,1,3,4).reshape(T, S, n_act*(S-1), nd)
        f_h = None
        if hist is not None:
            f_h = histb[:,:,sources].transpose(0,2,1,3,4,5).reshape((T, S, n_act*(S-1)) + hist.shape[2:])
        new_m[:,0], new_s2[:,0], new_LP[:,0], fused_h = fuse_with_history(f_m, f_s2, f_LP, f_h, 2)
        if hist is not None:
            new_hist[:,0] = fused_h

        # stays : the age advances; when the buffer is full the 2 oldest slabs merge
        s_LP = LPb[:,:,stay,stay]
        s_m = mb[:,:,stay,stay]
        s_s2 = s2b[:,:,stay,stay]
        if n_act < L:
            new_LP[:,1:], new_m[:,1:], new_s2[:,1:] = s_LP, s_m, s_s2
            if hist is not None:
                new_hist[:,1:] = histb
        else:
            new_LP[:,1:L-1], new_m[:,1:L-1], new_s2[:,1:L-1] = s_LP[:,:L-2], s_m[:,:L-2], s_s2[:,:L-2]
            if hist is not None:
                new_hist[:,1:L-1] = histb[:,:L-2]
            old = slice(L-2, L)
            new_m[:,L-1], new_s2[:,L-1], new_LP[:,L-1], fused_h = fuse_with_history(
                s_m[:,old], s_s2[:,old], s_LP[:,old],
                histb[:,old] if hist is not None else None, 1)
            if hist is not None:
                new_hist[:,L-1] = fused_h

        if hist is not None: # the state reached at this step is known for every target
            new_hist[:,:,:,new_state_time] = 0
            for s in range(S):
                new_hist[:,:,s,new_state_time,s] = 1

        return (new_m.reshape(T, n_new*S, nd), new_s2.reshape(T, n_new*S, nd),
                new_LP.reshape(T, n_new*S),
                new_hist.reshape((T, n_new*S) + hist.shape[2:]) if hist is not None else None,
                n_new)

    if numba_active():
        # the whole recursion is one kernel call: its state per track is a few tens
        # of doubles, so it stays in L1 and no temporary is ever written out
        end_LL = np.log(pBL + (1-p_stay) - pBL * (1-p_stay))
        # the whole data set as a single segment: every track starts here
        # (isfirst = 1) and ends here (islast = 2 when the track stops, 1 when it
        # runs to the end of the movie), so the carry buffers stay unused
        nsteps = np.full(nb_Tracks, nb_locs - 1, dtype = np.int64)
        out_LP, out_cat = ages_kernel_call(
            asnumpy(Cs), asnumpy(LocErr2), asnumpy(log_TrMat), asnumpy(cp.log(Fs)),
            asnumpy(pair_d2[0, 0]), asnumpy(Lp_stay), asnumpy(end_LL), L, min_len,
            nsteps, np.ones(nb_Tracks, dtype = np.int64),
            np.full(nb_Tracks, 2 if isBL else 1, dtype = np.int64),
            np.zeros(nb_Tracks, dtype = np.int64),
            None, nb_locs if do_preds else 1, bool(do_preds), nb_states)
        return out_LP[:, None], None, (out_cat if do_preds else [])

    # time 0 : one hypothesis per state, carrying the posterior of r_0 given c_0
    n_act = 1
    m_arr = cp.repeat(Cs[:,0][:,None], nb_states, 1)
    s2_arr = cp.zeros((nb_Tracks, nb_states, nb_dims)) + l2_at(0)
    LP = cp.repeat(cp.log(Fs)[None], nb_Tracks, 0)
    hist = None
    if do_preds:
        hist = cp.zeros((nb_Tracks, nb_states, nb_locs, nb_states))
        for s in range(nb_states):
            hist[:, s, 0, s] = 1

    # first transition : no observation is folded, it only opens the age 0 and 1 slabs
    m_arr, s2_arr, LP, hist, n_act = advance(m_arr, s2_arr, LP, hist, None, 0., n_act, 1)

    # times 1 .. nb_locs-2 : fold the observation, then transition
    for tau in range(1, nb_locs-1):
        m_arr, s2_arr, K = kalman_fold(Cs[:,tau][:,None], l2_at(tau), m_arr, s2_arr)
        LL = Lp_stay[None,None,None] if tau + 1 >= min_len else 0.
        m_arr, s2_arr, LP, hist, n_act = advance(m_arr, s2_arr, LP, hist, K, LL, n_act, tau+1)

    # last observation, then the end of track term
    LP = LP + kalman_fold(Cs[:,nb_locs-1][:,None], l2_at(nb_locs-1), m_arr, s2_arr)[2]
    if isBL:
        end_LL = cp.log(pBL + (1-p_stay) - pBL * (1-p_stay))[None]     # indexed by the state reached
        cur_states = np.tile(np.arange(nb_states), n_act)              # component index = age*nb_states + state
        LP = LP + cp.log(cp.sum(cp.exp(log_TrMat[cur_states] + end_LL), axis = -1))[None]

    preds = []
    if do_preds:
        weights = cp.exp(LP - cp.max(LP, axis = 1, keepdims = True))
        weights = weights / cp.sum(weights, axis = 1, keepdims = True)
        preds = asnumpy(cp.sum(weights[:,:,None,None] * hist, axis = 1))
    return LP, hist, preds

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
    LT = cp.zeros((all_Bs.shape[:2]), dtype = float)
    # change from binary base 10 numbers to identify the consecutive states (from ternary if 3 states) 
    for k in range(len(all_Bs[0,0])-1):
        LT += cp.log(TrMat[all_Bs[:,:,k], all_Bs[:,:,k+1]])
    return LT

def Proba_Cs(Cs, LocErr, ds, Fs, TrMat, pBL, isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, sequence_scheme = 'sequences'):
    '''
    inputs the observed localizations and determine the probability of 
    observing these data knowing the localization error, D the diffusion coef,
    pu the proba of unbinding per step and pb the proba of binding per step
    sum the proba of Cs inter Bs (calculated with P_Cs_inter_bound_stats)
    over all Bs to get the proba of Cs (knowing the initial position c0)
    '''
    
    kernel = get_sequence_kernel(sequence_scheme)
    LP_CB, _, _  = kernel(Cs, LocErr, ds, Fs, TrMat, pBL,isBL,cell_dims, nb_substeps, frame_len, do_preds = 0, min_len = min_len, threshold = threshold, max_nb_states = max_nb_states)
    np.sum(LP_CB)
    # calculates P(C) the sum of P(C inter B) for each track
    max_LP = np.max(LP_CB, axis = 1, keepdims = True)
    LP_CB = LP_CB - max_LP
    max_LP = max_LP[:,0]
    P_CB = np.exp(LP_CB)
    P_C = cp.sum(P_CB, axis = 1) # sum over B
    LP_C = np.log(P_C) + max_LP # back to log proba of C without overflow due to exponential
    return LP_C

def get_sequence_kernel(sequence_scheme):
    '''
    'sequences' : one hypothesis per sequence of states over the last frame_len
                  frames, adaptively grouped (P_Cs_inter_bound_stats_th).
    'ages'      : one hypothesis per (segment age, current state), a fixed buffer
                  of frame_len*nb_states (P_Cs_inter_bound_stats_ages).
    '''
    if sequence_scheme == 'sequences':
        return P_Cs_inter_bound_stats_th
    elif sequence_scheme == 'ages':
        return P_Cs_inter_bound_stats_ages
    raise ValueError("sequence_scheme must be 'sequences' or 'ages', got %s"%repr(sequence_scheme))

def sequences_batch(Cs, track_lens, LocErr, ds, Fs, TrMat, pBL, max_len, cell_dims,
                    frame_len, do_preds, min_len, threshold, max_nb_states):
    '''
    One batch of the sequences recursion holding tracks of several lengths.

    Cs         : (nb_tracks, longest, nb_dims), chronological, rows sorted by
                 DECREASING track length.
    track_lens : (nb_tracks,) decreasing, the number of points of each track.
    LocErr     : (nb_tracks|1, longest|1, nb_dims|1) localization error, squared
                 here as P_Cs_inter_bound_stats_th does.
    max_len    : the length at or above which a track has not ended, i.e.
                 ExTrack's isBL = 0 case.

    A track leaves the batch the moment its own recursion is over, and it leaves
    *before* the fusion of that step -- which is what `if current_step <
    nb_locs-1` does in the per length version: the last step of a track is never
    fused. Because the rows are sorted, the survivors are always a prefix and
    leaving is a slice.

    Unlike the (age, state) scheme, this one shares a single *dynamic* set of
    branches across the batch, so the result is only identical to the per length
    path when no fusion takes place -- `frame_len >= longest` together with a
    threshold small enough that only a branch matches itself. Outside that regime
    the grouping heuristic samples the batch's 30 first tracks and re-batching
    changes which branches merge.

    Returns (LP, preds), LP of shape (nb_tracks,) and preds of
    (nb_tracks, longest, nb_states) when do_preds, in the input row order.
    '''
    nb_Tracks, longest, nb_dims = np.shape(Cs)
    nb_states = TrMat.shape[0]
    Cs = cp.array(np.asarray(Cs, dtype = 'float64'))[:, None]        # (n, 1, longest, dims)
    LocErr2 = cp.array(np.asarray(LocErr, dtype = 'float64'))[:, None]**2
    per_track_err = LocErr2.shape[0] > 1
    per_frame_err = LocErr2.shape[2] > 1
    track_lens = np.asarray(track_lens)

    LP_out = np.zeros(nb_Tracks)
    preds_out = np.zeros((nb_Tracks, longest, nb_states)) if do_preds else None

    TrMatT = cp.array(np.asarray(TrMat).T)
    ds = cp.array(ds)
    Fs = cp.array(Fs)

    cur_Bs = get_all_Bs(2, nb_states)[None]
    cur_Bs_cat = (cur_Bs[:,:,:,None] == np.arange(nb_states)[None,None,None,:]).astype('float64')
    cur_states = cur_Bs[:,:,0:2].astype(int)
    LP = cp.repeat(get_Ts_from_Bs(cur_states, TrMatT) + cp.log(Fs[cur_states[:,:,-1]]),
                   nb_Tracks, axis = 0)

    cur_d2s = ds[cur_states]**2
    cur_d2s = cp.mean((cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2, axis = 2)[:,:,None]

    sub_Bs = cur_Bs.copy()[:,:cur_Bs.shape[1]//nb_states,:1]
    sub_ds = asnumpy((cp.mean(ds[sub_Bs]**2, axis = 2)**0.5).astype(float))
    p_stay = np.ones(sub_ds.shape[-1])
    for cell_len in cell_dims:
        xs = np.linspace(0+cell_len/2000, cell_len-cell_len/2000, 1000)
        p_stay = p_stay * cp.mean(scipy.stats.norm.cdf((cell_len-xs[:,None])/(sub_ds+1e-200)) - scipy.stats.norm.cdf(-xs[:,None]/(sub_ds+1e-200)), 0)
    p_stay = cp.array(p_stay)
    Lp_stay = cp.log(p_stay * (1-pBL))

    rows = np.arange(nb_Tracks)

    def obs(sel_rows, t):
        return Cs[sel_rows][:, :, t]

    def l2(sel_rows, t):
        i = t if per_frame_err else 0
        return LocErr2[sel_rows][:, :, i] if per_track_err else LocErr2[:, :, i]

    m_arr, s2_arr = first_log_integrale_dif(obs(rows, 0), l2(rows, 0), cur_d2s)
    m_arr = cp.repeat(m_arr, cur_Bs.shape[1], axis = 1)

    def retire(sel, m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat):
        '''finish the tracks whose recursion ends at this step'''
        T = int(track_lens[rows[sel[0]]])
        isBL = 0 if T >= max_len else 1
        m_sel = m_arr[sel]
        s2_sel = s2_arr[sel] if s2_arr.shape[0] > 1 else s2_arr
        LP_sel = LP[sel]
        cat = cur_Bs_cat[sel] if cur_Bs_cat.shape[0] > 1 else cur_Bs_cat
        if isBL:
            ext_Bs = np.concatenate((np.mod(np.arange(cur_Bs.shape[1]*nb_states), nb_states)[None,:,None],
                                     np.repeat(cur_Bs, nb_states, 1)), -1)
            new_states = np.repeat(np.mod(np.arange(cat.shape[1]*nb_states, dtype = 'int8'), nb_states)[None,:,None,None] == np.arange(nb_states, dtype = 'int8')[None,None,None], cat.shape[0], 0).astype('int8')
            cat = np.concatenate((new_states, np.repeat(cat, nb_states, 1)), -2)
            ext_states = ext_Bs[:,:,0:2].astype(int)
            LT = get_Ts_from_Bs(ext_states, TrMatT)
            m_sel = cp.repeat(m_sel, nb_states, axis = 1)
            s2_sel = cp.repeat(s2_sel, nb_states, axis = 1)
            LP_sel = cp.repeat(LP_sel, nb_states, axis = 1)
            end_p_stay = p_stay[ext_states[:,None:,:-1]][:,:,0]
            LL_end = cp.log(pBL + (1-end_p_stay) - pBL * (1-end_p_stay)) + LT
            cat = cat[:,:,1:]
        else:
            LL_end = 0

        LP_sel = final_integration(obs(rows[sel], T-1), l2(rows[sel], T-1),
                                   m_sel, s2_sel, LP_sel, LL_end)
        mx = LP_sel.max(1)
        LP_out[rows[sel]] = mx + np.log(np.exp(LP_sel - mx[:,None]).sum(1))
        if do_preds:
            pred_LP = LP_sel
            if np.max(LP_sel) > 600:
                pred_LP = LP_sel - (np.max(LP_sel)-600)
            P = np.exp(pred_LP)
            sum_P = np.sum(P, axis = 1, keepdims = True)[:,:,None]
            pr = np.sum(P[:,:,None,None]*cat, axis = 1) / sum_P
            preds_out[np.ix_(rows[sel], np.arange(T))] = pr[:, ::-1]

    def shrink(cut, m_arr, s2_arr, LP, cur_Bs_cat):
        keep = np.where(track_lens[rows] > cut)[0]
        return (rows[keep], m_arr[keep],
                s2_arr[keep] if s2_arr.shape[0] > 1 else s2_arr,
                LP[keep],
                cur_Bs_cat[keep] if cur_Bs_cat.shape[0] > 1 else cur_Bs_cat)

    # tracks of 2 points never enter the loop, exactly as `nb_locs - 1 < 2` skips it
    if np.any(track_lens[rows] == 2):
        retire(np.where(track_lens[rows] == 2)[0], m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat)
        rows, m_arr, s2_arr, LP, cur_Bs_cat = shrink(2, m_arr, s2_arr, LP, cur_Bs_cat)

    current_step = 2
    while current_step <= longest - 1 and len(rows):
        cur_Bs = np.concatenate((np.mod(np.arange(cur_Bs.shape[1]*nb_states), nb_states)[None,:,None],
                                 np.repeat(cur_Bs, nb_states, 1)), -1)
        new_states = np.repeat(np.mod(np.arange(cur_Bs_cat.shape[1]*nb_states, dtype = 'int8'), nb_states)[None,:,None,None] == np.arange(nb_states, dtype = 'int8')[None,None,None], cur_Bs_cat.shape[0], 0).astype('int8')
        cur_Bs_cat = np.concatenate((new_states, np.repeat(cur_Bs_cat, nb_states, 1)), -2)

        cur_states = cur_Bs[:1,:,0:2].astype(int)
        cur_d2s = ds[cur_states]**2
        cur_d2s = cp.mean((cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2, axis = 2)[:,:,None]
        LT = get_Ts_from_Bs(cur_states, TrMatT)
        if current_step >= min_len:
            LL = Lp_stay[np.argmax(np.all(cur_states[:,None,:,:-1] == sub_Bs[:,:,None],-1),1)]
        else:
            LL = 0

        m_arr, s2_arr, LP = recurrence_step(obs(rows, current_step-1),
                                            l2(rows, current_step-1),
                                            cur_d2s, m_arr, s2_arr, LP, LT, LL, nb_states)
        del LT

        if cur_Bs.shape[1] > max_nb_states:
            threshold = threshold*1.2

        # a track leaves before the fusion of its own last step, never after
        if np.any(track_lens[rows] == current_step + 1):
            retire(np.where(track_lens[rows] == current_step + 1)[0],
                   m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat)
            rows, m_arr, s2_arr, LP, cur_Bs_cat = shrink(current_step + 1, m_arr, s2_arr,
                                                         LP, cur_Bs_cat)

        if len(rows):
            m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat = fuse_tracks_th(
                m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat, len(rows),
                nb_states = nb_states, nb_dims = nb_dims, do_preds = do_preds,
                threshold = threshold, frame_len = frame_len)
        current_step += 1

    return LP_out, preds_out

def pack_tracks(all_tracks, segment_length = None, batch_size = 2000, input_LocErr = None):
    '''
    Build the batches Proba_Cs_batched consumes, once.

    The packing is a function of the track lengths alone, so a fit builds it once
    and passes it to every likelihood evaluation. Doing it inside the likelihood
    instead costs more than the recursion.
    '''
    from extrack import segmentation
    tracks, locerrs, origin = segmentation.flatten_tracks(all_tracks, input_LocErr)
    batches, order, lengths = segmentation.segment_tracks(
        tracks, segment_length, batch_size,
        locerrs = None if input_LocErr is None else locerrs)
    return dict(batches = batches, order = order, lengths = lengths,
                origin = origin, tracks = tracks)

def Proba_Cs_batched(all_tracks, LocErr, ds, Fs, TrMat, pBL, cell_dims, nb_substeps,
                     frame_len, min_len, threshold, max_nb_states, input_LocErr = None,
                     segment_length = None, batch_size = 2000, max_len = None,
                     do_preds = 0, packing = None):
    '''
    The log likelihood of tracks of any lengths, batched instead of grouped by
    length.

    ExTrack's usual path runs one recursion per track length; a data set with
    lengths 5 to 20 pays 16 of them, most on a few hundred tracks. This sorts
    every track by decreasing length, cuts them into segments of
    `segment_length` points (segments share their boundary point) and runs one
    recursion per batch of segments, handing the message from one segment to the
    next through a carry buffer. `isfirst` decides, per track, whether a segment
    initialises the recursion or resumes it.

    Only the `ages` scheme is batched this way: its buffer is
    `frame_len * nb_states` hypotheses whatever the track, so tracks of different
    lengths and different histories can sit in the same batch and each advances
    on its own step count. The `sequences` scheme shares one *dynamic* set of
    branches across the batch, so a fresh track and a resumed one cannot be
    mixed; call this with the tracks already grouped by length there.

    arguments:
    all_tracks: dict of (nb_tracks, track_len, nb_dims) arrays keyed by track length.
    segment_length: number of points per segment, None for one segment per track.
    batch_size: maximum number of tracks per batch.
    packing: the output of `pack_tracks`, reused across calls. The packing depends
        only on the track lengths, never on the parameters, so a fit must build it
        once and hand it back in; rebuilding it inside the likelihood costs more
        than the recursion itself.
    max_len: the length above which a track is considered not to have ended (its
        end of track term is skipped), i.e. ExTrack's isBL = 0 case. Defaults to
        the longest track present.

    outputs:
    LP: dict keyed by track length of (nb_tracks,) log likelihoods, in the input order.
    preds: same, of (nb_tracks, track_len, nb_states) state probabilities, if do_preds.
    '''
    from extrack import segmentation

    if not numba_active():
        raise RuntimeError('the batched path needs the numba kernels; '
                           'install numba or use the per length path')
    if nb_substeps != 1:
        raise NotImplementedError('the batched path only supports nb_substeps = 1')

    tracks, locerrs, origin = segmentation.flatten_tracks(all_tracks, input_LocErr)
    if len(tracks) == 0:
        return {}, {}
    lengths = np.array([len(t) for t in tracks])
    if max_len is None:
        max_len = int(lengths.max())
    nb_states = TrMat.shape[0]
    nb_dims = tracks[0].shape[1]
    hist_len = int(lengths.max())

    ds = np.asarray(ds)
    if ds.ndim != 1:
        raise NotImplementedError('the batched path only supports one diffusion length per state')
    d2 = ds**2
    pair_d2 = (d2[:, None] + d2[None, :]) / 2
    log_TrMat = np.log(np.asarray(TrMat))
    log_Fs = np.log(np.asarray(Fs))

    p_stay = np.ones(nb_states)
    for cell_len in cell_dims:
        xs = np.linspace(0+cell_len/2000, cell_len-cell_len/2000, 1000)
        p_stay = p_stay * np.mean(scipy.stats.norm.cdf((cell_len-xs[:,None])/(ds+1e-200)) - scipy.stats.norm.cdf(-xs[:,None]/(ds+1e-200)), 0)
    Lp_stay = np.log(p_stay * (1-pBL))
    end_LL = np.log(pBL + (1-p_stay) - pBL * (1-p_stay))

    shared_LocErr = None
    if input_LocErr is None:
        shared_LocErr = np.asarray(LocErr)**2       # (1, 1, nb_dims|1)
        shared_LocErr = shared_LocErr.reshape((1, 1, -1))

    if packing is None:
        packing = pack_tracks(all_tracks, segment_length, batch_size, input_LocErr)
    batches = packing['batches']

    LP_flat = np.zeros(len(tracks))
    preds_flat = np.zeros((len(tracks), hist_len, nb_states)) if do_preds else None
    carries = {}
    for b in batches:
        rows = b['rows']
        n = len(rows)
        key = b['chunk']
        if key not in carries:
            # the first batch of a chunk holds all of its tracks and every later
            # one a prefix of them, so it sizes the carry buffers
            carries[key] = new_ages_carry(n, frame_len, nb_states, nb_dims,
                                          hist_len, bool(do_preds))
        full = carries[key]
        pos = np.arange(n)                          # a batch is a prefix of its chunk
        carry = dict(LP = full['LP'][:n], m = full['m'][:n], s2 = full['s2'][:n],
                     cat = full['cat'][:n], nact = full['nact'][:n])

        l2 = shared_LocErr
        if l2 is None:
            l2 = np.ascontiguousarray(b['LocErr'])**2

        # islast = 2 marks a track that stopped before the end of the movie and
        # therefore pays the end of track term, 1 one that ran to the end
        islast = b['islast'] * (1 + (lengths[rows] < max_len).astype(np.int64))

        out_LP, out_cat = ages_kernel_call(
            b['Cs'], l2, log_TrMat, log_Fs, pair_d2, Lp_stay, end_LL,
            frame_len, min_len, b['nsteps'], b['isfirst'], islast, b['start'],
            carry, hist_len, bool(do_preds), nb_states)

        done = b['islast'] == 1
        LP_flat[rows[done]] = out_LP[done]
        if do_preds:
            preds_flat[rows[done]] = out_cat[done]

    LP = {}
    preds = {}
    for key in sorted(all_tracks.keys(), key = int):
        block = np.asarray(all_tracks[key])
        if len(block) == 0:
            continue
        LP[key] = np.zeros(len(block))
        if do_preds:
            preds[key] = np.zeros((len(block), int(key), nb_states))
    for flat_i, (key, row) in enumerate(origin):
        LP[key][row] = LP_flat[flat_i]
        if do_preds:
            preds[key][row] = preds_flat[flat_i, :int(key)]
    return LP, preds

def Pool_star_P_inter(args):
    args = list(args)
    kernel = get_sequence_kernel(args.pop() if len(args) > 14 else 'sequences')
    return kernel(*args)[2] # returns the 3rd output which is the predictions

def predict_Bs(all_tracks,
               dt,
               params,
               cell_dims=[1],
               nb_states=4,
               frame_len=5,
               max_nb_states = 200,
               threshold = 0.1,
               workers = 1,
               input_LocErr = None,
               verbose = 0,
               nb_max = 1,
               sequence_scheme = 'sequences'):
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
    nb_max: integer, number of simultanous predictions. Higher numbers strongly increase the speed but might affect the predictions quality.
    sequence_scheme: 'sequences' (default) keeps one hypothesis per sequence of states over
        the last frame_len frames, 'ages' keeps one per (segment age, current state), a fixed
        buffer of frame_len*nb_states (see get_sequence_kernel).
    
    outputs:
    pred_Bs: dict describing the state probability of each track for each time position with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = state.
    extrack.visualization.visualize_states_durations
    '''
    
    l_list = np.sort(np.array(list(all_tracks.keys())).astype(int)).astype(str)
    sorted_tracks = []
    sorted_LocErrs = []
    sorted_dt = []
    for l in l_list:
        if len(all_tracks[l]) > 0 :
            sorted_tracks.append(all_tracks[l])
            if input_LocErr != None:
                sorted_LocErrs.append(input_LocErr[l])
            if type(dt) == dict:
                sorted_dt.append(dt[l])
    all_tracks = sorted_tracks
    if input_LocErr != None:
        input_LocErr = sorted_LocErrs
    if type(dt) == dict:
        dt = sorted_dt
    
    nb_substeps=1 # substeps should not impact the step labelling
    if type(params) == type(Parameters()):
        LocErr, ds, Fs, TrMat, pBL = extract_params(params, dt, nb_states, nb_substeps, input_LocErr)
        '''elif type(params) == type({}):
        param_kwargs = []
        for param in params:
            param_kwargs.append({'name' : param, 'value' : params[param], 'vary': False})
        new_params = Parameters()
        [new_params.add(**param_kwargs[k]) for k in range(len(params))]
        LocErr, ds, Fs, TrMat, pBL = extract_params(new_params, dt, nb_states, nb_substeps)'''
    else:
        raise TypeError("params must be either of the class 'lmfit.parameter.Parameters' or a dictionary of the relevant parameters")
    all_pred_Bs = []
    
    min_len = int(l_list[0])
    max_len = int(l_list[-1])
    
    Csss = []
    sigss = []
    isBLs = []
    dsss = []
    for k in range(len(all_tracks)):
        Css = all_tracks[k]
        if type(dt) == list:
            dss = ds[k]
        if input_LocErr != None:
            sigs = LocErr[k]
        for n in range(int(np.ceil(len(Css)/nb_max))):
            Csss.append(Css[n*nb_max:(n+1)*nb_max])
            if input_LocErr != None:
                sigss.append(sigs[n*nb_max:(n+1)*nb_max])
            if type(dt) == list:
                dsss.append(dss[n*nb_max:(n+1)*nb_max])
            if Css.shape[1] == max_len:
                isBLs.append(0) # last position correspond to tracks which didn't disapear within maximum track length
            else:
                isBLs.append(1)
    do_preds = 1
    #Cs, LocErr, ds, Fs, TrMat,pBL,isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states = args_prod[0]
    #Cs, LocErr, ds, Fs, TrMat, pBL, isBL, cell_dims, nb_substeps, frame_len, do_preds, min_len, threshold, max_nb_states = args_prod[0]
    if type(dt) == list:
        args_prod = np.array(list(product(Csss, [0], [dsss[0]], [Fs], [TrMat],[pBL], [0],[cell_dims], [nb_substeps], [frame_len], [do_preds], [min_len], [threshold], [max_nb_states], [sequence_scheme])), dtype=object)
        args_prod[:, 2] = dsss
    else:
        args_prod = np.array(list(product(Csss, [0], [ds], [Fs], [TrMat],[pBL], [0],[cell_dims], [nb_substeps], [frame_len], [do_preds], [min_len], [threshold], [max_nb_states], [sequence_scheme])), dtype=object)
    args_prod[:, 6] = isBLs
    if input_LocErr != None:
        args_prod[:,1] = sigss
    else:
        args_prod[:,1] = LocErr
    
    if workers >= 2:
        with multiprocessing.Pool(workers) as pool:
            all_pred_Bs = pool.map(Pool_star_P_inter, args_prod)
    else:
        all_pred_Bs = []
        for args in args_prod:
            all_pred_Bs.append(Pool_star_P_inter(args))
            if verbose:
                print('.', end = '')
    
    all_pred_Bs_dict = {}
    for l in l_list:
        all_pred_Bs_dict[l] = np.empty((0,int(l),nb_states))
    for i, pred_Bs in enumerate(all_pred_Bs):
        all_pred_Bs_dict[str(pred_Bs.shape[1])] = np.concatenate((all_pred_Bs_dict[str(pred_Bs.shape[1])],pred_Bs))

    return all_pred_Bs_dict

'''
20/input_LocErr[l]
Fs = np.array([0.3, 0.3, 0.3])
input_LocErr[l]*1e-5
'''
def extract_params(params, dt, nb_states, nb_substeps, input_LocErr = None, Matrix_type = 1):
    '''
    turn the parameters which differ deppending on the number of states into lists
    ds (diffusion lengths), Fs (fractions), TrMat (substep transiton matrix)
    '''
    param_names = np.sort(list(params.keys()))
    
    LocErr = []
    for param in param_names:
        if param.startswith('LocErr'):
            LocErr.append(params[param].value)

    LocErr = [np.array(LocErr)[None,None]]
    if input_LocErr != None:
        LocErr = []
        if np.any(np.array(list(params.keys())) == 'slope_LocErr'):
            for l in range(len(input_LocErr)):
                LocErr.append(np.clip(input_LocErr[l] * params['slope_LocErr'].value + params['offset_LocErr'].value, 0.000001, np.inf))
        else:
            LocErr = input_LocErr
    Ds = []
    Fs = []
    for param in param_names:
        if param.startswith('D') and len(param)<3:
            Ds.append(params[param].value)
        elif param.startswith('F'):
            Fs.append(params[param].value)
    Ds = np.array(Ds)
    Fs = np.array(Fs)
    #Fs = np.clip(Fs, a_min=1e-10, a_max = np.inf)
    TrMat = np.zeros((len(Ds),len(Ds)))
    for param in params:
        if param == 'pBL':
            pBL = params[param].value
        elif param.startswith('p'):
            i = int(param[1])
            j = int(param[2])
            TrMat[i,j] = params[param].value
    
    TrMat = TrMat/nb_substeps
    
    if Matrix_type == 0:
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 1-np.sum(TrMat,1)
    if Matrix_type == 1: # 1 - exp(-)
        TrMat = 1 - np.exp(-TrMat)
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 1-np.sum(TrMat,1)
    elif Matrix_type == 2:
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = -np.sum(TrMat,1)
        TrMat = linalg.expm(TrMat)
    elif Matrix_type == 3:
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 0
        G = np.copy(TrMat)
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 1-np.sum(TrMat,1)
        G[np.arange(len(Ds)), np.arange(len(Ds))] = -np.sum(G,1)
        TrMatG = linalg.expm(G)
        TrMat = np.mean([TrMat, TrMatG], axis = 0)
    elif Matrix_type == 4:
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 0
        G = np.copy(TrMat)
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 1-np.sum(TrMat,1)
        G[np.arange(len(Ds)), np.arange(len(Ds))] = -np.sum(G,1)
        TrMatG = linalg.expm(G)
        TrMat = (TrMat* TrMatG)**0.5  
    #TrMat = 1 - np.exp(-TrMat)
    #TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 1-np.sum(TrMat,1)
    #print(TrMat)
    if type(dt) == list:
        ds = []
        for t in dt:
            ds.append(np.sqrt(2*Ds[None, None]*t[:,:,None]))
    else:
        ds = np.sqrt(2*Ds*dt)
    
    return LocErr, ds, Fs, TrMat, pBL

def pool_star_proba(args):
    return Proba_Cs(*args)

def cum_Proba_Cs(params, all_tracks, dt, cell_dims, input_LocErr, nb_states, nb_substeps, frame_len, verbose = 1, workers = 1, Matrix_type = 1, threshold = 0.2, max_nb_states = 120, max_number_of_tracks_per_matrix = 2000, sequence_scheme = 'sequences'):
    '''
    each probability can be multiplied to get a likelihood of the model knowing
    the parameters LocErr, D0 the diff coefficient of state 0 and F0 fraction of
    state 0, D1 the D coef at state 1, p01 the probability of transition from
    state 0 to 1 and p10 the proba of transition from state 1 to 0.
    here sum the logs(likelihood) to avoid too big numbers
    '''
    
    LocErr, ds, Fs, TrMat, pBL = extract_params(params, dt, nb_states, nb_substeps, input_LocErr, Matrix_type)
    # LocErr[0,0,1] = 0.028
    
    '''
    if input_LocErr != None:
        LocErr = input_LocErr
    else:
        LocErr = [LocErr] # putting LocErr in a list to perform the cartesian product of lists for parallelisation
    '''
    min_len = all_tracks[0].shape[1]
    max_len = all_tracks[-1].shape[1]
    
    if type(dt) == list:
        avg_ds = np.median(ds[0], axis = (0,1))
    else:
        avg_ds = ds
    
    if np.all(TrMat>0) and np.all(Fs>0) and np.all(avg_ds[1:]-avg_ds[:-1]>=0):
        Cum_P = 0
        Csss = []
        sigss = []
        isBLs = []
        dsss = []
        
        for k in range(len(all_tracks)):
            Css = all_tracks[k]
            if input_LocErr != None:
                sigs = LocErr[k]
            if type(dt) == list:
                dss =  ds[k]
            nb_max = max_number_of_tracks_per_matrix
            for n in range(int(np.ceil(len(Css)/nb_max))):
                Csss.append(Css[n*nb_max:(n+1)*nb_max])
                if input_LocErr != None:
                    sigss.append(sigs[n*nb_max:(n+1)*nb_max])
                if type(dt) == list:
                    dsss.append(dss[n*nb_max:(n+1)*nb_max])
                if Css.shape[1] == max_len:
                    isBLs.append(0) # last position correspond to tracks which didn't disapear within maximum track length
                else:
                    isBLs.append(1)
        Csss.reverse()
        sigss.reverse()
        isBLs.reverse()
        dsss.reverse()
        
        if type(dt) == list:
            args_prod = np.array(list(product(Csss, [0], [dsss[0]], [Fs], [TrMat],[pBL], [0],[cell_dims], [nb_substeps], [frame_len], [min_len], [threshold], [max_nb_states], [sequence_scheme])), dtype=object)
            args_prod[:, 2] = dsss
        else:
            args_prod = np.array(list(product(Csss, [0], [ds], [Fs], [TrMat],[pBL], [0],[cell_dims], [nb_substeps], [frame_len], [min_len], [threshold], [max_nb_states], [sequence_scheme])), dtype=object)
        
        args_prod[:, 6] = isBLs
        if input_LocErr != None:
            args_prod[:,1] = sigss
        else:
            args_prod[:,1] = LocErr

        #Cs, LocErr, ds, Fs, TrMat,pBL,isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states = args_prod[0]
        
        #if workers >= 2 and start_method == 'fork':
        if workers >= 2 and start_method == 'fork':
            with multiprocessing.Pool(workers) as pool:
                LP = pool.map(pool_star_proba, args_prod)
        else:
            LP = []
            for args in args_prod:
                LP.append(pool_star_proba(args))
        
        Cum_P += cp.sum(cp.concatenate(LP))
        Cum_P = asnumpy(Cum_P)
        
        if verbose == 1:
            q = [param + ' = ' + str(np.round(params[param].value, 6)) for param in params]
            print(Cum_P, q)
        else:
            print('.', end='')
        out = - Cum_P # normalize by the number of tracks and number of displacements
    else:
        out = np.inf
        print('x',end='')
        if verbose == 1:
            q = [param + ' = ' + str(np.round(params[param].value, 4)) for param in params]
            print(q)
    if np.isnan(out):
        out = np.inf
        print('input parameters give nans, you may want to pick more suitable parameter initial values')
    #print(time() - t0)
    return out

def get_params(nb_states = 2,
               steady_state = False,
               vary_params = {'LocErr' : True, 'D0' : True, 'D1' : True, 'F0' : True, 'p01' : True, 'p10' : True, 'pBL' : True},
               estimated_vals = {'LocErr' : 0.025, 'D0' : 1e-20, 'D1' : 0.05, 'F0' : 0.45, 'p01' : 0.05, 'p10' : 0.05, 'pBL' : 0.1},
               min_values = {'LocErr' : 0.007, 'D0' : 1e-12, 'D1' : 0.00001, 'F0' : 0.001, 'p01' : 0.01, 'p10' : 0.01, 'pBL' : 0.01},
               max_values = {'LocErr' : 0.6, 'D0' : 1, 'D1' : 10, 'F0' : 0.999, 'p01' : 1., 'p10' : 1., 'pBL' : 0.99}):
    if 0:
        1
        '''
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
                            #{'name' : 'F0', 'value' : estimated_vals['F0'], 'min' : min_values['F0'], 'max' : max_values['F0'], 'vary' : vary_params['F0']},
                            #{'name' : 'F1_minus_F0', 'value' : (estimated_vals['F1'])/(1-estimated_vals['F0']), 'min' : min_values['F1'], 'max' : max_values['F1'], 'vary' : vary_params['F1']},
                            #{'name' : 'F1', 'expr' : 'F1_minus_F0*(1-F0)'},
                            {'name' : 'F0', 'value' : estimated_vals['F0'], 'min' : min_values['F0'], 'max' : max_values['F0'], 'vary' : vary_params['F0']},
                            {'name' : 'F1', 'value' : estimated_vals['F1'], 'min' : min_values['F1'], 'max' : max_values['F1'], 'vary' : vary_params['F1']},
                            {'name' : 'F2', 'expr' : '1-F0-F1'},
                            {'name' : 'pBL', 'value' : estimated_vals['pBL'], 'min' :  min_values['pBL'], 'max' :  max_values['pBL'], 'vary' : vary_params['pBL']}]
        '''
    else :
        param_kwargs = []
        if np.any(np.array(list(estimated_vals.keys())) == 'slope_LocErr'):
            param_kwargs.append({'name' : 'slope_LocErr', 'value' :  estimated_vals['slope_LocErr'], 'min' :  min_values['slope_LocErr'], 'max' :  max_values['slope_LocErr'], 'vary' :  vary_params['slope_LocErr']})
            param_kwargs.append({'name' : 'offset_LocErr', 'value' :  estimated_vals['offset_LocErr'], 'min' :  min_values['offset_LocErr'], 'max' :  max_values['offset_LocErr'], 'vary' :  vary_params['offset_LocErr']})
    
        if np.any(np.array(list(estimated_vals.keys())) == 'LocErr'):
            LocErr = estimated_vals['LocErr']
            # consider LocErr as a parameter or not depending of its format
            if type(LocErr) == float:
                param_kwargs.append({'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' : min_values['LocErr'], 'max' : max_values['LocErr'] , 'vary' : vary_params['LocErr']})
            elif type(LocErr) == np.ndarray or type(LocErr) == list: # if one Localization error parameter per axis
                for s in range(len(LocErr)):
                    param_kwargs.append({'name' : 'LocErr' + str(s), 'value' : estimated_vals['LocErr'][s], 'min' : min_values['LocErr'][s], 'max' : max_values['LocErr'][s] , 'vary' : vary_params['LocErr'][s]})
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
        
        param_kwargs.append({'name' : 'F0', 'value' : estimated_vals['F0'], 'min' : min_values['F0'], 'max' : max_values['F0'], 'brute_step' : 0.04, 'vary' : vary_params['F0']})
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

def generate_params(nb_states = 3,
                    LocErr_type = 1,
                    nb_dims = 3, # only matters if LocErr_type == 2,
                    LocErr_bounds = [0.005, 0.1], # the initial guess on LocErr will be the geometric mean of the boundaries
                    D_max = 10, # maximal diffusion coefficient allowed
                    Fractions_bounds = [0.001, 0.99],
                    estimated_LocErr = None,
                    estimated_Ds = None, # D will be arbitrary spaced from 0 to D_max if None, otherwise input 1D array/list of Ds for each state from state 0 to nb_states - 1.
                    estimated_Fs = None, # fractions will be equal if None, otherwise input 1D array/list of fractions for each state from state 0 to nb_states - 1.
                    estimated_transition_rates = 0.1, # transition rate per step. [0.1,0.05,0.03,0.07,0.2,0.2]
                    slope_offsets_estimates = None # need to specify the list [slop, offset] if LocErr_type = 4,
                    ):
    '''
    nb_states: number of states of the model.
    LocErr_type: 1 for a single localization error parameter,
                 2 for a localization error parameter for each dimension,
                 3 for a shared localization error for x and y dims (the 2 first dimensions) and another for z dim.
                 4 for an affine relationship between localization error a peak-wise input specified with input_LocErr (like an estimate of localization error/quality of peak/signal to noise ratio, etc).
                 None for no localization error fits, localization error is then directly assumed from a prior peak-wise estimate of localization error specified in input_LocErr.
    '''
    param_kwargs = []
    if estimated_Ds == None:
        for s in range(nb_states):
            param_kwargs.append({'name' : 'D'+str(s), 'value' : 0.5*s**2 * D_max / (nb_states-1)**2, 'min' : 0, 'max' : D_max, 'vary' : True})
    else:
        for s in range(nb_states):
            param_kwargs.append({'name' : 'D'+str(s), 'value' : estimated_Ds[s], 'min' : 0, 'max' : D_max, 'vary' : True})
    if estimated_LocErr == None:
        if LocErr_type == 1:
            param_kwargs.append({'name' : 'LocErr', 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 2:
            for d in range(nb_dims):
                param_kwargs.append({'name' : 'LocErr' + str(d), 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 3:
            param_kwargs.append({'name' : 'LocErr0', 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
            param_kwargs.append({'name' : 'LocErr1', 'expr' : 'LocErr0'})
            param_kwargs.append({'name' : 'LocErr2', 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
    else:
        if LocErr_type == 1:
            param_kwargs.append({'name' : 'LocErr', 'value' : estimated_LocErr[0], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 2:
            for d in range(nb_dims):
                param_kwargs.append({'name' : 'LocErr' + str(d), 'value' : estimated_LocErr[d], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 3:
            param_kwargs.append({'name' : 'LocErr0', 'value' : estimated_LocErr[0], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
            param_kwargs.append({'name' : 'LocErr1', 'expr' : 'LocErr0'})
            param_kwargs.append({'name' : 'LocErr2', 'value' : estimated_LocErr[-1], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})

    if LocErr_type == 4:
        param_kwargs.append({'name' : 'slope_LocErr', 'value' : slope_offsets_estimates[0], 'min' : -1, 'max' : 20, 'vary' : True})
        param_kwargs.append({'name' : 'offset_LocErr', 'value' : slope_offsets_estimates[1], 'min' : -1, 'max' : 1, 'vary' : True})
    
    F_expr = '1' 
    if estimated_Fs == None:
        for s in range(nb_states-1):
            param_kwargs.append({'name' : 'F'+str(s), 'value' : 1/nb_states, 'min' : Fractions_bounds[0], 'max' : Fractions_bounds[1], 'vary' : True})
            F_expr +=  ' - F'+str(s)
    else:
        for s in range(nb_states-1):
            param_kwargs.append({'name' : 'F'+str(s), 'value' : estimated_Fs[s], 'min' : Fractions_bounds[0], 'max' : Fractions_bounds[1], 'vary' : True})
            F_expr +=  ' - F'+str(s)
    param_kwargs.append({'name' : 'F'+str(nb_states-1), 'expr' : F_expr})
    
    if not (type(estimated_transition_rates) == np.ndarray or type(estimated_transition_rates) == list):
        estimated_transition_rates = [estimated_transition_rates] * (nb_states * (nb_states-1))
    idx = 0
    for i in range(nb_states):
        for j in range(nb_states):
            if i != j:
                param_kwargs.append({'name' : 'p'+ str(i) + str(j), 'value' : estimated_transition_rates[idx], 'min' : 0.0001, 'max' : 1, 'vary' : True})
                idx += 1
    param_kwargs.append({'name' : 'pBL', 'value' : 0.1, 'min' : 0.0001, 'max' : 1, 'vary' : True})
  
    params = Parameters()
    [params.add(**param_kwargs[k]) for k in range(len(param_kwargs))]
    
    return params

from copy import deepcopy

'''
all_tracks = tracks
params = lmfit_params
'''
#all_tracks = tracks
def param_fitting(all_tracks,
                  dt,
                  params = None,
                  nb_states = 2,
                  nb_substeps = 1,
                  frame_len = 6,
                  verbose = 1,
                  workers = 1,
                  Matrix_type = 1,
                  method = 'bfgs',
                  steady_state = False,
                  cell_dims = [1], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                  input_LocErr = None, 
                  threshold = 0.2,
                  max_nb_states = 120,
                  sequence_scheme = 'sequences'):
    
    '''
    fitting the parameters to the data set
    arguments:
    all_tracks: Dictionary describing the tracks with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = x, y position. This means 15 tracks of 7 time points in 2D will correspond to an array of shape [15,7,2].
    dt: Time in between frames.
    params: Parameters previously instanciated.
    nb_states: Number of states. vary_params, estimated_vals, min_values, max_values should be changed accordingly to describe all states and transitions.
    nb_substeps: Number of considered transition steps in between consecutive 2 positions.
    frame_len: Number of frames for which the probability is perfectly computed. See method of the paper for more details.
    verbose: If 1, print the intermediate values for each iteration of the fit.
    steady_state: True if tracks are considered at steady state (fractions independent of time), this is most likely not true as tracks join and leave the FOV.
    workers: Number of workers used for the fitting, allows to speed up computation. Do not work from windows at the moment.
    input_LocErr: Optional peakwise localization errors used as an input with the same format than all_tracks.
    cell_dims: Dimension limits (um) (default [1], can also be [1,2] for instance in case of two limiting dimensions).
    threshold: threshold for the fusion of the sequences of states (default value = 0.2). The threshold is applied to mu the mean position and s the standard deviation of the particle position (see the article for more details).
    max_nb_states: maximum number of sequences of states to consider.
    sequence_scheme: 'sequences' (default) keeps one hypothesis per sequence of states over
        the last frame_len frames, 'ages' keeps one per (segment age, current state), a fixed
        buffer of frame_len*nb_states (see get_sequence_kernel).
    method: Optimization method used by the lmfit package (default = 'BFGS'). Other methods may not work.

    outputs:
    model_fit: lmfit model
    '''
    
    if params == None:
        params = generate_params(nb_states = nb_states,
                               LocErr_type = 1,
                               LocErr_bounds = [0.005, 0.1], # the initial guess on LocErr will be the geometric mean of the boundaries
                               D_max = 3, # maximal diffusion length allowed
                               Fractions_bounds = [0.001, 0.99],
                               estimated_transition_rates = 0.1 # transition rate per step.
                               )
        
    l_list = np.sort(np.array(list(all_tracks.keys())).astype(int)).astype(str)
    sorted_tracks = []
    sorted_LocErrs = []
    if type(dt) == dict: 
        sorted_dt = []
    for l in l_list:
        if len(all_tracks[l]) > 0 :
            sorted_tracks.append((all_tracks[l]))
            if input_LocErr != None:
                sorted_LocErrs.append(input_LocErr[l])
            if type(dt) == dict: 
                sorted_dt.append(dt[l])

    all_tracks = sorted_tracks
    if len(all_tracks) < 1:
        raise ValueError('No track could be detected. The loaded tracks seem empty. Errors often come from wrong input paths.')

    if input_LocErr != None:
        input_LocErr = sorted_LocErrs
    
    if type(dt) == dict: 
        dt = sorted_dt
    
    print('cell_dims', cell_dims)
    
    fit = minimize(cum_Proba_Cs, params, args=(all_tracks, dt, cell_dims,input_LocErr, nb_states, nb_substeps, frame_len, verbose, workers, Matrix_type, threshold, max_nb_states, 2000, sequence_scheme), method = method, nan_policy = 'propagate')
    if verbose == 0:
        print('')
        
    '''
    #to inverse state indexes:
    import copy
    idxs = [1,0,2,3]
    corr_params = copy.deepcopy(fit.params)
    for param in params[-13:-1]:
        i =  idxs[int(param[1])]
        j =  idxs[int(param[2])]
        val = float(res[param][3])
        corr_params['p' + i + j][3] = val
    '''
    return fit
