#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Nov 11 17:01:56 2021
@author: francois
"""

import numpy as np
from itertools import product

GPU_computing = False

if GPU_computing :
    import cupy as cp
    from cupy import asnumpy
else :
    import numpy as cp
    def asnumpy(x):
        return np.array(x)

import multiprocessing
import scipy
from extrack.tracking import extract_params, get_all_Bs, get_Ts_from_Bs, first_log_integrale_dif, log_integrale_dif

def P_segment_len(Cs, LocErr, ds, Fs, TrMat, min_l = 3, pBL=0.1, isBL = 1, cell_dims = [0.5], nb_substeps=1, max_nb_states = 1000) :
    '''
    compute the product of the integrals over Ri as previousily described
    work in log space to avoid overflow and underflow
    
    Cs : dim 0 = track ID, dim 1 : states, dim 2 : peaks postions through time,
    dim 3 : x, y position
    
    we process by steps, at each step we account for 1 more localization, we compute
    the canstant (LC), the mean (m_arr) and std (s2_arr) of of the normal distribution 
    resulting from integration.
    
    each step is made of substeps if nb_substeps > 1, and we increase the matrix
    of possible Bs : cur_Bs accordingly
    
    to be able to process long tracks with good accuracy, for each track we fuse m_arr and s2_arr
    of sequences of states equal exept for the state 'frame_len' steps ago.
    '''
    nb_Tracks = Cs.shape[0]
    nb_locs = Cs.shape[1] # number of localization per track
    nb_dims = Cs.shape[2] # number of spatial dimentions (x, y) or (x, y, z)
    Cs = Cs.reshape((nb_Tracks,1,nb_locs, nb_dims))
    Cs = cp.array(Cs)
    nb_states = TrMat.shape[0]
    Cs = Cs[:,:,::-1]
    LocErr = LocErr[:,None]
    LocErr = LocErr[:,:,::-1] # useful when single peak localization error is inputed,
    LocErr2 = LocErr**2
    if LocErr.shape[2] == 1: # combined to min(LocErr_index, nb_locs-current_step) it will select the right index
        LocErr_index = -1
    elif LocErr.shape[2] == nb_locs:
        LocErr_index = nb_locs
    else:
        raise ValueError("Localization error is not specified correctly, in case of unique localization error specify a float number in estimated_vals['LocErr'].\n If one localization error per dimension, specify a list or 1D array of elements the localization error for each dimension.\n If localization error is predetermined by another method for each position the argument input_LocErr should be a dict for each track length of the 3D arrays corresponding to all_tracks (can be obtained from the reader functions using the opt_colname argument)")
    
    cell_dims = np.array(cell_dims)
    cell_dims = cell_dims[cell_dims!=None]
    
    cur_Bs = get_all_Bs(nb_substeps + 1, nb_states)[None]
    
    TrMat = cp.array(TrMat.T)
    current_step = 1
    if nb_locs ==1:
        
        cur_states = get_all_Bs(1, nb_states)[None] #states of interest for the current displacement
        cur_Bs = get_all_Bs(2, nb_states)[None]
        
        cur_d2s = ds[cur_states]**2
        cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps
        
        # we can average the variances of displacements per step to get the actual std of displacements per step
        cur_d2s = cp.mean(cur_d2s, axis = 2)
        cur_d2s = cur_d2s[:,:,None]
        cur_d2s = cp.array(cur_d2s)
        
        sub_Bs = cur_Bs.copy()[:1,:cur_Bs.shape[1]//nb_states,:nb_substeps] # list of possible current states we can meet to compute the proba of staying in the FOV
        sub_ds = cp.mean(ds[sub_Bs]**2, axis = 2)**0.5 # corresponding list of d
        
        p_stay = np.ones(sub_ds.shape[-1])
        for cell_len in cell_dims:
            xs = np.linspace(0+cell_len/2000,cell_len-cell_len/2000,1000)
            cur_p_stay = ((np.mean(scipy.stats.norm.cdf((cell_len-xs[:,None])/(sub_ds+1e-200)) - scipy.stats.norm.cdf(-xs[:,None]/(sub_ds+1e-200)),0))*2)/2 # proba to stay in the FOV for each of the possible cur Bs
            p_stay = p_stay*cur_p_stay
        Lp_stay = np.log(p_stay * (1-pBL)) # proba for the track to survive = both stay in the FOV and not bleach
        
        LL = Lp_stay[np.argmax(np.all(cur_Bs[:,None,:,:-1] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states
        LP = np.zeros((LL.shape))
        
        cur_Bs = np.repeat(cur_Bs,nb_Tracks,axis = 0)
        cur_Bs = cur_Bs[:,:,1:]
        
    else:
        cur_Bs = get_all_Bs(nb_substeps + 1, nb_states)[None]
        cur_Bs = np.repeat(cur_Bs, nb_Tracks, 0)
        
        cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int) #states of interest for the current displacement
        cur_nb_Bs = cur_Bs.shape[1]
        # compute the vector of diffusion stds knowing the current states
        ds = cp.array(ds)
        Fs = cp.array(Fs)
        
        LT = get_Ts_from_Bs(cur_states, TrMat) # Log proba of transitions per step
        LF = cp.log(Fs[cur_states[:,:,-1]]) # Log proba of finishing/starting in a given state (fractions)
        
        LP = LT + LF #+ compensate_leaving
        LL = np.zeros(LP.shape)
        # current log proba of seeing the track
        #LP = cp.repeat(LP, nb_Tracks, axis = 0)
        cur_d2s = ds[cur_states]**2
        cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps
    
        # we can average the variances of displacements per step to get the actual std of displacements per step
        cur_d2s = cp.mean(cur_d2s, axis = 2)
        cur_d2s = cur_d2s[:,:,None]
        cur_d2s = cp.array(cur_d2s)
        
        sub_Bs = cur_Bs.copy()[:1,:cur_Bs.shape[1]//nb_states,:nb_substeps] # list of possible current states we can meet to compute the proba of staying in the FOV
        sub_ds = cp.mean(ds[sub_Bs]**2, axis = 2)**0.5 # corresponding list of d
        
        p_stay = np.ones(sub_ds.shape[-1])
        for cell_len in cell_dims:
            xs = np.linspace(0+cell_len/2000,cell_len-cell_len/2000,1000)
            cur_p_stay = ((np.mean(scipy.stats.norm.cdf((cell_len-xs[:,None])/(sub_ds+1e-200)) - scipy.stats.norm.cdf(-xs[:,None]/(sub_ds+1e-200)),0))) # proba to stay in the FOV for each of the possible cur Bs
            p_stay = p_stay*cur_p_stay
        Lp_stay = np.log(p_stay * (1-pBL)) # proba for the track to survive = both stay in the FOV and not bleach
        
        if current_step >= min_l:
            LL = LL + Lp_stay[np.argmax(np.all(cur_states[:,None,:,:-1] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states

        # inject the first position to get the associated m_arr and s2_arr :
        m_arr, s2_arr = first_log_integrale_dif(Cs[:,:, nb_locs-current_step], LocErr2[:,:, min(LocErr_index, nb_locs-current_step)], cur_d2s)
        current_step += 1
        m_arr = cp.repeat(m_arr, cur_nb_Bs, axis = 1)
        removed_steps = 0
        
        while current_step <= nb_locs-1:
            # update cur_Bs to describe the states at the next step :
            #cur_Bs = get_all_Bs(current_step*nb_substeps+1 - removed_steps, nb_states)[None]
            #cur_Bs = all_Bs[:,:nb_states**(current_step*nb_substeps+1 - removed_steps),:current_step*nb_substeps+1 - removed_steps]
            for iii in range(nb_substeps):
                cur_Bs = np.concatenate((np.repeat(np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], nb_Tracks,0),np.repeat(cur_Bs,nb_states,1)),-1)
            
            cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int)
            # compute the vector of diffusion stds knowing the states at the current step
            cur_d2s = ds[cur_states]**2
            cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps
        
            # we can average the variances of displacements per step to get the actual std of displacements per step
            cur_d2s = cp.mean(cur_d2s, axis = 2)
            cur_d2s = cur_d2s[:,:,None]
            cur_d2s = cp.array(cur_d2s)
            LT = get_Ts_from_Bs(cur_states, TrMat)
    
            # repeat the previous matrix to account for the states variations due to the new position
            m_arr = cp.repeat(m_arr, nb_states**nb_substeps , axis = 1)
            s2_arr = cp.repeat(s2_arr, nb_states**nb_substeps, axis = 1)
            LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)
            LL = cp.repeat(LL, nb_states**nb_substeps, axis = 1)
            # inject the next position to get the associated m_arr, s2_arr and Constant describing the integral of 3 normal laws :
            m_arr, s2_arr, LC = log_integrale_dif(Cs[:,:,nb_locs-current_step], LocErr2[:,:, min(LocErr_index, nb_locs-current_step)], cur_d2s, m_arr, s2_arr)
            #print('integral',time.time() - t0); t0 = time.time()
            if current_step >= min_l :
                LL = LL + Lp_stay[np.argmax(np.all(cur_states[:,None,:,:-1] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states

            LP += LT + LC # current (log) constants associated with each track and sequences of states
            del LT, LC
            cur_nb_Bs = len(cur_Bs[0]) # current number of sequences of states
            
            ''''idea : the position and the state 6 steps ago should not impact too much the 
            probability of the next position so the m_arr and s2_arr of tracks with the same 6 last 
            states must be very similar, we can then fuse the parameters of the pairs of Bs
            which vary only for the last step (7) and sum their probas'''
            if current_step < nb_locs-1:
                if cur_nb_Bs > max_nb_states:
                    
                    #new_s2_arr = cp.array((s2_arr + LocErr2))[:,:,0]
                    #log_integrated_term = -cp.log(2*np.pi*new_s2_arr) - cp.sum((Cs[:,:,nb_locs-current_step] - m_arr)**2,axis=2)/(2*new_s2_arr)
                    new_s2_arr = cp.array((s2_arr + LocErr2[:,:, min(LocErr_index, nb_locs-current_step-1)]))
                    log_integrated_term = cp.sum(-0.5*cp.log(2*np.pi*new_s2_arr) - (Cs[:,:,nb_locs-current_step-1] - m_arr)**2/(2*new_s2_arr),axis=2)
                    LF = 0 #cp.log(Fs[cur_Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
                    
                    test_LP = LP + log_integrated_term + LF
                    LP.shape
                    if np.max(test_LP)>600: # avoid overflow of exponentials, mechanically also reduces the weight of longest tracks
                        test_LP = test_LP - (np.max(test_LP)-600)
                    
                    P = np.exp(test_LP)
                    
                    argP = P.argsort()
                    argP = argP[:,::-1]
                    
                    m_arr = np.take_along_axis(m_arr, argP[:,:,None], axis = 1)[:,:max_nb_states]
                    s2_arr = np.take_along_axis(s2_arr, argP[:,:,None], axis = 1)[:,:max_nb_states]
                    LP = np.take_along_axis(LP, argP, axis = 1)[:,:max_nb_states]
                    LL = np.take_along_axis(LL, argP, axis = 1)[:,-max_nb_states:]
                    cur_Bs = np.take_along_axis(cur_Bs, argP[:,:,None], axis = 1)[:,:max_nb_states]
                    
                    P = np.take_along_axis(P, argP, axis = 1)[:,:max_nb_states]
                    
                    cur_nb_Bs = cur_Bs.shape[1]
                    removed_steps += 1
                    
            current_step += 1
        
        
        if isBL:
            for iii in range(nb_substeps):
                cur_Bs = np.concatenate((np.repeat(np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], nb_Tracks,0),np.repeat(cur_Bs,nb_states,1)),-1)
            
            cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int)
    
            LT = get_Ts_from_Bs(cur_states, TrMat)
            #cur_states = cur_states[:,:,0]
            # repeat the previous matrix to account for the states variations due to the new position
            m_arr = cp.repeat(m_arr, nb_states**nb_substeps , axis = 1)
            s2_arr = cp.repeat(s2_arr, nb_states**nb_substeps, axis = 1)
            LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)# + LT
            LL = cp.repeat(LL, nb_states**nb_substeps, axis = 1)
            #LL = Lp_stay[np.argmax(np.all(cur_states[:,None] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states
            end_p_stay = p_stay[np.argmax(np.all(cur_states[:,None,:] == sub_Bs[:,:,None],-1),1)]
            
            np.all(cur_states[:,None,:] == sub_Bs[:,:,None],-1).shape
            cur_states[:,:,0]
            sub_Bs[:,:,None].shape
            end_p_stay
            LP.shape
            LL = LL + np.log(pBL + (1-end_p_stay) - pBL * (1-end_p_stay))
            cur_Bs = cur_Bs[:,:,1:]
            #isBL = 0
        
        #new_s2_arr = cp.array((s2_arr + LocErr2))[:,:,0]
        #log_integrated_term = -cp.log(2*np.pi*new_s2_arr) - cp.sum((Cs[:,:,0] - m_arr)**2,axis=2)/(2*new_s2_arr)
        new_s2_arr = cp.array((s2_arr + LocErr2[:,:, min(LocErr_index, 0)]))
        log_integrated_term = cp.sum(-0.5*cp.log(2*np.pi*new_s2_arr) - (Cs[:,:,0] - m_arr)**2/(2*new_s2_arr),axis=2)

        #LF = cp.log(Fs[cur_Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
        #LF = cp.log(0.5)
        # cp.mean(cp.log(Fs[cur_Bs[:,:,:].astype(int)]), 2) # Log proba of starting in a given state (fractions)
        LP += log_integrated_term
        
        LP = LP
        if np.max(LP)>600: # avoid overflow of exponentials, mechanically also reduces the weight of longest tracks
            LP = LP - (np.max(LP, axis = 0, keepdims=True)-600)
        
    P = np.exp(LP+LL)
    #P = np.exp(LP)
    
    cur_nb_Bs = len(cur_Bs[0])
    cur_pos = cur_Bs[:,:,0]
    cur_seg_len = np.ones((nb_Tracks, cur_nb_Bs))
    seg_lens = np.zeros((nb_Tracks, cur_nb_Bs, nb_locs, nb_states)) # dims : track ID, sequence of states ID, position in the sequence, state of the sequence
    if nb_locs == 1:
        last_len = 1
    else:
        for k in range(1, nb_locs):
            is_Tr = cur_pos != cur_Bs[:,:,k]
            cur_seg_len = cur_seg_len + (is_Tr==0).astype(int)
            len_before_tr = (cur_seg_len * is_Tr.astype(int))[:,:,None] # cur_seg_len * is_Tr.astype(int)) selects the segments that stops so we can add them, if 0 : not transition if n : transition after n consecutive steps
            cat_states = (cur_pos[:,:,None] == np.arange(nb_states)[None,None]).astype(int) # position of the segment in categorcal format so it can fit seg_lens
            seg_lens[:,:,k-1] = len_before_tr * cat_states  # add the segment len of the corresponding states 
            cur_seg_len[is_Tr] = 1
            cur_pos = cur_Bs[:,:,k]
            last_len = (nb_locs - np.sum(seg_lens, (2,3)))[:,:,None]
    
    cat_states = (cur_pos[:,:,None] == np.arange(nb_states)[None,None]).astype(int)
    seg_lens[:,:,-1] = last_len * cat_states # include the lenght of the last position (bleaching or leaving the FOV)
    #seg_lens[:,:,-1] = 0 # do not include the lenght of the last position (bleaching or leaving the FOV)
    #seg_lens[:,:,0] = 0 # do not include the lenght of the last position (bleaching or leaving the FOV)
    #last_seg_lens = seg_lens[:,:,-1:]
    
    seg_len_hist = np.zeros((nb_locs-1, nb_states))
    #last_seg_len_hist = np.zeros((nb_locs, nb_states))

    for k in range(1, nb_locs):
        P_seg = ((P/np.sum(P, axis=1, keepdims=True))[:,:,None,None] * (seg_lens == k).astype(int))
        #P_seg = np.sum(P_seg, (1,2)) / (np.sum(P_seg,(1,2,3))[:,None] +1e-300) # seemed good, normalize with regard to the proba of the track without the proba of leaving the FOV (to sum up over different track lengths) 
        P_seg = np.sum(P_seg, (1,2)) # seemed good, normalize with regard to the proba of the track without the proba of leaving the FOV (to sum up over different track lengths) 
        P_seg = np.sum(P_seg,0)
        seg_len_hist[k-1] = P_seg
    
    return LP, cur_Bs, seg_len_hist

def pool_star_P_seg(args):
    return P_segment_len(*args)[-1]

# all_tracks = tracks
# params = lmfit_params

def len_hist(all_tracks,
             params, 
             dt, 
             cell_dims=[0.5,None,None], 
             nb_states=2, 
             max_nb_states = 500,
             workers = 1,
             nb_substeps=1,
             input_LocErr = None
             ):
    '''
    each probability can be multiplied to get a likelihood of the model knowing
    the parameters LocErr, D0 the diff coefficient of state 0 and F0 fraction of
    state 0, D1 the D coef at state 1, p01 the probability of transition from
    state 0 to 1 and p10 the proba of transition from state 1 to 0.
    here sum the logs(likelihood) to avoid too big numbers
    '''
    min_l = np.min((np.array(list(all_tracks.keys()))).astype(int))
    
    if type(input_LocErr) == dict:
        new_input_LocErr = []
        for l in input_LocErr:
            new_input_LocErr.append(input_LocErr[l])
        input_LocErr = new_input_LocErr
    
    if type(all_tracks) == dict:
        new_all_tracks = []
        for l in all_tracks:
            new_all_tracks.append(all_tracks[l])
        all_tracks = new_all_tracks
       
    LocErr, ds, Fs, TrMat, pBL = extract_params(params, dt, nb_states, nb_substeps, input_LocErr)
    
    Csss = []
    sigss = []
    isBLs = []
    for k in range(len(all_tracks)):
        if k == len(all_tracks)-1:
            isBL = 0 # last position correspond to tracks which didn't disapear within maximum track length
        else:
            isBL = 1
        Css = all_tracks[k]
        if input_LocErr != None:
            sigs = LocErr[k]
        nb_max = 50
        for n in range(int(np.ceil(len(Css)/nb_max))):
            Csss.append(Css[n*nb_max:(n+1)*nb_max])
            if input_LocErr != None:
                sigss.append(sigs[n*nb_max:(n+1)*nb_max])
            if k == len(all_tracks)-1:
                isBLs.append(0) # last position correspond to tracks which didn't disapear within maximum track length
            else:
                isBLs.append(1)
    #Csss.reverse()
    #sigss.reverse()
    print('number of chunks:', len(isBLs))
    
    args_prod = np.array(list(product(Csss, [0], [ds], [Fs], [TrMat], [min_l],[pBL], [0],[cell_dims], [nb_substeps],  [max_nb_states])), dtype=object)
    args_prod[:, 7] = isBLs
    if input_LocErr != None:
        args_prod[:,1] = sigss
    else:
        args_prod[:,1] = LocErr

    Cs, LocErr, ds, Fs, TrMat, min_l, pBL,isBL, cell_dims, nb_substeps, max_nb_states = args_prod[0]
    
    if workers >= 2:
        with multiprocessing.Pool(workers) as pool:
            list_seg_len_hists = pool.map(pool_star_P_seg, args_prod)
    else:
        list_seg_len_hists = []
        for k, args in enumerate(args_prod):
            
            list_seg_len_hists.append(pool_star_P_seg(args))
    
    seg_len_hists = np.zeros((all_tracks[-1].shape[1],nb_states))
    for seg_len_hist in list_seg_len_hists:
        seg_len_hists[:seg_len_hist.shape[0]] = seg_len_hists[:seg_len_hist.shape[0]] + seg_len_hist
    print('')
    return seg_len_hists

'''
    if type(all_tracks) == dict:
        new_all_tracks = []
        for l in all_tracks:
            new_all_tracks.append(all_tracks[l])
        all_tracks = new_all_tracks
       
    #seg_len_hists = np.zeros((all_tracks[-1].shape[1]+1,nb_states))
    seg_len_hists = np.zeros((all_tracks[-1].shape[1],nb_states))
    for k in range(len(all_tracks)):
        print('.', end='')
        if k == len(all_tracks)-1:
            isBL = 1 # last position correspond to tracks which didn't disapear within maximum track length
        else:
            isBL = 0
        
        Css = all_tracks[k]
        if len(Css) > 0:
            nb_max = 50
            for n in range(int(np.ceil(len(Css)/nb_max))):
                Csss = Css[n*nb_max:(n+1)*nb_max]
                LP, cur_Bs, seg_len_hist  = P_segment_len(Csss, LocErr, ds, Fs, TrMat, min_l = min_l, pBL=pBL, isBL = isBL, cell_dims = cell_dims, nb_substeps=nb_substeps, max_nb_states = max_nb_states)
                isBL = 0
                seg_len_hists[:seg_len_hist.shape[0]] = seg_len_hists[:seg_len_hist.shape[0]] + seg_len_hist
    print('')
    return seg_len_hists
'''

def ground_truth_hist(all_Bs,
                      nb_states = 2,
                      long_tracks = False, # return hist from long tracks only
                      nb_steps_lim = 20): # if long_tracks = True minimum track length considered
    
    if long_tracks:
        for i, l in enumerate(list(all_Bs.keys())):
            if int(l) < nb_steps_lim:
                del all_Bs[l]
    seg_len_hists = np.zeros((np.max(np.array(list(all_Bs.keys())).astype(int)),nb_states))
    
    for i, l in enumerate(all_Bs):
        cur_Bs = all_Bs[l][:,None]
        if len(cur_Bs)>0:
            if cur_Bs.shape[-1] == 1 :
                nb_Tracks = cur_Bs.shape[0]
                nb_locs = cur_Bs.shape[2]
                cur_nb_Bs = len(cur_Bs[0])
                #seg_len_hists[0] =  seg_len_hists[0] + np.sum(cur_Bs[:,0] == np.arange(nb_states)[None],0)
                seg_lens = np.zeros((nb_Tracks, cur_nb_Bs, nb_locs+1, nb_states)) # dims : track ID, sequence of states ID, position in the sequence, state of the sequence
                seg_lens[:,:,1,:] = (cur_Bs[:,:,0,None] == np.arange(nb_states)[None,None]).astype(float)
            if 1:
                nb_Tracks = cur_Bs.shape[0]
                nb_locs = cur_Bs.shape[2]
                cur_nb_Bs = len(cur_Bs[0])
                cur_pos = cur_Bs[:,:,0]
                cur_seg_len = np.ones((nb_Tracks, cur_nb_Bs))
                seg_lens = np.zeros((nb_Tracks, cur_nb_Bs, nb_locs+1, nb_states)) # dims : track ID, sequence of states ID, position in the sequence, state of the sequence
                for k in range(1, nb_locs):
                    is_Tr = cur_pos != cur_Bs[:,:,k]
                    cur_seg_len = cur_seg_len + (is_Tr==0).astype(int)
                    len_before_tr = (cur_seg_len * is_Tr.astype(int))[:,:,None] # cur_seg_len * is_Tr.astype(int)) selects the segments that stops so we can add them, if 0 : not transition if n : transition after n consecutive steps
                    cat_states = (cur_pos[:,:,None] == np.arange(nb_states)[None,None]).astype(int) # position of the segment in categorcal format so it can fit seg_lens
                    seg_lens[:,:,k-1] = len_before_tr * cat_states  # add the segment len of the corresponding states 
                    cur_seg_len[is_Tr] = 1
                    cur_pos = cur_Bs[:,:,k]
                
                last_len = (nb_locs - np.sum(seg_lens, (2,3)))[:,:,None]
                cat_states = (cur_pos[:,:,None] == np.arange(nb_states)[None,None]).astype(int)
                seg_lens[:,:,-1] = last_len * cat_states
                seg_len_hist = np.zeros((nb_locs, nb_states))
                
                for k in range(1, nb_locs+1):
                    #P_seg = (P[:,:,None,None] * np.exp(-LL)[:,:,None,None] * (seg_lens == k).astype(int))
                    P_seg = (1 * (seg_lens == k).astype(int))
                    #np.sum(P_seg*np.exp(-LL)[:,:,None,None],(1,2,3))[:,None]==0
                    #P_seg = np.sum(P_seg, (1,2)) / (np.sum(P_seg,(1,2,3))[:,None]+1e-300) # normalize with regard to the proba of the track without the proba of leaving the FOV (to sum up over different track lengths) 
                    P_seg = np.sum(P_seg, (0,1,2)) # normalize with regard to the proba of the track without the proba of leaving the FOV (to sum up over different track lengths) 
                    #P_seg = np.sum(P_seg, (1,2)) / (np.sum(P_seg,(1,2)) +1e-300) # normalize with regard to the proba of the track without the proba of leaving the FOV (to sum up over different track lengths) 
                    #P_seg = np.sum(P_seg, (1,2))  # normalize with regard to the proba of the track without the proba of leaving the FOV (to sum up over different track lengths) 
                    seg_len_hist[k-1] = P_seg
            
            seg_len_hists[:seg_len_hist.shape[0]] = seg_len_hists[:seg_len_hist.shape[0]] + seg_len_hist
            
    return seg_len_hists
