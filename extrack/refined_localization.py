#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:31:38 2021

@author: francois
"""
GPU_computing = 0

import numpy as np

if GPU_computing :
    import cupy as cp
    from cupy import asnumpy
else:
    # if CPU computing :
    import numpy as cp
    def asnumpy(x):
        return cp.array(x)
try:
    from matplotlib import pyplot as plt
    import imageio
except:
    pass

#from extrack.old_tracking import extract_params, predict_Bs, P_Cs_inter_bound_stats, log_integrale_dif, first_log_integrale_dif, ds_froms_states, fuse_tracks, get_all_Bs, get_Ts_from_Bs
from extrack.tracking_0 import extract_params, predict_Bs, P_Cs_inter_bound_stats, log_integrale_dif, first_log_integrale_dif, ds_froms_states, fuse_tracks, get_all_Bs, get_Ts_from_Bs
from extrack.tracking import fuse_tracks_th
from extrack.tracking_0 import P_Cs_inter_bound_stats
from extrack.exporters import extrack_2_matrix
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def prod_2GaussPDF(sigma1,sigma2, mu1, mu2):
    sigma = ((sigma1**2*sigma2**2)/(sigma1**2+sigma2**2))**0.5
    mu = (mu1*sigma2**2 + mu2*sigma1**2)/(sigma1**2 + sigma2**2)
    LK = np.sum(-0.5*np.log(2*np.pi*(sigma1**2 + sigma2**2)) -(mu1-mu2)**2/(2*(sigma1**2 + sigma2**2)),-1)
    return sigma, mu, LK

def prod_3GaussPDF(sigma1,sigma2,sigma3, mu1, mu2, mu3):
    sigma, mu, LK = prod_2GaussPDF(sigma1,sigma2, mu1, mu2)
    sigma, mu, LK2 = prod_2GaussPDF(sigma,sigma3, mu, mu3)
    LK = LK + LK2
    return sigma, mu, LK

def gaussian(x, sig, mu):
    return np.product(1/(2*np.pi*sig**2)**0.5 * np.exp(-(x-mu)**2/(2*sig**2)), -1)

def get_LC_Km_Ks(Cs, LocErr, ds, Fs, TrMat, nb_substeps=1, frame_len = 4, threshold = 0.2, max_nb_states = 1000):
    '''
    variation of the main function to extract LC, Km and Ks for all positions
    '''
    nb_Tracks = len(Cs)
    nb_locs = len(Cs[0]) # number of localization per track
    nb_dims = len(Cs[0,0]) # number of spatial dimentions (x, y) or (x, y, z)
    Cs = Cs.reshape((nb_Tracks,1,nb_locs, nb_dims))
    Cs = cp.array(Cs)
   
    nb_states = TrMat.shape[0]
    all_Km = []
    all_Ks = []
    all_LP = []
    all_cur_Bs = []
   
    LocErr = LocErr[:,None]
    LocErr = LocErr[:,:,::-1] # useful when single peak localization error is inputed,
    LocErr2 = LocErr**2
    if LocErr.shape[2] == 1: # combined to min(LocErr_index, nb_locs-current_step) it will select the right index
        LocErr_index = -1
    elif LocErr.shape[2] == nb_locs:
        LocErr_index = nb_locs
    else:
        raise ValueError("Localization error is not specified correctly, in case of unique localization error specify a float number in estimated_vals['LocErr'].\n If one localization error per dimension, specify a list or 1D array of elements the localization error for each dimension.\n If localization error is predetermined by another method for each position the argument input_LocErr should be a dict for each track length of the 3D arrays corresponding to all_tracks (can be obtained from the reader functions using the opt_colname argument)")
   
    preds = np.zeros((nb_Tracks, nb_locs, nb_states))-1
   
    TrMat = cp.array(TrMat) # transition matrix of the markovian process
    current_step = 1
   
    cur_Bs = get_all_Bs(nb_substeps + 1, nb_states)[None] # get initial sequences of states
    cur_Bs_cat = (cur_Bs[:,:,:,None] == np.arange(nb_states)[None,None,None,:]).astype('float64')
   
    cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int) #states of interest for the current displacement
    cur_nb_Bs = cur_Bs.shape[1]
   
    sub_Bs = get_all_Bs(nb_substeps, nb_states)[None]
    TrMat = cp.array(TrMat.T)
   
    # compute the vector of diffusion stds knowing the current states
    ds = cp.array(ds)
    ds2 = ds**2
    Fs = cp.array(Fs)
   
    LT = get_Ts_from_Bs(cur_states, TrMat) # Log proba of transitions per step
    #LF = cp.log(Fs[cur_states[:,:,-1]]) # Log proba of finishing/starting in a given state (fractions)
   
    LP = LT # current log proba of seeing the track
    LP = cp.repeat(LP, nb_Tracks, axis = 0)
       
    cur_d2s = ds2[cur_states]
    cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps

    # we can average the variances of displacements per step to get the actual std of displacements per step
    cur_d2s = cp.mean(cur_d2s, axis = 2)
    cur_d2s = cur_d2s[:,:,None]
    cur_d2s = cp.array(cur_d2s)
   
    # inject the first position to get the associated Km and Ks :
    Km, Ks = first_log_integrale_dif(Cs[:,:, nb_locs - current_step], LocErr2[:,:,min(LocErr_index,nb_locs-current_step)], cur_d2s)
    
    if len(Ks)==1:
        Ks = cp.repeat(Ks, nb_Tracks, axis = 0)
   
    current_step += 1
    Km = cp.repeat(Km, cur_nb_Bs, axis = 1)
    removed_steps = 0
   
    all_Km.append(Km)
    all_Ks.append(Ks**0.5)
    all_LP.append(LP)
    all_cur_Bs.append(cur_Bs)
   
    while current_step <= nb_locs-1:
        # update cur_Bs to describe the states at the next step :
        for iii in range(nb_substeps):
            #cur_Bs = np.concatenate((np.repeat(np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], nb_Tracks, 0), np.repeat(cur_Bs,nb_states,1)),-1)
            cur_Bs = np.concatenate((np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], np.repeat(cur_Bs,nb_states,1)),-1)
            new_states = np.repeat(np.mod(np.arange(cur_Bs_cat.shape[1]*nb_states, dtype = 'int8'),nb_states)[None,:,None,None] == np.arange(nb_states, dtype = 'int8')[None,None,None], cur_Bs_cat.shape[0], 0).astype('int8')
            cur_Bs_cat = np.concatenate((new_states, np.repeat(cur_Bs_cat,nb_states,1)),-2)
       
        cur_states = cur_Bs[:1,:,0:nb_substeps+1].astype(int)
        # compute the vector of diffusion stds knowing the states at the current step
       
        cur_d2s = ds2[cur_states]
        cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps
        cur_d2s = cp.mean(cur_d2s, axis = 2)
        cur_d2s = cur_d2s[:,:,None]
        cur_d2s = cp.array(cur_d2s)
       
        LT = get_Ts_from_Bs(cur_states, TrMat)
       
        # repeat the previous matrix to account for the states variations due to the new position
        Km = cp.repeat(Km, nb_states**nb_substeps , axis = 1)
        Ks = cp.repeat(Ks, nb_states**nb_substeps, axis = 1)
        LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)
        # inject the next position to get the associated Km, Ks and Constant describing the integral of 3 normal laws :
        Km, Ks, LC = log_integrale_dif(Cs[:,:,nb_locs-current_step], LocErr2[:,:,min(LocErr_index,nb_locs-current_step)], cur_d2s, Km, Ks)
        
        #print('integral',time.time() - t0); t0 = time.time()
        LP += LT + LC # current (log) constants associated with each track and sequences of states
        del LT, LC
        cur_nb_Bs = len(cur_Bs[0]) # current number of sequences of states

        cur_nb_Bs = len(cur_Bs[0]) # current number of sequences of states
       
        if cur_nb_Bs > max_nb_states:
            threshold = threshold*1.2
       
        '''idea : the position and the state 6 steps ago should not impact too much the
        probability of the next position so the Km and Ks of tracks with the same 6 last
        states must be very similar, we can then fuse the parameters of the pairs of Bs
        which vary only for the last step (7) and sum their probas'''
       
       
        if current_step < nb_locs-1: # do not fuse sequences at the last step as it doesn't improves speed.            
            Km, Ks, LP, cur_Bs, cur_Bs_cat = fuse_tracks_th(Km,
                                                            Ks,
                                                            LP,
                                                            cur_Bs,
                                                            cur_Bs_cat,
                                                            nb_Tracks,
                                                            nb_states = nb_states,
                                                            nb_dims = nb_dims,
                                                            do_preds = 1,
                                                            threshold = threshold,
                                                            frame_len = frame_len) # threshold on values normalized by sigma.
           
            cur_nb_Bs = len(cur_Bs[0])
            #print(current_step, m_arr.shape)
            removed_steps += 1
           
        current_step += 1
        #print(current_step)
        all_Km.append(Km)
        all_Ks.append(Ks**0.5)
        all_LP.append(LP)
        all_cur_Bs.append(cur_Bs)
       
    newKs =  cp.array((Ks + LocErr2[:,:, min(LocErr_index, nb_locs-current_step)]))
    log_integrated_term = cp.sum(-0.5*cp.log(2*np.pi*newKs) - (Cs[:,:,0] - Km)**2/(2*newKs),axis=2)
    LF = cp.log(Fs[cur_Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
    #LF = cp.log(0.5)
    # cp.mean(cp.log(Fs[cur_Bs[:,:,:].astype(int)]), 2) # Log proba of starting in a given state (fractions)
    LP += log_integrated_term + LF
   
    pred_LP = LP
    if cp.max(LP)>600: # avoid overflow of exponentials, mechanically also reduces the weight of longest tracks
        pred_LP = LP - (cp.max(LP)-600)
   
    P = np.exp(pred_LP)
    sum_P = np.sum(P, axis = 1, keepdims = True)[:,:,None]
    preds = np.sum(P[:,:,None,None]*cur_Bs_cat, axis = 1) / sum_P
    preds = preds[:,::-1]
   
    return LP, cur_Bs, all_cur_Bs, preds, all_Km, all_Ks, all_LP

# LocErr = cur_LocErr
def get_pos_PDF(Cs, LocErr, ds, Fs, TrMat, frame_len = 7, threshold = 0.2, max_nb_states = 1000):
    nb_substeps = 1 
    ds = cp.array(ds)
    Cs = cp.array(Cs)
    
    LocErr.shape
    # get Km, Ks and LC forward
    LP1, final_Bs1, all_cur_Bs1, preds1, all_Km1, all_Ks1, all_LP1 = get_LC_Km_Ks(Cs, LocErr, ds, Fs, TrMat, nb_substeps, frame_len, threshold, max_nb_states)
    #get Km, Ks and LC backward
    TrMat2 = np.copy(TrMat).T # transpose the matrix for the backward transitions
    Cs2 = Cs[:,::-1,:] # inverse the time steps
    LP2, final_Bs2, all_cur_Bs2, preds2, all_Km2, all_Ks2, all_LP2 = get_LC_Km_Ks(Cs2, LocErr, ds, cp.ones(TrMat2.shape[0],)/TrMat2.shape[0], TrMat2, nb_substeps, frame_len, threshold, max_nb_states) # we set a neutral Fs so it doesn't get counted twice
    
    # do the approximation for the first position, product of 2 gaussian PDF, (integrated term and localization error)    
    sig, mu, LC = prod_2GaussPDF(LocErr[:,None,0], all_Ks1[-1], Cs[:,None,0], all_Km1[-1])
    
    LP = all_LP1[-1] + LC
    all_pos_means = [mu]
    all_pos_stds = [sig]
    all_pos_weights = [LP]
    all_pos_Bs = [final_Bs1]
   
    for k in range(1,Cs.shape[1]-1):
        '''
        we take the corresponding Km1, Ks1, LP1, Km2, Ks2, LP2
        which are the corresponding stds and means of the resulting
        PDF surrounding the position k
        with localization uncertainty, we have 3 gaussians to compress to 1 gaussian * K
        This has to be done for all combinations of set of consective states before and after.step k
        to do so we set dim 1 as dim for consecutive states computed by the forward proba and
        dim 2 for sets of states computed by the backward proba.
        '''
        LP1 = all_LP1[-1-k][:,:,None]
        Km1 = all_Km1[-1-k][:,:,None]
        Ks1 = all_Ks1[-1-k][:,:,None]
        cur_Bs1 = all_cur_Bs1[-1-k][0,:,0]
        
        LP2 = all_LP2[k-1][:,None]
        Km2 = all_Km2[k-1][:,None]
        Ks2 = all_Ks2[k-1][:,None]
        cur_Bs2 = all_cur_Bs2[k-1][0,:,0]
       
        nb_Bs1 = Ks1.shape[1]
        nb_Bs2 = Ks2.shape[2]
        nb_tracks = Km1.shape[0]
        nb_dims = Km1.shape[3]
        nb_states = TrMat.shape[0]
       
        mu = np.zeros((nb_tracks, 0, Km2.shape[-1]))
        sig = np.zeros((nb_tracks, 0, Ks2.shape[-1]))
        LP = np.zeros((nb_tracks, 0))
       
        Bs2_len = np.min([k+1, frame_len-1])
        # we must reorder the metrics so the Bs from the backward terms correspond to the forward terms
        for state in range(nb_states):
            
            sub_Ks1 = Ks1[:,np.where(cur_Bs1==state)[0]]
            sub_Ks2 = Ks2[:,:,np.where(cur_Bs2==state)[0]]
            sub_Km1 = Km1[:,np.where(cur_Bs1==state)[0]]
            sub_Km2 = Km2[:,:,np.where(cur_Bs2==state)[0]]
            sub_LP1 = LP1[:,np.where(cur_Bs1==state)[0]]
            sub_LP2 = LP2[:,:,np.where(cur_Bs2==state)[0]]
           
            if LocErr.shape[1]>1:
                cur_LocErr = LocErr[:,None,None,k]
            else:
                cur_LocErr = LocErr[:,None]
           
            sub_sig, sub_mu, sub_LC = prod_3GaussPDF(sub_Ks1, cur_LocErr, sub_Ks2, sub_Km1, Cs[:,None,None,k], sub_Km2)
            
            sub_LP = sub_LP1 + sub_LP2 + sub_LC
           
            sub_sig = sub_sig.reshape((nb_tracks,sub_sig.shape[1]*sub_sig.shape[2],1))
            sub_mu = sub_mu.reshape((nb_tracks,sub_mu.shape[1]*sub_mu.shape[2], nb_dims))
            sub_LP = sub_LP.reshape((nb_tracks,sub_LP.shape[1]*sub_LP.shape[2]))
           
            mu = np.concatenate((mu, sub_mu), axis = 1)
            sig = np.concatenate((sig, sub_sig), axis = 1)
            LP = np.concatenate((LP, sub_LP), axis = 1)
        
        all_pos_means.append(mu)
        all_pos_stds.append(sig)
        all_pos_weights.append(LP)
    
    sig, mu, LC = prod_2GaussPDF(LocErr[:,None,-1], all_Ks2[-1], Cs[:,None,-1], all_Km2[-1])
    LP = all_LP2[-1] + LC
    
    all_pos_means.append(mu)
    all_pos_stds.append(sig)
    all_pos_weights.append(LP)

    return all_pos_means, all_pos_stds, all_pos_weights

# all_tracks = tracks
# LocErr = input_LocErr
# LocErr = LocErr[0]

def position_refinement(all_tracks, LocErr, ds, Fs, TrMat, frame_len = 7, threshold = 0.1, max_nb_states = 1000):
    if type(LocErr) == float or type(LocErr) == np.float64 or type(LocErr) == np.float32:
        LocErr = np.array([[[LocErr]]])
        LocErr_type = 'array'
        cur_LocErr = LocErr

    elif type(LocErr) == np.ndarray:
        LocErr_type = 'array'
        if len(LocErr.shape) == 1:
            LocErr = LocErr[None, None]
        if len(LocErr.shape) == 2:
            LocErr = LocErr[None]
        cur_LocErr = LocErr

    elif type(LocErr) == dict:
        LocErr_type = 'dict'
    else:
        LocErr_type = 'other'
    print('LocErr_type', LocErr_type)
    
    all_mus = {}
    all_sigmas = {}
    for l in all_tracks.keys():
        Cs = all_tracks[l]
        if LocErr_type == 'dict':
            cur_LocErr = LocErr[l]
        all_mus[l] = np.zeros((Cs.shape[0], int(l), Cs.shape[2]))
        all_sigmas[l] = np.zeros((Cs.shape[0], int(l)))
        all_pos_means, all_pos_stds, all_pos_weights = get_pos_PDF(Cs, cur_LocErr, ds, Fs, TrMat, frame_len, threshold, max_nb_states)
        #best_mus, best_sigs, best_Bs = get_all_estimates(all_pos_weights, all_pos_Bs, all_pos_means, all_pos_stds)
        for k, (pos_means, pos_stds, pos_weights) in enumerate(zip(all_pos_means, all_pos_stds, all_pos_weights)):
            P = np.exp(pos_weights - np.max(pos_weights, 1, keepdims = True))
            all_mus[l][:, k] = np.sum(P[:,:,None]*pos_means, 1) / np.sum(P, 1)[:,None]
            all_sigmas[l][:, k] = (np.sum(P[:,:]*pos_stds[:,:,0]**2, 1) / np.sum(P, 1))**0.5
    return all_mus, all_sigmas

def get_all_estimates(all_pos_weights, all_pos_Bs, all_pos_means, all_pos_stds):
    nb_Bs = []
    nb_pos = len(all_pos_weights)
    for weights in all_pos_weights:
        nb_Bs.append(weights.shape[1])
    nb_Bs = np.max(nb_Bs)
    nb_states = (np.max(all_pos_Bs[0])+1).astype(int)
    max_frame_len = (np.log(nb_Bs) // np.log(nb_states)).astype(int)
    mid_frame_pos = ((max_frame_len-0.1)//2).astype(int) # -0.1 is used for mid_frame_pos to be the good index for both odd and pair numbers
    mid_frame_pos = np.max([1,mid_frame_pos])
    best_Bs = []
    best_mus = []
    best_sigs = []
    for k, (weights, Bs, mus, sigs)  in enumerate(zip(all_pos_weights, all_pos_Bs, all_pos_means, all_pos_stds)):
        if k <= nb_pos/2 :
            idx = np.min([k, mid_frame_pos])
        else:
            idx = np.max([-mid_frame_pos-1, k - nb_pos])
        best_args = np.argmax(weights, 1)
        best_Bs.append(Bs[0,best_args][:,idx])
        best_sigs.append(sigs[[cp.arange(len(mus)), best_args]])
        best_mus.append(mus[[cp.arange(len(mus)), best_args]])
    best_Bs = cp.array(best_Bs).T.astype(int)
    best_sigs = cp.transpose(cp.array(best_sigs), (1,0,2))
    best_mus = cp.transpose(cp.array(best_mus), (1,0,2))
    return asnumpy(best_mus), asnumpy(best_sigs), asnumpy(best_Bs)

def save_gifs(Cs, all_pos_means, all_pos_stds, all_pos_weights, all_pos_Bs, gif_pathnames = './tracks', lim = None, nb_pix = 200, fps=1):
    try:
        plt
        imageio
    except:
        raise ImportError('matplotlib and imageio has to be installed to use save_gifs')
    best_mus, best_sigs, best_Bs = get_all_estimates(all_pos_weights, all_pos_Bs, all_pos_means, all_pos_stds)
    for ID in range(len(Cs)):
        all_images = []
        Cs_offset = np.mean(Cs[ID], 0)
        Cs[ID] = Cs[ID]
       
        if lim == None:
            cur_lim = np.max(np.abs(Cs[ID]))*1.1
        else:
            cur_lim = lim
        pix_size = nb_pix / (2*cur_lim)
        for k in range(len(all_pos_means)):
               
            sig = asnumpy(all_pos_stds[k])
            mu =  asnumpy(all_pos_means[k])
            LP =  asnumpy(all_pos_weights[k])
            
            fig = plt.figure()            
            plt.plot((Cs[ID, :,1] - Cs_offset[1] + cur_lim)*pix_size-0.5, (Cs[ID, :,0]- Cs_offset[0]+cur_lim)*pix_size-0.5)
            plt.scatter((best_mus[ID, :,1] - Cs_offset[1] + cur_lim)*pix_size-0.5, (best_mus[ID, :,0] - Cs_offset[0]+cur_lim)*pix_size-0.5, c='r', s=3)

            P_xs = gaussian(np.linspace(-cur_lim,cur_lim,nb_pix)[None,:,None], sig[ID][:,:,None], mu[ID][:,:1,None] - Cs_offset[0]) * np.exp(LP[ID]-np.max(LP[ID]))[:,None]
            P_ys = gaussian(np.linspace(-cur_lim,cur_lim,nb_pix)[None,:,None], sig[ID][:,:,None], mu[ID][:,1:,None] - Cs_offset[1]) * np.exp(LP[ID]-np.max(LP[ID]))[:,None]
           
            heatmap = np.sum(P_xs[:,:,None]*P_ys[:,None] * np.exp(LP[ID]-np.max(LP[ID]))[:,None,None],0)
           
            heatmap = heatmap/np.max(heatmap)
            plt.imshow(heatmap)
            plt.xticks(np.linspace(0,nb_pix-1, 5), np.round(np.linspace(-cur_lim,cur_lim, 5), 2))
            plt.yticks(np.linspace(0,nb_pix-1, 5), np.round(np.linspace(-cur_lim,cur_lim, 5), 2))
            canvas = FigureCanvas(fig)
            canvas.draw()
            s, (width, height) = canvas.print_to_buffer()
            image = np.fromstring(s, dtype='uint8').reshape((height, width, 4))
           
            all_images.append(image)
            plt.close()
       
        imageio.mimsave(gif_pathnames + str(ID)+'.gif', all_images,fps=fps)


def get_LC_Km_Ks_fixed_Bs(Cs, LocErr, ds, Fs, TrMat, Bs):
    '''
    variation of the main function to extract LC, Km and Ks for all positions
    '''
    nb_Tracks = len(Cs)
    nb_locs = len(Cs[0]) # number of localization per track
    nb_dims = len(Cs[0,0]) # number of spatial dimentions (x, y) or (x, y, z)
    Cs = Cs.reshape((nb_Tracks,1,nb_locs, nb_dims))
    Cs = cp.array(Cs)
   
    all_Km = []
    all_Ks = []
    all_LP = []
       
    TrMat = cp.array(TrMat) # transition matrix of the markovian process
    current_step = 1
   
    cur_states = Bs[:,:,-2:].astype(int) #states of interest for the current displacement
   
    # compute the vector of diffusion stds knowing the current states
    ds = cp.array(ds)
    Fs = cp.array(Fs)
   
    LT = get_Ts_from_Bs(cur_states, TrMat) # Log proba of transitions per step
    LF = cp.log(Fs[cur_states[:,:,-1]]) # Log proba of finishing/starting in a given state (fractions)
   
    LP = LT # current log proba of seeing the track
    LP = cp.repeat(LP, nb_Tracks, axis = 0)
   
    cur_ds = ds_froms_states(ds, cur_states)
   
    # inject the first position to get the associated Km and Ks :
    Km, Ks = first_log_integrale_dif(Cs[:,:, nb_locs-current_step], LocErr, cur_ds)
    all_Km.append(Km)
    all_Ks.append(Ks)
    all_LP.append(LP)
    current_step += 1
   
    while current_step <= nb_locs-1:
        # update cur_Bs to describe the states at the next step :
        cur_states = Bs[:,:,-current_step-1:-current_step+1].astype(int)
        # compute the vector of diffusion stds knowing the states at the current step
        cur_ds = ds_froms_states(ds, cur_states)
        LT = get_Ts_from_Bs(cur_states, TrMat)

        # inject the next position to get the associated Km, Ks and Constant describing the integral of 3 normal laws :
        Km, Ks, LC = log_integrale_dif(Cs[:,:,nb_locs-current_step], LocErr, cur_ds, Km, Ks)
        #print('integral',time.time() - t0); t0 = time.time()
        LP += LT + LC # current (log) constants associated with each track and sequences of states
        del LT, LC

        all_Km.append(Km)
        all_Ks.append(Ks)
        all_LP.append(LP)
       
        current_step += 1

    newKs = cp.array((Ks**2 + LocErr**2)**0.5)[:,:,0]
    log_integrated_term = -cp.log(2*np.pi*newKs**2) - cp.sum((Cs[:,:,0] - Km)**2,axis=2)/(2*newKs**2)
    LF = cp.log(Fs[Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
    #LF = cp.log(0.5)
    # cp.mean(cp.log(Fs[cur_Bs[:,:,:].astype(int)]), 2) # Log proba of starting in a given state (fractions)
    LP += log_integrated_term + LF

    all_Ks = cp.array(all_Ks)[:,0,0]
    all_Km = cp.array(all_Km)[:,0,0]
    all_LP = cp.array(all_LP)[:,0,0]
    return all_Km, all_Ks, all_LP

def get_pos_PDF_fixedBs(Cs, LocErr, ds, Fs, TrMat, Bs):
    '''
    get mu and sigma for each position given inputed Bs,
    ideally used for a single track with its most likely set of states
    '''
    ds = np.array(ds)
    Cs = cp.array(Cs)
    # get Km, Ks and LC forward
    all_Km1, all_Ks1, all_LP1 =  get_LC_Km_Ks_fixed_Bs(Cs, LocErr, ds, Fs, TrMat, Bs)
    #get Km, Ks and LC backward
    TrMat2 = np.copy(TrMat).T # transpose the matrix for the backward transitions
    Cs2 = Cs[:,::-1,:] # inverse the time steps
    all_Km2, all_Ks2, all_LP2 = get_LC_Km_Ks_fixed_Bs(Cs2, LocErr, ds, cp.ones(TrMat2.shape[0],)/TrMat2.shape[0], TrMat2, Bs[:,:,::-1])
    # do the approximation for the first position, product of 2 gaussian PDF, (integrated term and localization error)    

    sig, mu, LC = prod_2GaussPDF(LocErr,all_Ks1[-1], Cs[:,0], all_Km1[-1])
    np
    all_pos_means = [mu]
    all_pos_stds = [sig[None]]
    
    for k in range(1,Cs.shape[1]-1):

        Km1 = all_Km1[-k][None]
        Ks1 = all_Ks1[-1-k][None]
        Km2 = all_Km2[k-1][None]
        Ks2 = all_Ks2[k-1][None]
       
        sig, mu, LC = prod_3GaussPDF(Ks1,LocErr,Ks2, Km1, Cs[:,k], Km2)

        all_pos_means.append(mu)
        all_pos_stds.append(sig)
       
    sig, mu, LC = prod_2GaussPDF(LocErr,all_Ks2[-1], Cs[:,-1], all_Km2[-1])
   
    all_pos_means.append(mu)
    all_pos_stds.append(sig[None])
    return cp.array(all_pos_means)[:,0], cp.array(all_pos_stds)[:,0]

def get_global_sigs_mus(all_pos_means, all_pos_stds, all_pos_weights, idx = 0):
    w_sigs = []
    w_mus = []
    for mus, sigs, LC in zip(all_pos_means, all_pos_stds, all_pos_weights):
        mus = mus[idx]
        sigs = sigs[idx]
        LC = LC[idx]
        LC = LC - np.max(LC, keepdims = True)
        #sigs = sigs[LC > np.max(LC) + np.log(1e-5)] # remove the unlikely set of states
        #mus = mus[LC > np.max(LC) + np.log(1e-5)] # remove the unlikely set of states
        #LC = LC[LC > np.max(LC) + np.log(1e-5)] # remove the unlikely set of states
        w_sigs.append(np.sum(np.exp(LC[:,None])**2 * sigs) / np.sum(np.exp(LC[:,None])**2))
        w_mus.append(np.sum(np.exp(LC[:,None]) * mus,0) / np.sum(np.exp(LC[:,None]),0))
    return np.array(w_mus), np.array(w_sigs)

def full_extrack_2_matrix(all_tracks, params, dt, all_frames = None, cell_dims = [1,None,None], nb_states = 2, frame_len = 15):
    nb_dims = list(all_tracks.items())[0][1].shape[2]
    pred_Bss = predict_Bs(all_tracks, dt, params, nb_states=nb_states, frame_len=frame_len, cell_dims = cell_dims)
   
    DATA = extrack_2_matrix(all_tracks, pred_Bss, dt, all_frames = all_frames)
    DATA = np.concatenate((DATA, np.empty((DATA.shape[0], nb_dims+1))),1)
    LocErr, ds, Fs, TrMat = extract_params(params, dt, nb_states, nb_substeps = 1)
    for ID in np.unique(DATA[:,2]):
        track = DATA[DATA[:,2]==ID,:nb_dims][None]
        all_pos_means, all_pos_stds, all_pos_weights, all_pos_Bs = get_pos_PDF(track, LocErr, ds, Fs, TrMat, frame_len = frame_len//2+3)
        w_mus, w_sigs = get_global_sigs_mus(all_pos_means, all_pos_stds, all_pos_weights, idx = 0)
        DATA[DATA[:,2]==ID,-1] = w_sigs
        DATA[DATA[:,2]==ID,-nb_dims-1:-1] = w_mus
    return DATA

def get_best_estimates(Cs, LocErr, ds, Fs, TrMat, frame_len = 10):
    all_mus = []
    all_sigs = []
    for track in Cs:
        a,b, preds = P_Cs_inter_bound_stats(track[None], LocErr, ds, Fs, TrMat, nb_substeps=1, do_frame = 1, frame_len = frame_len, do_preds = 1)
        Bs = np.argmax(preds, 2)[None]
        mus, sigs = get_pos_PDF_fixedBs(Cs, LocErr, ds, Fs, TrMat, Bs)
    all_mus.append(mus)
    all_sigs.append(sigs)
    return mus, sigs

def do_gifs_from_params(all_tracks, params, dt, gif_pathnames = './tracks', frame_len = 9, nb_states = 2, nb_pix = 200, fps = 1):
    for Cs in all_tracks:
        LocErr, ds, Fs, TrMat = extract_params(params, dt, nb_states, nb_substeps = 1)
        all_pos_means, all_pos_stds, all_pos_weights, all_pos_Bs = get_pos_PDF(Cs, LocErr, ds, Fs, TrMat, frame_len = frame_len)
        save_gifs(Cs, all_pos_means, all_pos_stds, all_pos_weights, all_pos_Bs, gif_pathnames = gif_pathnames + '_' + str(len(Cs[0])) + '_pos', lim = None, nb_pix = nb_pix, fps=fps)    
