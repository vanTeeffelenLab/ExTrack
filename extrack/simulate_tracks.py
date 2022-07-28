#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:16:58 2021

@author: mcm
"""

import numpy as np

def markovian_process(TrMat, initial_fractions, nb_tracks, track_len):
    nb_states = len(TrMat)
    cumMat = np.cumsum(TrMat, 1)
    cumF = np.cumsum(initial_fractions)
    states = np.zeros((nb_tracks, track_len)).astype(int)
    randoms = np.random.rand(nb_tracks, track_len)
    for s in range(nb_states -1):
        states[:,0] += (randoms[:,0]>cumF[s]).astype(int)
    for k in range(1,track_len):
        for s in range(nb_states -1):
            states[:,k] += (randoms[:,k] > cumMat[states[:,k-1]][:,s]).astype(int)
    return states

def get_fractions_from_TrMat(TrMat):
    if len(TrMat) == 2:
        p01 = TrMat[0,1]
        p10 = TrMat[1,0]
        initial_fractions = np.array([p10 / (p10+p01), p01 / (p10+p01)])
    elif len(TrMat) == 3:
        p01 = TrMat[0,1]
        p02 = TrMat[0,2]
        p10 = TrMat[1,0]
        p12 = TrMat[1,2]
        p20 = TrMat[2,0]
        p21 = TrMat[2,1]
        F0 = (p10*(p21+p20)+p20*p12)/((p01)*(p12 + p21) + p02*(p10 + p12 + p21) + p01*p20 + p21*p10 + p20*(p10+p12))
        F1 = (F0*p01 + (1-F0)*p21)/(p10 + p12 + p21)
        initial_fractions =  np.array([F0, F1, 1-F0-F1])
    else:
        '''
        if more states lets just run the transition process until equilibrium
        '''
        A0 = np.ones(len(TrMat))/len(TrMat)
        k = 0
        prev_A = A0
        A = np.dot(A0, TrMat)
        while not np.all(prev_A == A):
            k += 1
            prev_A = A
            A = np.dot(A, TrMat)
            if k > 10000000:
                raise Exception("We could't find a steady state, convergence didn't occur within %s steps, \ncurrent fractions : %s"%(k, A))
        initial_fractions = A
    return initial_fractions

def sim_noBias(track_lengths = [7,8,9,10,11], # create arrays of tracks of specified number of localizations  
               track_nb_dist = [1000, 800, 700, 600, 550], # list of number of tracks per array
               LocErr = 0.02, # Localization error in um
               Ds = [0, 0.05], # diffusion coef of each state in um^2/s
               TrMat = np.array([[0.9,0.1], [0.2,0.8]]), # transition matrix e.g. np.array([[1-p01,p01], [p10,1-p10]])
               initial_fractions = None, # fraction in each states at the beginning of the tracks, if None assums steady state determined by TrMat
               dt = 0.02, # time in between positions in s
               nb_dims = 2 # number of spatial dimensions : 2 for (x,y) and 3 for (x,y,z)
               ):
    '''
    create tracks with specified kinetics
    outputs the tracks and the actual hidden states using the ExTrack format (dict of np.arrays)
    any states number 'n' can be producted as long as len(Ds) = len(initial_fractions) and TrMat.shape = (n,n)
    '''
    
    Ds = np.array(Ds)
    TrMat = np.array(TrMat)
    nb_sub_steps = 30
    
    if initial_fractions is None:
        initial_fractions = get_fractions_from_TrMat(TrMat)
        
    nb_states = len(TrMat)
    sub_dt = dt/nb_sub_steps
    
    TrSubMat = TrMat / nb_sub_steps
    TrSubMat[np.arange(nb_states),np.arange(nb_states)] = 0
    TrSubMat[np.arange(nb_states),np.arange(nb_states)] = 1 - np.sum(TrSubMat,1)
    
    all_Css = []
    all_Bss = []
    
    for nb_tracks, track_len in zip(track_nb_dist, track_lengths):
        print(nb_tracks, track_len)
        
        states = markovian_process(TrSubMat,
                                   initial_fractions = initial_fractions, 
                                   track_len =(track_len-1) * nb_sub_steps + 1,
                                   nb_tracks = nb_tracks)
        # determines displacements then MSDs from stats
        positions = np.random.normal(0, 1, (nb_tracks, (track_len-1) * nb_sub_steps+1, nb_dims)) * np.sqrt(2*Ds*sub_dt)[states[:,:, None]]
        positions = np.cumsum(positions, 1)

        positions = positions + np.random.normal(0, LocErr, (nb_tracks, (track_len-1) * nb_sub_steps + 1, nb_dims))
        
        all_Css.append(positions[:,np.arange(0,(track_len-1) * nb_sub_steps +1, nb_sub_steps)])
        all_Bss.append(states[:,np.arange(0,(track_len-1) * nb_sub_steps +1, nb_sub_steps)])
    
    all_Css_dict = {}
    all_Bss_dict = {}
    for Cs, Bs in zip(all_Css, all_Bss):
        l = str(Cs.shape[1])
        all_Css_dict[l] = Cs
        all_Bss_dict[l] = Bs
        
    return all_Css_dict, all_Bss_dict

def is_in_FOV(positions, cell_dims):
    inFOV = np.ones((len(positions)+1,)) == 1
    for i, l in enumerate(cell_dims):
        if not l == None:
            cur_inFOV = (positions[:,i] < l) * (positions[:,i] > 0)
            cur_inFOV = np.concatenate((cur_inFOV,[False]))
            inFOV = inFOV*cur_inFOV
    return inFOV


def sim_FOV(nb_tracks=10000,
            max_track_len=40,
            min_track_len = 2,
            LocErr=0.02, # localization error in x, y and z (even if not used)
            Ds = np.array([0,0.05]),
            nb_dims = 2,
            initial_fractions = np.array([0.6,0.4]),
            TrMat = np.array([[0.9,0.1],[0.1,0.9]]),
            LocErr_std = 0,
            dt = 0.02,
            pBL = 0.1, 
            cell_dims = [0.5,None,None]): # dimension limits in x, y and z respectively
    '''
    simulate tracks able to come and leave from the field of view :
    
    nb_tracks: number of tracks simulated.
    max_track_len: number of steps simulated per track.
    LocErr: standard deviation of the localization error.
    Ds: 1D array of the diffusion coefs for each state.
    TrMat: transition array per step (lines: state at time n, cols: states at time n+1).
    dt: time in between frames.
    pBL: probability of bleaching per step.
    cell_dims: dimension limits in x, y and z respectively. x, y dimension limits are useful when tracking membrane proteins in tirf when the particles leave the field of view from the sides of the cells. z dimension is relevant for cytoplasmic proteins which call leave from the z axis. Consider the particle can leave from both ends of each axis: multiply axis limit by 2 to aproximate tracks leaving from one end.
    min_track_len: minimal track length for the track to be considered.
    
    outputs:
    all_tracks: dict describing the tracks with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = x, y position.
    all_Bs: dict descibing the true states of tracks with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = x, y position.    
    '''

    if type(LocErr) != np.ndarray:
        LocErr = np.array(LocErr)
        if LocErr.shape == ():
            LocErr = LocErr[None]
    
    nb_sub_steps = 20
    nb_strobo_frames = 1
    nb_states = len(TrMat)
    sub_dt = dt/nb_sub_steps
    cell_dims = np.array(cell_dims)
    
    TrSubMat = TrMat / nb_sub_steps
    TrSubMat[np.arange(nb_states),np.arange(nb_states)] = 0
    TrSubMat[np.arange(nb_states),np.arange(nb_states)] = 1 - np.sum(TrSubMat,1)
    
    all_Css = [[]]*(max_track_len - min_track_len + 1)
    all_Bss = [[]]*(max_track_len - min_track_len + 1)
    all_sigs = [[]]*(max_track_len - min_track_len + 1)
    
    nb_tracks = 2**np.sum(cell_dims!=None)*nb_tracks
    states = markovian_process(TrSubMat, initial_fractions, nb_tracks, (max_track_len) * nb_sub_steps)
    cell_dims0 = np.copy(cell_dims)
    cell_dims0[cell_dims0==None] = 1
    
    for state in states:

        cur_track_len = max_track_len
        positions = np.zeros(((max_track_len) * nb_sub_steps, 3))
        positions[0,:] = 2*np.random.rand(3)*cell_dims0-cell_dims0
        positions[1:] = np.random.normal(0, 1, ((max_track_len) * nb_sub_steps - 1, 3))* np.sqrt(2*Ds*sub_dt)[state[:-1, None]]
        state = state[np.arange(0,(max_track_len-1) * nb_sub_steps +1, nb_sub_steps)]
        positions = np.cumsum(positions, 0)
        positions = positions.reshape((max_track_len, nb_sub_steps, 3))
        positions = positions[:,:nb_strobo_frames]
        positions = np.mean(positions,axis = 1)
        
        inFOV =  is_in_FOV(positions, cell_dims)
        inFOV = np.concatenate((inFOV,[False]))
        while np.any(inFOV):
            if inFOV[0] == False:
                positions = positions[np.argmax(inFOV):]
                state = state[np.argmax(inFOV):]
                inFOV = inFOV[np.argmax(inFOV):]

            cur_sub_track = positions[:np.argmin(inFOV)]
            cur_sub_state = state[:np.argmin(inFOV)]
            
            pBLs = np.random.rand((len(cur_sub_track)))
            pBLs = pBLs < pBL
            if not np.all(pBLs==0):
                cur_sub_track = cur_sub_track[:np.argmax(pBLs)+1]
                cur_sub_state = cur_sub_state[:np.argmax(pBLs)+1]
                inFOV = np.array([False])
            
            k = 2 / (LocErr_std**2+1e-20) # compute the parameter of the chi2 distribution with results in the specified standard deviation
            cur_sub_sigmas = np.random.chisquare(k, (len(cur_sub_track), 3)) * LocErr[None] / k
            cur_sub_errors = np.random.normal(0, cur_sub_sigmas, (len(cur_sub_track), 3))
            cur_real_pos = np.copy(cur_sub_track)
            cur_sub_track = cur_sub_track + cur_sub_errors
            #cur_sub_track = cur_sub_track + np.random.normal(0, std_spurious, (len(cur_sub_track), 3)) * (np.random.rand(len(cur_sub_track),1)<p_spurious)
    
            arg_add = np.argwhere(np.arange(min_track_len,max_track_len+1)==len(cur_sub_track))
            
            for a in arg_add:
                all_Css[a[0]] = all_Css[a[0]] + [cur_sub_track[:,:nb_dims]]
                all_Bss[a[0]] = all_Bss[a[0]] + [cur_sub_state]
                all_sigs[a[0]] = all_sigs[a[0]] + [cur_sub_sigmas[:,:nb_dims]]

            positions = positions[np.argmin(inFOV):]
            state = state[np.argmin(inFOV):]
            inFOV = inFOV[np.argmin(inFOV):]
    
    print('number of tracks:')
    for k in range(len(all_Css)):
        all_Css[k] = np.array(all_Css[k])
        all_Bss[k] = np.array(all_Bss[k])
        all_sigs[k] = np.array(all_sigs[k])
        if len(all_Css[k])>0:
            print(str(all_Css[k].shape[1])+' pos :',str(len(all_Css[k]))+', ', end = '')
    print('')
    
    all_Css_dict = {}
    all_Bss_dict = {}
    all_sigs_dict = {}
    for Cs, Bs, sigs in zip(all_Css, all_Bss, all_sigs):
        if Cs.shape[0] > 0:
            l = str(Cs.shape[1])
            all_Css_dict[l] = Cs
            all_Bss_dict[l] = Bs    
            all_sigs_dict[l] = sigs

    return all_Css_dict, all_Bss_dict, all_sigs_dict
