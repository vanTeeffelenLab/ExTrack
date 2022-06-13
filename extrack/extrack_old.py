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
from lmfit import minimize, Parameters

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

def P_Cs_inter_bound_stats(Cs, LocErr, ds, Fs, TR_params, nb_substeps=1, do_frame = 1, frame_len = 6, do_preds = 0) :
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
    nb_Tracks = len(Cs)
    nb_locs = len(Cs[0]) # number of localization per track
    nb_dims = len(Cs[0,0]) # number of spatial dimentions (x, y) or (x, y, z)
    Cs = Cs.reshape((nb_Tracks,1,nb_locs, nb_dims))
    Cs = cp.array(Cs)
    
    nb_states = TR_params[0]
    
    if do_preds:
        preds = np.zeros((nb_Tracks, nb_locs, nb_states))-1
    else :
        preds = []
    
    if do_frame == 0:
        frame_len = 100
    
    all_Bs = get_all_Bs(np.min([(nb_locs-1)*nb_substeps+1, frame_len + nb_substeps - 1]), nb_states)[None]
    
    TrMat = cp.array(TR_params[1]) # transition matrix of the markovian process
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

    LP = LT # current log proba of seeing the track
    LP = cp.repeat(LP, nb_Tracks, axis = 0)
    cur_ds = ds[cur_states]
    cur_ds = (cur_ds[:,:,1:]**2 + cur_ds[:,:,:-1]**2)**0.5 / 2**0.5 # assuming a transition at the middle of the substeps
    # we can average the variances of displacements per step to get the actual std of displacements per step
    cur_ds = cp.mean(cur_ds**2, axis = 2)**0.5
    cur_ds = cur_ds[:,:,None]
    cur_ds = cp.array(cur_ds)
    # inject the first position to get the associated Km and Ks :
    Km, Ks = first_log_integrale_dif(Cs[:,:, nb_locs-current_step], LocErr, cur_ds)
    current_step += 1
    Km = cp.repeat(Km, cur_nb_Bs, axis = 1)
    removed_steps = 0
    
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

        # repeat the previous matrix to account for the states variations due to the new position
        Km = cp.repeat(Km, nb_states**nb_substeps , axis = 1)
        Ks = cp.repeat(Ks, nb_states**nb_substeps, axis = 1)
        LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)
        # inject the next position to get the associated Km, Ks and Constant describing the integral of 3 normal laws :
        Km, Ks, LC = log_integrale_dif(Cs[:,:,nb_locs-current_step], LocErr, cur_ds, Km, Ks)
        #print('integral',time.time() - t0); t0 = time.time()
        LP += LT + LC # current (log) constants associated with each track and sequences of states
        del LT, LC
        cur_nb_Bs = len(cur_Bs[0]) # current number of sequences of states

        ''''idea : the position and the state 6 steps ago should not impact too much the 
        probability of the next position so the Km and Ks of tracks with the same 6 last 
        states must be very similar, we can then fuse the parameters of the pairs of Bs
        which vary only for the last step (7) and sum their probas'''
        if do_frame and current_step < nb_locs-1:
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
                cur_Bs.shape
                cur_Bs = cur_Bs[:,:cur_nb_Bs//nb_states, :-1]
                Km, Ks, LP = fuse_tracks(Km, Ks, LP, cur_nb_Bs, nb_states)
                cur_nb_Bs = len(cur_Bs[0])
                removed_steps += 1
        
        #print('frame',time.time() - t0)
        #print(current_step,time.time() - t0)
        current_step += 1

    newKs = cp.array((Ks**2 + LocErr**2)**0.5)[:,:,0]
    log_integrated_term = -cp.log(2*np.pi*newKs**2) - cp.sum((Cs[:,:,0] - Km)**2,axis=2)/(2*newKs**2)
    LF = cp.log(Fs[cur_Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
    #LF = cp.log(0.5)
    # cp.mean(cp.log(Fs[cur_Bs[:,:,:].astype(int)]), 2) # Log proba of starting in a given state (fractions)
    LP += log_integrated_term + LF
    
    pred_LP = LP
    if np.max(LP)>600: # avoid overflow of exponentials, mechanically also reduces the weight of longest tracks
        pred_LP = LP - (np.max(LP)-600)
    
    P = np.exp(pred_LP)
    if do_preds :
        for state in range(nb_states):
            B_is_state = cur_Bs[:,:] == state
            preds[:,0:frame_len, state] = asnumpy(np.sum(B_is_state*P[:,:,None],axis = 1)/np.sum(P[:,:,None],axis = 1))
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

def Proba_Cs(Cs, LocErr, ds, Fs, TR_params, nb_substeps, do_frame, frame_len):
    '''
    inputs the observed localizations and determine the probability of 
    observing these data knowing the localization error, D the diffusion coef,
    pu the proba of unbinding per step and pb the proba of binding per step
    sum the proba of Cs inter Bs (calculated with P_Cs_inter_bound_stats)
    over all Bs to get the proba of Cs (knowing the initial position c0)
    '''
    LP_CB, _, _  = P_Cs_inter_bound_stats(Cs, LocErr, ds, Fs, TR_params, nb_substeps, do_frame, frame_len)
    # calculates P(C) the sum of P(C inter B) for each track
    max_LP = np.max(LP_CB, axis = 1, keepdims = True)
    LP_CB = LP_CB - max_LP
    max_LP = max_LP[:,0]
    P_CB = np.exp(LP_CB)
    P_C = cp.sum(P_CB, axis = 1) # sum over B
    LP_C = np.log(P_C) + max_LP # back to log proba of C without overflow due to exponential
    return LP_C

def predict_Bs(all_Cs, dt, params, states_nb, frame_len):
    '''
    inputs the observed localizations and parameters and determines the proba
    of each localization to be in a given state.
    '''
    if type(all_Cs) == type({}):
        all_Cs_list = []
        for l in all_Cs:
            all_Cs_list.append(all_Cs[l])
        all_Cs = all_Cs_list
    
    nb_substeps=1 # substeps should not impact the step labelling
    LocErr, ds, Fs, TR_params = extract_params(params, dt, states_nb, nb_substeps)
    all_pred_Bs = []
    for k in range(len(all_Cs)):
        LP_Cs, trunkated_Bs, pred_Bs = P_Cs_inter_bound_stats(all_Cs[k], LocErr, ds, Fs, TR_params, nb_substeps = 1, do_frame = 1,frame_len = frame_len, do_preds = 1)
        all_pred_Bs.append(pred_Bs)
        
    all_pred_Bs_dict = {}
    for pred_Bs in all_pred_Bs:
        all_pred_Bs_dict[str(pred_Bs.shape[1])] = pred_Bs

    return all_pred_Bs_dict

'''
def acc_Bs(all_Cs, dt, params, True_Bs, states_nb, frame_len):
    """
    inputs the observed localizations, parameters and true palbels to determines
    accuracy of the method to label the states for each localization
    """
    nb_substeps=1 # substeps should not impact the step labelling
    LocErr, ds, Fs, TR_params = extract_params(params, dt, states_nb, nb_substeps)
    P_Cs, trunkated_Bs, pred_Bs = P_Cs_inter_bound_stats(all_Cs, LocErr, ds, Fs, TR_params, nb_substeps = 1, do_frame = 1,frame_len = frame_len, do_preds = 1)
    True_Bs = cp.array(True_Bs)
    pred_Bs = np.argmax(pred_Bs, axis = 2)
    accuracy = cp.mean(pred_Bs==True_Bs, axis=0)
    accuracy = asnumpy(accuracy)
    print('accuracy :', accuracy)
    return accuracy
'''

def extract_params(params, dt, states_nb, nb_substeps):
    '''
    turn the parameters which differ deppending on the number of states into lists
    ds (diffusion lengths), Fs (fractions), TrMat (substep transiton matrix)
    '''
    TR_params = []
    TR_params.append(states_nb)
    LocErr = params['LocErr'].value
    
    if states_nb == 2:
        D0 = params['D0'].value
        D1 = params['D1'].value
        Ds = np.array([D0,D1])
        F0 = params['F0'].value
        Fs = np.array([F0, 1-F0])
        p01 = params['p01']
        p10 =  params['p10']
        # correct p10 and p01 which actually correspond to r*dt to get the 
        # corresponding discrete probabilities of at least 1 transition during 
        # a substep :
        p01 = 1 - np.exp(-p01/nb_substeps)
        p10 = 1 - np.exp(-p10/nb_substeps)
        
        TrMat = np.array([[1-p01, p01],[p10, 1- p10]])
        TR_params.append(TrMat)
    
    elif states_nb == 3:
        D0 = params['D0']
        D1 = params['D1']
        D2 = params['D2']
        Ds = np.array([D0,D1,D2])
        p01 = params['p01']
        p02 = params['p02']
        p10 = params['p10']
        p12 = params['p12']
        p20 = params['p20']
        p21 = params['p21']
        
        p01 = np.max([1e-10,p01])
        p02 = np.max([1e-10,p02])
        p10 = np.max([1e-10,p10])
        p12 = np.max([1e-10,p12])
        p20 = np.max([1e-10,p20])
        p21 = np.max([1e-10,p21])
        
        F0 = params['F0']
        F1 = params['F1']
        F2 = params['F2']
        
        Fs = np.array([F0, F1, F2])
        p01 = 1 - np.exp(-p01/nb_substeps)
        p02 = 1 - np.exp(-p02/nb_substeps)
        p10 = 1 - np.exp(-p10/nb_substeps)
        p12 = 1 - np.exp(-p12/nb_substeps)
        p20 = 1 - np.exp(-p20/nb_substeps)
        p21 = 1 - np.exp(-p21/nb_substeps)
        
        TrMat = np.array([[1-p01-p02, p01, p02],
                          [p10, 1-p10-p12, p12],
                          [p20, p21, 1-p20-p21]])
        
        TR_params.append(TrMat)
    ds = np.sqrt(2*Ds*dt)
    return LocErr, ds, Fs, TR_params

def cum_Proba_Cs(params, all_Cs, dt, states_nb, nb_substeps, do_frame, frame_len, verbose = 1):
    '''
    each probability can be multiplied to get a likelihood of the model knowing
    the parameters LocErr, D0 the diff coefficient of state 0 and F0 fraction of
    state 0, D1 the D coef at state 1, p01 the probability of transition from
    state 0 to 1 and p10 the proba of transition from state 1 to 0.
    here sum the logs(likelihood) to avoid too big numbers
    '''
    LocErr, ds, Fs, TR_params = extract_params(params, dt, states_nb, nb_substeps)
    
    Cum_P = 0
    for k in range(len(all_Cs)):
        Css = all_Cs[k]
        nb_max = 700
        for n in range(int(np.ceil(len(Css)/nb_max))):
            Csss = Css[n*nb_max:(n+1)*nb_max]
            LP = Proba_Cs(Csss, LocErr, ds, Fs, TR_params, nb_substeps, do_frame, frame_len)
            #plt.clf()
            #plt.hist(np.log(Ps), np.arange(-5,50, 0.5), density = True)
            #plt.hist(Ps, np.arange(0,np.exp(18), np.exp(18)/200), density = True)
            #print(len(Css[0]), cp.sum(LP)/(len(Css) * (len(Css[0])-1)))
            Cum_P += cp.sum(LP)
    Cum_P = asnumpy(Cum_P)
    
    if verbose == 1:
        q = [param + ' = ' + str(np.round(params[param].value, 4)) for param in params]
        print(Cum_P, q)
    #nb_locs = [Cs.shape[0]*Cs.shape[1] for Cs in all_Cs]
    #nb_locs = np.sum(nb_locs)
    #out = - np.exp(Cum_P/len(all_Cs)/(len(all_Cs[0])-1)) # normalize by the number of tracks and number of displacements
    out = - Cum_P # normalize by the number of tracks and number of displacements
    if out == np.nan:
        out = 1E100
    return out

def get_2DSPT_params(all_Cs,
                     dt,
                     nb_substeps = 1,
                     states_nb = 2,
                     do_frame = 1,
                     frame_len = 10,
                     verbose = 1,
                     method = 'powell',
                     steady_state = True,
                     vary_params = {'LocErr' : True, 'D0' : False, 'D1' : True, 'F0' : True, 'p01' : True, 'p10' : True},
                     estimated_vals =  {'LocErr' : 0.025, 'D0' : 1e-20, 'D1' : 0.05, 'F0' : 0.45, 'p01' : 0.05, 'p10' : 0.05},
                     min_values = {'LocErr' : 0.007, 'D0' : 1e-12, 'D1' : 0.00001, 'F0' : 0.001, 'p01' : 0.01, 'p10' : 0.01},
                     max_values = {'LocErr' : 0.6, 'D0' : 1, 'D1' : 10, 'F0' : 0.999, 'p01' : 1., 'p10' : 1.}):

    '''
    all_Cs : list of 3D arrays of tracks, dim 0 = track ID, dim 1 = sequence of positions, dim 2 = x, y, (z) axes
    estimated_vals : list of parameters [LocError, D0, D1, F0, p01, p10] if 2 states,
    [LocError, D0, D1, F0, F1, p01, p02, p10, p12, p20, p21] if 3 states.

    dt : time in between each frame
    verbose : if 1 returns the parameter values and log likelihood for each step of the fit, if 0 return nothing 
    nb_substeps : nb of substeps per step (1 = no substeps)
    states_nb : number of states in the model
    method : lmfit optimization method
    vary_params = list of bool stating if the method varies each parameters in the same order than in estimated_vals
    steady_state : bool stating if assuming stady state or not (constrains rates and Fractions to 2 free params for a 2 states model or 6 for a 3 states model)
    min_values, max_values : minimum values and maximum values of each parameters in the order of estimated_vals
    
    in case of 3 states models vary_params, estimated_vals, min_values and max_values can be replaced :
    vary_params = {'LocErr' : True, 'D0' : False, 'D1' :  True, 'D2' : True, 'F0' : True, 'F1' : True, 'p01' : True, 'p02' : True, 'p10' : True,'p12' :  True,'p20' :  True, 'p21' : True},
    estimated_vals = {'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'F0' : 0.33,  'F1' : 0.33, 'p01' : 0.1, 'p02' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p20' :0.1, 'p21' :0.1},
    min_values = {'LocErr' : 0.007, 'D0' : 1e-20, 'D1' : 0.0000001, 'D2' :  0.000001, 'F0' : 0.001,  'F1' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p20' :0.001, 'p21' :0.001},
    max_values = {'LocErr' : 0.6, 'D0' : 1e-20, 'D1' : 1, 'D2' :  10, 'F0' : 0.999,  'F1' : 0.999, 'p01' : 1, 'p02' : 1, 'p10' : 1, 'p12' : 1, 'p20' : 1, 'p21' : 1}
    '''
    if str(all_Cs.__class__) == "<class 'dict'>":
        all_Cs_list = []
        for l in all_Cs:
            all_Cs_list.append(all_Cs[l])
        all_Cs = all_Cs_list

    if  states_nb == 2:
        if not (len(min_values) == 6 and len(max_values) == 6 and len(estimated_vals) == 6 and len(vary_params) == 6):
            raise ValueError('estimated_vals, min_values, max_values and vary_params should all containing 6 parameters')
        if steady_state:
                print(estimated_vals)
                param_kwargs = [{'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : max_values['D0'], 'vary' : vary_params['D0']},
                                {'name' : 'D1_minus_D0', 'value' : estimated_vals['D1'] - estimated_vals['D0'], 'min' : min_values['D1']-min_values['D0'], 'max' : max_values['D1'], 'vary' : vary_params['D1']},
                                {'name' : 'D1', 'expr' : 'D0 + D1_minus_D0'},
                                {'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' :  min_values['LocErr'],'max' :  max_values['LocErr'], 'vary' : vary_params['LocErr']},
                                {'name' : 'F0', 'value' : estimated_vals['F0'], 'min' :  min_values['F0'], 'max' :  max_values['F0'], 'vary' :  vary_params['F0']},
                                {'name' : 'F1', 'expr' : '1 - F0'},
                                {'name' : 'p01', 'value' : estimated_vals['p01'], 'min' :  min_values['p01'], 'max' :  max_values['p01'], 'vary' :  vary_params['p01']},
                                {'name' : 'p10', 'expr' : 'p01/(1/F0-1)'}]
        else :
                param_kwargs = [{'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : max_values['D0'], 'vary' : vary_params['D0']},
                                {'name' : 'D1_minus_D0', 'value' : estimated_vals['D1'] - estimated_vals['D0'], 'min' : min_values['D1']-min_values['D0'], 'max' : max_values['D1'], 'vary' : vary_params['D1']},
                                {'name' : 'D1', 'expr' : 'D0 + D1_minus_D0' },
                                {'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' :  min_values['LocErr'],'max' :  max_values['LocErr'], 'vary' : vary_params['LocErr']},
                                {'name' : 'F0', 'value' : estimated_vals['F0'], 'min' :  min_values['F0'], 'max' :  max_values['F0'], 'vary' :  vary_params['F0']},
                                {'name' : 'F1', 'expr' : '1 - F0'},
                                {'name' : 'p01', 'value' : estimated_vals['p01'], 'min' :  min_values['p01'], 'max' :  max_values['p01'], 'vary' :  vary_params['p01']},
                                {'name' : 'p10', 'value' : estimated_vals['p10'], 'min' :  min_values['p10'], 'max' :  max_values['p10'], 'vary' : vary_params['p10']}]

    elif states_nb == 3:
        '''
        e.g. :
        vary_params = { 'LocErr' : True, 'D0' : False, 'D1' :  True, 'D2' : True, 'F0' : True, 'F1' : True, 'p01' : True, 'p02' : True, 'p10' : True,'p12' :  True,'p20' :  True, 'p21' : True}
        estimated_vals = { 'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'F0' : 0.33,  'F1' : 0.33, 'p01' : 0.1, 'p02' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p20' :0.1, 'p21' :0.1}
        min_values = { 'LocErr' : 0.007, 'D0' : 1e-20, 'D1' : 0.0000001, 'D2' :  0.000001, 'F0' : 0.001,  'F1' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p20' :0.001, 'p21' :0.001}
        max_values = { 'LocErr' : 0.6, 'D0' : 1e-20, 'D1' : 1, 'D2' :  10, 'F0' : 0.999,  'F1' : 0.999, 'p01' : 1, 'p02' : 1, 'p10' : 1, 'p12' : 1, 'p20' : 1, 'p21' : 1})
        '''
        if not (len(min_values) == 12 and len(max_values) == 12 and len(estimated_vals) == 12 and len(vary_params) == 12):
            raise ValueError('estimated_vals, min_values, max_values and vary_params should all containing 12 parameters')

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
                            {'name' : 'F2', 'expr' : '1-F0-F1'}]
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
                            {'name' : 'F2', 'expr' : '1-F0-F1'}]

    else :
        raise ValueError("wrong number of states, must be either 2 or 3")
    
    params = Parameters()
    [params.add(**param_kwargs[k]) for k in range(len(param_kwargs))]
    #print(params)
    
    fit = minimize(cum_Proba_Cs, params, args=(all_Cs, dt, states_nb, nb_substeps, do_frame, frame_len, verbose), method = method, nan_policy = 'propagate')
    return fit

