# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:23:41 2024

@author: franc
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN
from sklearn.mixture import GaussianMixture
import pandas as pd

dtype = 'float64'
clipping_value = np.log(1e-40)

def read_table(paths, # path of the file to read or list of paths to read multiple files.
               lengths = np.arange(5,40), # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions 
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
               colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = True):
    
    if type(paths) == str or type(paths) == np.str_:
        paths = [paths]

    tracks = {}
    frames = {}
    opt_metrics = {}
    for m in opt_colnames:
        opt_metrics[m] = {}
    nb_peaks = 0
    for l in lengths:
        tracks[str(l)] = []
        frames[str(l)] = []
        for m in opt_colnames:
            opt_metrics[m][str(l)] = []
    
    for path in paths:
        
        if fmt == 'csv':
            data = pd.read_csv(path, sep=',')
        elif fmt == 'pkl':
            data = pd.read_pickle(path)
        else:
            data = pd.read_csv(path, sep = fmt)
        
        if not (type(colnames[3]) == str or type(colnames[3]) == np.str_):
            # in this case we remove the NA values for simplicity
            None_ID = (data[colnames[3]] == 'None') + pd.isna(data[colnames[3]])
            data = data.drop(data[np.any(None_ID,1)].index)
                
            new_ID = data[colnames[3][0]].astype(str)
            
            for k in range(1,len(colnames[3])):
                new_ID = new_ID + '_' + data[colnames[3][k]].astype(str)
            data['unique_ID'] = new_ID
            colnames[3] = 'unique_ID'        
        try:
            # in this case, peaks without an ID are assumed alone and are added a unique ID, only works if ID are integers
            None_ID = (data[colnames[3]] == 'None' ) + pd.isna(data[colnames[3]])
            max_ID = np.max(data[colnames[3]][(data[colnames[3]] != 'None' ) * (pd.isna(data[colnames[3]]) == False)].astype(int))
            data.loc[None_ID, colnames[3]] = np.arange(max_ID+1, max_ID+1 + np.sum(None_ID))
        except:
            None_ID = (data[colnames[3]] == 'None' ) + pd.isna(data[colnames[3]])
            data = data.drop(data[None_ID].index)
        
        data = data[colnames + opt_colnames]
        
        zero_disp_tracks = 0
            
        try:
            for ID, track in data.groupby(colnames[3]):
                
                track = track.sort_values(colnames[2], axis = 0)
                track_mat = track.values[:,:3].astype('float64')
                dists = np.sum((track_mat[1:, :2] - track_mat[:-1, :2])**2, axis = 1)**0.5
                if track_mat[0, 2] >= frames_boundaries[0] and track_mat[0, 2] <= frames_boundaries[1] : #and np.all(dists<dist_th):
                    if not np.any(dists>dist_th):
                        
                        if np.any([len(track_mat)]*len(lengths) == np.array(lengths)):
                            l = len(track)
                            tracks[str(l)].append(track_mat[:, 0:2])
                            frames[str(l)].append(track_mat[:, 2])
                            for m in opt_colnames:
                                opt_metrics[m][str(l)].append(track[m].values)
        
                        elif len(track_mat) > np.max(lengths):
                            l = np.max(lengths)
                            tracks[str(l)].append(track_mat[:l, 0:2])
                            frames[str(l)].append(track_mat[:l, 2])
                            for m in opt_colnames:
                                opt_metrics[m][str(l)].append(track[m].values[:l]) 
                        
                        elif len(track_mat) < np.max(lengths) and len(track_mat) > np.min(lengths) : # in case where lengths between min(lengths) and max(lentghs) are not all present:
                            l_idx =   np.argmin(np.floor(len(track_mat) / lengths))-1
                            l = lengths[l_idx]
                            tracks[str(l)].append(track_mat[:l, 0:2])
                            frames[str(l)].append(track_mat[:l, 2])
        except :
            print('problem with file :', path)
        
    for l in list(tracks.keys()):
        if len(tracks[str(l)])>0:
            print(l)
            tracks[str(l)] = np.array(tracks[str(l)])
            frames[str(l)] = np.array(frames[str(l)])
            for m in opt_colnames:
                opt_metrics[m][str(l)] = np.array(opt_metrics[m][str(l)])
        else:
            del tracks[str(l)], frames[str(l)]
            for k, m in enumerate(opt_colnames):
                del opt_metrics[m][str(l)]        
                    
    if zero_disp_tracks and not remove_no_disp:
        print('Warning: some tracks show no displacements. To be checked if normal or not. These tracks can be removed with remove_no_disp = True')
    return tracks, frames, opt_metrics

class Brownian_Initial_layer(tf.keras.layers.Layer):
    def __init__(
        self,
        Fixed_LocErr,
        Initial_params = {'LocErr': 0.02, 'd': 0.1}, 
        **kwargs,
    ):
        self.Fixed_LocErr = Fixed_LocErr
        self.Initial_params = Initial_params
        super().__init__(**kwargs)
       
    def build(self, input_shape):
        print(input_shape)
        self.Logd2 = tf.Variable(np.full((input_shape[0], 1), 2*np.log(self.Initial_params['d'])), dtype = dtype, name = 'Logd2', constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        self.LogLocErr2 = tf.Variable(np.full((input_shape[0], 1), 2*np.log(self.Initial_params['LocErr'])), dtype = dtype, name = 'LogLocErr2', trainable = not self.Fixed_LocErr, constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        self.built = True
   
    def call(self, inputs):
        init_mu = inputs[:,0] # initial most likely real postion
        init_s2 = tf.math.exp(self.LogLocErr2) + tf.math.exp(self.Logd2) # initial variance
        print(inputs)
        init_LP = tf.zeros_like(inputs[:,:1,0])
        initial_state = [init_mu, init_s2, init_LP]
        return inputs, initial_state

    def get_parameters(self):
        return np.exp(self.LogLocErr2)**0.5, np.exp(self.Logd2)**0.5

class Brownian_RNNCell(tf.keras.layers.Layer):
   
    def __init__(self, state_size, parent, **kwargs):
        self.state_size = state_size # [nb_dims, 1,1]
        self.parent = parent
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
   
    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True

    def call(self, inputs, states): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        prev_output = states
        output = self.Brownian_function(inputs, prev_output)
        return output[2], output
        #return [output[2], tf.math.exp(0.5*self.parent.weights[0]), tf.math.exp(0.5*self.parent.weights[1])], output # the first output is the output of the layer, the second is the hidden states
       
    def Brownian_function(self, input_i, prev_output):
        d2 = tf.math.exp(self.parent.weights[0])
        LocErr2 = tf.math.exp(self.parent.weights[1])
        mu, s2, LP = prev_output
        variance = LocErr2 + s2
        top = input_i - mu
        
        new_LP = LP + tf.math.reduce_sum(self.log_gaussian(top, variance), axis = -1, keepdims = True)
        new_mu = (mu * LocErr2 + input_i * s2)/variance
        new_s2 = (d2*s2 + s2*LocErr2 + LocErr2*d2)/variance
        output = [new_mu, new_s2, new_LP]
        return output
       
    def log_gaussian(self, top, variance):
        return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)

def Brownian_fit(tracks, nb_dims, verbose = 0, Fixed_LocErr = True, Initial_params = {'LocErr': 0.02, 'd': 0.1}, nb_epochs = 400):
    '''
    Fit single tracks to a model with diffusion (+ localizatione error). If memory issues occur, split your data 
    set into multiple arrays and perform a fitting on each array separately.
    
    Parameters
    ----------
    tracks : numpy array
        3-dimension array of tracks of same length. dims: track ID, time point, space coordinates (x,y,..)
    verbose : int
        tensorflow model fit verbose. The default is 0.
    Fixed_LocErr : bool, optional
        Fix the localization error to its initial value if True. The default is True.
    nb_epochs : int
        Number of epochs for the model fitting. The default is 400.
    Initial_params : dict,
        Dictionary of the initial parameters. The values for each key must be a list of floats of length `nb_states`.
        The default is {'LocErr': 0.02, 'd': 0.1}.
        The parameters represent the localization error, the diffusion length per step, the change per step of the x, y speed of the
        directed motion and the standard deviation of the initial speed of the particle. 
    
    Returns
    -------
    pd_params: Log likelihood and model parameters in pandas format
        est_LocErrs: 
            Estiamted localization errors for each state 
        est_ds:
            Estimated d, the diffusion length per step of the particle, for each state (the diffusion coefficient D = d**2/(2dt)
        LP: Log probability of each track according to the model.
    '''
    
    nb_tracks = len(tracks)
    input_size = nb_dims
        
    inputs = tf.keras.Input(shape=(None, input_size), batch_size = nb_tracks, dtype = dtype)
    layer1 = Brownian_Initial_layer(Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_params, dtype = dtype)
    tensor1, initial_state = layer1(inputs)
    cell = Brownian_RNNCell([input_size, 1, 1], layer1, dtype = dtype) # n
    RNN_layer = tf.keras.layers.RNN(cell, dtype = dtype)
    outputs = RNN_layer(tensor1[:,1:], initial_state = initial_state)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Diffusion_model")
    if verbose > 0:
        model.summary()
        
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        #print(y_pred)
        return - tf.math.reduce_sum(y_pred, axis=-1) # sum over the spatial dimensions axis
    
    # model.compile(loss=MLE_loss, optimizer='adam')    
    adam = tf.keras.optimizers.Adam(learning_rate=0.9, beta_1=0.1, beta_2=0.1) # we use a first set of parameters with hight learning_rate and low beta values to accelerate initial learning
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tracks[:nb_tracks,:,:input_size], tracks[:nb_tracks,:,:input_size], epochs = 20, batch_size = nb_tracks, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tracks[:nb_tracks,:,:input_size], tracks[:nb_tracks,:,:input_size], epochs = nb_epochs, batch_size = nb_tracks, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    
    LP = model.predict_on_batch(tf.constant(tracks[:nb_tracks,:,:input_size], dtype = dtype))
    est_LocErrs, est_ds = model.layers[1].get_parameters()
    
    pd_params = pd.DataFrame(np.concatenate((LP, est_LocErrs, est_ds), axis = 1), columns = ['Log_likelihood', 'LocErr', 'd'])
        
    return pd_params#est_LocErrs, est_ds, est_qs, est_ls, LP

class Directed_Initial_layer(tf.keras.layers.Layer):
    def __init__(
        self,
        Fixed_LocErr,
        Initial_params = {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01},
        **kwargs,
    ):
        self.Fixed_LocErr = Fixed_LocErr
        self.Initial_params = Initial_params
        super().__init__(**kwargs)
       
    def build(self, input_shape):
        def inverse_sigmoid(x):
            return tf.math.log(x/(1-x))
       
        self.Log_d2 = tf.Variable(np.full((input_shape[0], 1), 2*np.log(self.Initial_params['d'])), dtype = dtype, name = 'Logd2', constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        self.Log_q2 = tf.Variable(np.full((input_shape[0], 1), 2*np.log(self.Initial_params['q'])), dtype = dtype, name = 'Log_q2', constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        self.Log_l2 = tf.Variable(np.full((input_shape[0], 1), 2*np.log(self.Initial_params['l'])), dtype = dtype, name = 'Log_l2', constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        self.Log_LocErr2 = tf.Variable(np.full((input_shape[0], 1), 2*np.log(self.Initial_params['LocErr'])), dtype = dtype, name = 'LogLocErr2', trainable = not self.Fixed_LocErr, constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        self.built = True
    
    def call(self, inputs, nb_dims):
       
        LocErr2 =  tf.math.exp(self.Log_LocErr2)
        d2 = tf.math.exp(self.Log_d2)
        q2 = tf.math.exp(self.Log_q2)
        l2 = tf.math.exp(self.Log_l2)
        #print(LocErr2**0.5, d2**0.5, q2**0.5, l2**0.5)
       
        LP = tf.zeros_like(inputs[:,:1,0], dtype = dtype)
        #step 0:
        s2i = LocErr2 + d2
        x2i = l2 + q2
        g2i = (q2 *  s2i + s2i * l2 + l2 * q2) / x2i
        LocErr2_g2i = LocErr2 + g2i
        alphai = l2 / x2i
        gammai = inputs[:, 0]
        h2i = l2
        #step 1:
        ki = (inputs[:, 1] - gammai) / alphai
        
        h2i = (g2i + LocErr2) / alphai**2
        LP += - nb_dims * tf.math.log(alphai)
        #fs1
        ai = LocErr2 * alphai / LocErr2_g2i + 1
        a2i = ai**2
        bi = (g2i * inputs[:, 1] + LocErr2 * gammai) / LocErr2_g2i
        s2i = (LocErr2 * g2i + g2i * d2 + d2 * LocErr2) / (LocErr2_g2i * a2i)
        #ft1
        betai = x2i / (x2i + q2)
        t2i = betai * q2
        #chi1
        x2i = x2i + q2
        #G1
        h2i_t2i = h2i + t2i
        alphai = ai * betai * h2i / h2i_t2i
        gammai = bi + ai * t2i * ki / h2i_t2i
        g2i = a2i * (t2i *  s2i + s2i * h2i + h2i * t2i) / h2i_t2i
        #Q1
        kim1 = ki / betai
        q2i = (h2i + t2i) / betai**2
        LP += - nb_dims * tf.math.log(betai)
        
        initial_state = [LP, alphai, gammai, g2i, q2i, kim1, x2i, ki]
        
        return inputs, initial_state

    def get_parameters(self):
        return np.exp(self.Log_LocErr2)**0.5, np.exp(self.Log_d2)**0.5, np.exp(self.Log_q2)**0.5, 2**0.5*np.exp(self.Log_l2)**0.5 

class Directed_RNNCell(tf.keras.layers.Layer):
   
    def __init__(self, state_size, parent, nb_dims, **kwargs):
        self.state_size = state_size # [nb_dims, 1,1]
        self.parent = parent
        self.nb_dims = nb_dims
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
   
    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True

    def call(self, inputs, states): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        prev_output = states
        output = self.directed_motion_function(inputs, prev_output, self.nb_dims)
        return [output[0], output[7]], output # the first oupput is the output of the layer, the second is the hidden states
   
    def directed_motion_function(self, input_i, prev_output, nb_dims):
           
        def log_gaussian(top, variance):
            return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)
                
        d2 = tf.math.exp(self.parent.weights[0])
        q2 = tf.math.exp(self.parent.weights[1])
        #l2 = tf.math.exp(self.parent.weights[2])
        LocErr2 = tf.math.exp(self.parent.weights[3])
        
        [LP, alphai, gammai, g2i, q2i, kim1, x2i, ki] = prev_output
        ki = (input_i - gammai) / alphai
        h2i = (g2i + LocErr2) / alphai**2
        LP += - nb_dims * tf.math.log(alphai)
        #fsi
        ai = LocErr2 * alphai /(LocErr2 + g2i) + 1
        a2i = ai**2
        bi = (g2i * input_i + LocErr2 * gammai) / (LocErr2 + g2i)
        s2i = (LocErr2 * g2i + g2i * d2 + d2 * LocErr2) / ((LocErr2 + g2i) * a2i)
        # int of Q_{i-1}(w_i - k_{i-1}/beta_{i-1}) chi_{i-1}(w_i) K_i(w_i - k_i) fsi(r_{i+1} - (a_i * w_i + b_i)) fdz(w_{i+1} - w_i)    we need to reduce the 5 terms to 3: fusion of Q and K to L and K' and chi_{i-1} and fdz to chi_i and fti
        # fusion of Qi-1 and Ki (Ki * Qi-1 -> Li * Ki)
        h2i_q2i = h2i + q2i
        LP += tf.math.reduce_sum(log_gaussian(ki - kim1, h2i_q2i), axis = -1, keepdims=True)
        ki = (h2i * kim1 + q2i * ki) / h2i_q2i
        #LP += - tf.math.abs(tf.math.reduce_sum(ki**2, axis = -1, keepdims=True)**0.5 - l2**0.5)/speed_regularization_factor # additional term that reglates the difference between the linear motion speed parameter and the observed speed. This tackles the issue of having a model with uncorrelated x and y displacements.
        h2i = h2i * q2i / h2i_q2i
        #fti
        x2i_q2 = x2i + q2
        betai = x2i / x2i_q2
        t2i = betai * q2
        #chii
        x2i = x2i_q2
        #Gi
        h2i_t2i = h2i + t2i
        alphai = ai * betai * h2i / h2i_t2i
        gammai =  bi + ai * t2i * ki / h2i_t2i
        g2i = a2i * (t2i * s2i + s2i * h2i + h2i * t2i) / h2i_t2i
        #LocErr2_g2i = LocErr2 + g2i
        #Qi
        kim1 = ki / betai
        q2i = h2i_t2i / betai**2
        LP += - nb_dims * tf.math.log(betai)
        
        output = [LP, alphai, gammai, g2i, q2i, kim1, x2i, ki]
        #print(prev_output)
       
        return output # the first oupput is the output of the layer, the second is the hidden states

    def log_gaussian(self, top, variance):
        return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)

class Directed_Final_layer(tf.keras.layers.Layer):

    def __init__(self, parent, nb_dims, **kwargs):
        self.parent = parent
        self.nb_dims = nb_dims
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
   
    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True
   
    def call(self, inputs, states): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        prev_output = states
        output = self.directed_motion_function(inputs, prev_output, self.nb_dims)
        return output # the first oupput is the output of the layer, the second is the hidden states
       
    def directed_motion_function(self, inputs, prev_outputs, nb_dims):
       
        [LP, alphai, gammai, g2i, q2i, kim1, x2i, ki] = prev_outputs

        d2 = tf.math.exp(self.parent.weights[0])
        LocErr2 = tf.math.exp(self.parent.weights[3])

        #int over r0:
        #step n-1: (nothing changes for the dr int but the dw int does not have any fdz term)
        #i = nb_locs - 2
        #Ki
        ki = (inputs[:, -2] - gammai) / alphai
        LocErr2_g2i = LocErr2 + g2i
        h2i = LocErr2_g2i / alphai**2
        LP += - nb_dims * tf.math.log(alphai)
        #fsi
        ai = LocErr2 * alphai /LocErr2_g2i + 1
        a2i = ai**2
        bi = (g2i * inputs[:, -2] + LocErr2 * gammai) / LocErr2_g2i
        s2i = (LocErr2 * g2i + g2i * d2 + d2 * LocErr2) / (LocErr2_g2i * a2i)
        # fusion of Qi-1 and Ki (Ki * Qi-1 -> Li * Ki)
        h2i_q2i = h2i + q2i
        LP += tf.math.reduce_sum(self.log_gaussian(ki - kim1, h2i_q2i), axis = -1, keepdims=True)
        ki = (h2i * kim1 + q2i * ki) / h2i_q2i
               
        h2i = h2i * q2i / h2i_q2i
        # int of chi_{i-1}(w_i) K_i(w_i - k_i) fsi(r_{i+1} - (a_i * w_i + b_i))
        #Qn-1
        h2i_x2i = h2i + x2i
        LP += tf.math.reduce_sum(self.log_gaussian(ki, h2i_x2i), axis = -1, keepdims=True)

        gammai = bi + ai * x2i * ki / h2i_x2i
        g2i = a2i * (x2i * s2i + s2i * h2i + h2i * x2i) / h2i_x2i
        LocErr2_g2i = LocErr2 + g2i
       
        #i = nb_locs - 1
        LP += tf.math.reduce_sum(self.log_gaussian(inputs[:, -1] - gammai, LocErr2_g2i), axis = -1, keepdims=True)
        
        return LP
    
    def log_gaussian(self, top, variance):
        return -0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)

#from keras.callbacks import LambdaCallback
def Directed_fit(tracks, nb_dims=2, verbose = 1, Fixed_LocErr = True, Initial_params = {'LocErr': 0.02, 'd': 0.01, 'q': 0.01, 'l': 0.01}, nb_epochs = 400):
    '''
    Fit single tracks to a model with diffusion plus directed motion. If memory issues occur, split your data 
    set into multiple arrays and perform a fitting on each array separately.
    
    Parameters
    ----------
    tracks : numpy array
        3-dimension array of tracks of same length. dims: track ID, time point, space coordinates (x,y,..)
    verbose : int
        tensorflow model fit verbose. The default is 0.
    Fixed_LocErr : bool, optional
        Fix the localization error to its initial value if True. The default is True.
    nb_epochs : int
        Number of epochs for the model fitting. The default is 300.
    Initial_params : dict,
        Dictionary of the initial parameters. The values for each key must be a list of floats of length `nb_states`.
        The default is {'LocErr': [0.02, 0.022, 0.022], 'd': [0.1, 0.12, 0.12], 'q': [0.01, 0.012, 0.012], 'l': [0.01, 0.02, 0.012]}.
        The parameters represent the localization error, the diffusion length per step, the change per step of the x, y speed of the
        directed motion and the standard deviation of the initial speed of the particle. 
    
    Returns
    -------
    pd_params: Log likelihood and model parameters in pandas format
        Log_likelihood: 
            log probability of each track according to the model.
        LocErr: 
            Estiamted localization errors for each state 
        d:
            Estimated d, the diffusion length per step of the particle, for each state (the diffusion coefficient D = d**2/(2dt)
        q: 
            Estimated diffusion lengths per step of the potential well for each state.
        l:
            Estiamted standard deviation of the initial speed of the particle.
        mean_speed:
            Predicted average speed of the particle along the whole track (as opposed to l which represents the speed at the first time point)
    '''
    
    nb_tracks = len(tracks)
    input_size = nb_dims
    track_len = tracks.shape[1]
    if track_len > 4:
        inputs = tf.keras.Input(shape=(None, input_size), batch_size = nb_tracks, dtype = dtype)
        layer1 = Directed_Initial_layer(Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_params, dtype = dtype)
        tensor1, initial_state = layer1(inputs, input_size)
        cell = Directed_RNNCell([1, 1, input_size, 1, 1, input_size, 1, input_size], layer1, input_size, dtype = dtype) # n
        RNN_layer = tf.keras.layers.RNN(cell, return_state = True, return_sequences=True, dtype = dtype)
        tensor2 = RNN_layer(tensor1[:,2:-2], initial_state = initial_state)
        kis = tensor2[1]
        prev_outputs = tensor2[2:]
        layer3 = Directed_Final_layer(layer1, input_size, dtype = dtype)
        #LP, estimated_ds, estimated_qs, estimated_ls, estimated_LocErrs = layer3(inputs[:,:], prev_outputs)
        LP = layer3(inputs[:,:], prev_outputs)
    elif track_len <= 4:
        inputs = tf.keras.Input(shape=(None, input_size), batch_size = nb_tracks, dtype = dtype)
        layer1 = Directed_Initial_layer(Fixed_LocErr = Fixed_LocErr, Initial_params = {'LocErr': 0.02, 'd': 0.01, 'q': 0.01, 'l': 0.1}, dtype = dtype)
        tensor1, initial_state = layer1(inputs, input_size)
        prev_outputs = initial_state
        layer3 = Directed_Final_layer(layer1, input_size, dtype = dtype)
        #LP, estimated_ds, estimated_qs, estimated_ls, estimated_LocErrs = layer3(inputs[:,:], prev_outputs)
        LP = layer3(inputs[:,:], prev_outputs)
        kis = LP
    
    model = tf.keras.Model(inputs=inputs, outputs=LP, name="Diffusion_model")
    if verbose > 0:
        model.summary()
    
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        return - tf.math.reduce_sum(y_pred, axis=-1)
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.9, beta_1=0.1, beta_2=0.1) # we use a first set of parameters with hight learning_rate and low beta values to accelerate initial learning
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tracks[:nb_tracks,:,:input_size], tracks[:nb_tracks,:,:input_size], epochs = 20, batch_size = nb_tracks, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tracks[:nb_tracks,:,:input_size], tracks[:nb_tracks,:,:input_size], epochs = nb_epochs, batch_size = nb_tracks, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

    LP = model.predict_on_batch(tf.constant(tracks[:nb_tracks,:,:input_size], dtype = dtype))
    est_LocErrs, est_ds, est_qs, est_ls = layer1.get_parameters()
    
    kis_model = tf.keras.Model(inputs=inputs, outputs=kis, name="Diffusion_model")
    pred_kis = kis_model.predict_on_batch(tf.constant(tracks[:nb_tracks,:,:input_size], dtype = dtype))
    mean_pred_kis = np.mean(np.sum(pred_kis**2, axis=2)**0.5, axis=1, keepdims = True)
    
    pd_params = pd.DataFrame(np.concatenate((LP, est_LocErrs, est_ds, est_qs, est_ls, mean_pred_kis), axis = 1), columns = ['Log_likelihood', 'LocErr', 'd', 'q', 'l', 'mean_speed'])
        
    return pd_params#est_LocErrs, est_ds, est_qs, est_ls, LP

'''
Confined diffusion

'''

class Confinement_Initial_layer(tf.keras.layers.Layer):
    def __init__(
        self,
        Fixed_LocErr,
        Initial_params = {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01},
        **kwargs,
    ):
        self.Fixed_LocErr = Fixed_LocErr
        self.Initial_params = Initial_params
        super().__init__(**kwargs)
       
    def build(self, input_shape):
       
        def inverse_sigmoid(x):
            return np.log(x/(1-x))
        
        self.Log_d2 = tf.Variable(np.full((input_shape[0], 1), 2*np.log(self.Initial_params['d'])), dtype = dtype, name = 'Logd2', constraint=lambda x: tf.clip_by_value(x, clipping_value/2, np.inf))
        q  = np.min([self.Initial_params['q'], 0.9*self.Initial_params['d']], 0)
        ratio = q**2/self.Initial_params['d']**2
        self.invsig_q2 = tf.Variable(np.full((input_shape[0], 1), inverse_sigmoid(ratio)), dtype = dtype, name = 'Log_q2', constraint=lambda x: tf.clip_by_value(x, inverse_sigmoid(1e-6), inverse_sigmoid(1-1e-6))) # here we must make sure that q2 is in between 0 and d2, to do so from a weight that can take any real value we use the sigmoid function : q2 = sigmoid(weigth)*d2          
        self.invsig_l = tf.Variable(np.full((input_shape[0], 1), inverse_sigmoid(self.Initial_params['l'])), dtype = dtype, name = 'Log_l2', constraint=lambda x: tf.clip_by_value(x, inverse_sigmoid(1e-6), inverse_sigmoid(1-1e-6)))
        self.Log_LocErr2 = tf.Variable(np.full((input_shape[0], 1), 2*np.log(self.Initial_params['LocErr'])), dtype = dtype, name = 'LogLocErr2', trainable = not self.Fixed_LocErr, constraint=lambda x: tf.clip_by_value(x, clipping_value/2, np.inf))
        self.built = True
       
    def call(self, inputs, nb_dims):
            
        d2 = tf.math.exp(self.Log_d2)
        q2 = self.sigmoid(self.invsig_q2)*d2
        l = self.sigmoid(self.invsig_l)
        LocErr2 =  tf.math.exp(self.Log_LocErr2)
        #print(LocErr2**0.5, d2**0.5, q2**0.5, l)
       
        q02 = d2/(2*l) + LocErr2 # std of the initial well center compared to c0
        #q02 = d2 + LocErr2
        g2 =  LocErr2 * d2 /(LocErr2 + d2)/(1-l)**2
        a = LocErr2 / (LocErr2 + d2)
        b = 1 - a
        ap = a / (1-l)
        bp =  b / (1-l)
        #int over r0:
        k2 = LocErr2 + d2
       
        #int over z0:
        s2 = ((1-l)/l)**2 * (k2 + g2)
        zeta = a / l
        eta = b / l * inputs[:,1] - (1-l)/l * inputs[:,0]
        LP = tf.zeros_like(inputs[:,:1,0]) + nb_dims * tf.math.log((1-l)/l)
        #int over h0:
        x2 = q02 + q2
        alpha = q02  / (zeta * x2)
        gamma = (- eta + q2 / x2 * inputs[:,0])/zeta
        w2 = (s2*q2 + q2*q02 + q02*s2) / (zeta**2 * x2)
        LP += - nb_dims * tf.math.log(zeta)
        u = inputs[:,0]
       
        # int over z1
        m2p = k2 + w2
        tho = l/(1-l) + alpha * k2 / m2p
        s2 = (k2*w2 + w2*g2 + g2*k2)/(m2p * tho**2)
        zeta = ap / tho
        eta = (bp * inputs[:,2] - gamma * k2/m2p - w2/m2p *  inputs[:,1]) / tho
        m2 = m2p / alpha ** 2
        LP += - nb_dims * tf.math.log(tho * alpha * (1-l))
        #int over h1:
        e2 = x2 + m2
        LP += tf.math.reduce_sum(self.log_gaussian((inputs[:,1] - gamma)/alpha - u, e2), axis = 1, keepdims=True)
        u = x2 * (inputs[:,1] - gamma)/(alpha*e2) + m2/e2 * u
        phi2 = x2 * m2 / e2
        x2 = phi2 + q2
        w2 = (phi2*q2 + q2*s2 + s2*phi2)/(x2*zeta**2)
        alpha = phi2/(x2 * zeta)
        gamma = (q2/x2 * u - eta)/zeta
        LP += - nb_dims * tf.math.log(zeta)

        prev_inputs = inputs[:,2]
       
        initial_state = [LP, prev_inputs, x2, w2, alpha, u, gamma]
       
        return inputs, initial_state

    def log_gaussian(self, top, variance):
        return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)
   
    def sigmoid(self, x):
        return 1/(1+tf.math.exp(-x))

    def get_parameters(self):
        d2 = np.exp(self.Log_d2)
        return np.exp(self.Log_LocErr2)**0.5, d2**0.5, np.array((self.sigmoid(self.invsig_q2)*d2)**0.5), np.array(self.sigmoid(self.invsig_l))

class Confinement_RNNCell(tf.keras.layers.Layer):
   
    def __init__(self, state_size, parent, nb_dims, **kwargs):
        self.state_size = state_size # [nb_dims, 1,1]
        self.parent = parent
        self.nb_dims = nb_dims
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
   
    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True

    def call(self, inputs, states): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        prev_output = states
        output = self.Brownian_function(inputs, prev_output, self.nb_dims)
        return output[0], output # the first oupput is the output of the layer, the second is the hidden states
       
    def Brownian_function(self, input_i, prev_output, nb_dims):
       
        d2 = tf.math.exp(self.parent.weights[0])
        q2 = self.sigmoid(self.parent.weights[1])*d2
        l = self.sigmoid(self.parent.weights[2])
        LocErr2 = tf.math.exp(self.parent.weights[3])

        [LP, prev_inputs, x2, w2, alpha, u, gamma] = prev_output
       
        k2 = LocErr2 + d2
        g2 =  LocErr2 * d2 /(LocErr2 + d2)/(1-l)**2
        a = LocErr2 / (LocErr2 + d2)
        b = 1 - a
        ap = a / (1-l)
        bp =  b / (1-l)
       
        # int over z1
        m2p = k2 + w2
        tho = l/(1-l) + alpha * k2 / m2p
        s2 = (k2*w2 + w2*g2 + g2*k2)/(m2p * tho**2)
        zeta = ap / tho
        eta = (bp * input_i - gamma * k2/m2p - w2/m2p *  prev_inputs) / tho
        m2 = m2p / alpha ** 2
        LP += - nb_dims * tf.math.log(tho * alpha * (1-l))
        #int over h1:
        e2 = x2 + m2
       
        LP += tf.math.reduce_sum(self.log_gaussian((prev_inputs - gamma)/alpha - u, e2), axis = -1, keepdims=True)
        u = x2 * (prev_inputs - gamma)/(alpha*e2) + m2/e2 * u
        phi2 = x2 * m2 / e2
        x2 = phi2 + q2
        w2 = (phi2*q2 + q2*s2 + s2*phi2)/(x2*zeta**2)
        alpha = phi2/(x2 * zeta)
        gamma = (q2/x2 * u - eta)/zeta
        LP += - nb_dims * tf.math.log(zeta)
       
        prev_inputs = input_i
       
        output = [LP, prev_inputs, x2, w2, alpha, u, gamma]
        #print(prev_output)
       
        return output # the first oupput is the output of the layer, the second is the hidden states

    def log_gaussian(self, top, variance):
        return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)

    def sigmoid(self, x):
        return 1/(1+tf.math.exp(-x))

class Confinement_Final_layer(tf.keras.layers.Layer):

    def __init__(self, parent, nb_dims, **kwargs):
        self.parent = parent
        self.nb_dims = nb_dims
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
   
    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True
   
    def call(self, inputs, states): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        prev_output = states
        output = self.Brownian_function(inputs, prev_output, self.nb_dims)
        return output # the first oupput is the output of the layer, the second is the hidden states
       
    def Brownian_function(self, inputs, prev_outputs, nb_dims):
       
        [LP, prev_inputs, x2, w2, alpha, u, gamma] = prev_outputs
       
        d2 = tf.math.exp(self.parent.weights[0])
        l = self.sigmoid(self.parent.weights[2])
        LocErr2 = tf.math.exp(self.parent.weights[3])

        #int over r0:
        k2 = LocErr2 + d2
   
        #int over zn-1:
        m2p = k2 + w2
        m2 = m2p / alpha**2
        tho = l/(1-l) + alpha * k2 / m2p
        eta = (inputs/(1-l) - gamma * k2/m2p - w2/m2p * prev_inputs) / tho
        s2 = (k2*w2 + w2*LocErr2/(1-l)**2 + LocErr2/(1-l)**2*k2)/(m2p * tho**2)
        LP += - nb_dims * tf.math.log(tho * alpha * (1-l))
        #int over hn-1:
        v2 = (x2 + s2)
        mu = (s2*u + x2*eta)/v2
        LP += tf.math.reduce_sum(self.log_gaussian(u - eta, v2), axis = -1, keepdims=True) + tf.math.reduce_sum(self.log_gaussian((prev_inputs - gamma)/alpha - mu, (x2*s2 + s2*m2 + m2*x2)/v2), axis = -1, keepdims=True)
        #LP += - (nb_locs-2) * nb_dims * np.log(1-l)
       
        return LP

    def log_gaussian(self, top, variance):
        return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)

    def sigmoid(self, x):
        return 1/(1+tf.math.exp(-x))

def Confined_fit(tracks, verbose = 0, Fixed_LocErr = True, Initial_params = {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01}, nb_epochs = 400):
    '''
    Fit single tracks to a model with brownian diffusion and confinement. If a memory shortage occurs reduce the number of tracks analyzed at once.
    
    Parameters
    ----------
    tracks : numpy array
        3-dimension array of tracks of same length. dims: track ID, time point, space coordinates (x,y,..)
    verbose : int
        tensorflow model fit verbose. The default is 0.
    Fixed_LocErr : bool, optional
        Fix the localization error to its initial value if True. The default is True.
    nb_epochs : int
        Number of epochs for the model fitting. The default is 400.
    Initial_params : dict,
        Dictionary of the initial parameters. The values for each key must be a positive float.
        The default is {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01}.
        The parameters represent the localization error, the diffusion length per step of the particle, 
        the diffusion length per step of the potential well and the confinement factor respectively.
        
    Returns
    -------
    pd_params: pandas data frame
        data frame containing the log likelihood and parameters of individual tracks
        LP: 
            log probability of each track according to the model.
        est_LocErrs: 
            Estiamted localization errors for each state.
        est_ds:
            Estimated d, the diffusion length per step of the particle, for each state (the diffusion coefficient D = d**2/(2dt).
        est_qs: 
            Estimated diffusion lengths per step of the potential well for each state.
        est_ls:
            Estiamted confinement factor of each particle.
    '''
    
    nb_tracks = len(tracks)
    nb_dims = tracks.shape[2]
    input_size = nb_dims
    
    inputs = tf.keras.Input(shape=(None, input_size), batch_size = nb_tracks, dtype = dtype)
    layer1 = Confinement_Initial_layer(Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_params, dtype = dtype)
    tensor1, initial_state = layer1(inputs, input_size)
    cell = Confinement_RNNCell([1, input_size, 1, 1, 1, input_size, input_size], layer1, input_size, dtype = dtype) # n
    RNN_layer = tf.keras.layers.RNN(cell, return_state = True, dtype = dtype)
    tensor2 = RNN_layer(tensor1[:,3:-1], initial_state = initial_state)
    prev_outputs = tensor2[1:]
    layer3 = Confinement_Final_layer(layer1, input_size, dtype = dtype)
    LP = layer3(inputs[:,-1], prev_outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=LP, name="Diffusion_model")
    if verbose > 0:
        model.summary()
        
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        return - tf.math.reduce_sum(y_pred, axis=-1)
    
    # model.compile(loss=MLE_loss, optimizer='adam')
    adam = tf.keras.optimizers.Adam(learning_rate=0.9, beta_1=0.1, beta_2=0.1) # we use a first set of parameters with hight learning_rate and low beta values to accelerate initial learning
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tracks[:nb_tracks,:,:input_size], tracks[:nb_tracks,:,:input_size], epochs = 20, batch_size = nb_tracks, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tracks[:nb_tracks,:,:input_size], tracks[:nb_tracks,:,:input_size], epochs = nb_epochs, batch_size = nb_tracks, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

    LP = model.predict_on_batch(tf.constant(tracks[:nb_tracks,:,:input_size], dtype = dtype))
    est_LocErrs, est_ds, est_qs, est_ls = model.layers[1].get_parameters()
    
    pd_params = pd.DataFrame(np.concatenate((LP, est_LocErrs, est_ds, est_qs, est_ls), axis = 1), columns = ['Log_likelihood', 'LocErr', 'd', 'q', 'l'])
        
    return pd_params#est_LocErrs, est_ds, est_qs, est_ls, LP

class Brownian_Initial_layer_multi(tf.keras.layers.Layer):
    def __init__(
        self,
        nb_states = 1,
        Fixed_LocErr = True,
        Initial_params = {'LocErr': [0.02], 'd': [0.1]}, 
        **kwargs,
    ):
        self.nb_states = nb_states
        self.Fixed_LocErr = Fixed_LocErr
        self.Initial_params = Initial_params
        super().__init__(**kwargs)
       
    def build(self, input_shape):
        self.Logd2 = tf.Variable(2*np.log([self.Initial_params['d']])[:,:,None], dtype = dtype, name = 'Logd2', constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        self.LogLocErr2 = tf.Variable(2*np.log([self.Initial_params['LocErr']])[:,:,None], dtype = dtype, name = 'LogLocErr2', trainable = not self.Fixed_LocErr, constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        self.built = True
   
    def call(self, inputs):
        init_mu = tf.repeat(inputs[:,0], self.nb_states, axis = 1) # initial most likely real postion
        init_s2 = tf.math.exp(self.LogLocErr2) + tf.math.exp(self.Logd2) # initial variance
        #init_LP = tf.zeros((inputs.shape[0], nb_states))
        init_LP = tf.repeat(tf.zeros_like(inputs[:,:1,0, 0], dtype = dtype), self.nb_states, axis = 1)
        initial_state = [init_mu, init_s2, init_LP]
        return inputs, initial_state

    def get_parameters(self):
        return np.exp(self.LogLocErr2)[0]**0.5, np.exp(self.Logd2)[0]**0.5

#@keras.utils.register_keras_serializable
class Brownian_RNNCell_multi(tf.keras.layers.Layer):
   
    def __init__(self, state_size, parent, **kwargs):
        self.state_size = state_size # [nb_dims, 1,1]
        self.parent = parent
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
   
    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True

    def call(self, inputs, states): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        prev_output = states
        output = self.Brownian_function(inputs, prev_output)
        return output[2], output # the first oupput is the output of the layer, the second is the hidden states
       
    def Brownian_function(self, input_i, prev_output):
        d2 = tf.math.exp(self.parent.weights[0])
        LocErr2 = tf.math.exp(self.parent.weights[1])
        mu, s2, LP = prev_output
        #print(LP, mu, s2)
        variance = LocErr2 + s2
        top = input_i - mu
        #print('LP', LP)
        #print('variance', variance)
        #print('top', top)
   
        #print('log_gaussian(top, variance)', log_gaussian(top, variance))
        new_LP = LP + tf.math.reduce_sum(self.log_gaussian(top, variance), axis = -1, keepdims = False)
        new_mu = (mu * LocErr2 + input_i * s2)/variance
        new_s2 = (d2*s2 + s2*LocErr2 + LocErr2*d2)/variance
        output = [new_mu, new_s2, new_LP]
        return output
       
    def log_gaussian(self, top, variance):
        return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)

def Brownian_fit_multi(tracks, nb_states = 3, verbose = 1, Fixed_LocErr = False, nb_epochs = 400, batch_size = 5000, Initial_params = {'LocErr': [0.02, 0.022, 0.023], 'd': [0.05, 0.08, 0.1]}):
    '''
    Fit a population of tracks to a model with brownian diffusion (and localization error) with a number of states specified by `nb_states`.
    
    Parameters
    ----------
    tracks : numpy array
        3-dimension array of tracks of same length. dims: track ID, time point, space coordinates (x,y,..)
    nb_states : int
        Number of states assumed by the model. 
    verbose : int
        tensorflow model fit verbose. The default is 0.
    Fixed_LocErr : bool, optional
        Fix the localization error to its initial value if True. The default is True.
    nb_epochs : int
        Number of epochs for the model fitting. The default is 400.
    batch_size : int
        Number of tracks considered ber batch. The default is 5000.
    Initial_params : dict,
        Dictionary of the initial parameters. The values for each key must be a list of floats of length `nb_states`.
        The default is {'LocErr': [0.02, 0.022, 0.023], 'd': [0.05, 0.08, 0.1]}.
        The parameters represent the localization error and the diffusion length per step respectively.
        
    Returns
    -------
    est_LocErrs: 
        estiamted localization errors for each state 
    est_ds:
        estimated d, the diffusion length per step of the particle, for each state (the diffusion coefficient D = d**2/(2dt)
    est_qs: 
        estimated diffusion lengths per step of the potential well for each state.
    est_ls:
        estiamted standard deviation of the initial speed of the particle
    sum_LP: log probability of the data set.
    '''
    
    if  len(Initial_params['d']) != nb_states or len(Initial_params['LocErr']) != nb_states :
        print("the length of the initial parameter list `Initial_params` is not matching the number of states nb_states, instead, this code will use randomly distibuted initial parameters")
        for param in Initial_params:
            Initial_params[param] = np.random.rand(nb_states)*3*Initial_params[param][0]
    
    nb_dims = tracks.shape[2]
    nb_tracks = tracks.shape[0]
    
    tf_tracks = tf.constant(tracks[:nb_tracks,:,None, :nb_dims], dtype = dtype)

    inputs = tf.keras.Input(shape=(None, 1, nb_dims), dtype = dtype)
    layer1 = Brownian_Initial_layer_multi(nb_states = nb_states, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_params, dtype = dtype)
    tensor1, initial_state = layer1(inputs)
    
    #cell = Brownian_RNNCell_multi([(nb_states, nb_dims), (nb_states, 1), (nb_states, 1)], layer1) # n
    cell = Brownian_RNNCell_multi([tf.TensorShape([nb_states, nb_dims]), tf.TensorShape([nb_states, 1]), tf.TensorShape([nb_states])], layer1, dtype = dtype) # n
    RNN_layer = tf.keras.layers.RNN(cell, dtype = dtype)
    outputs = RNN_layer(tensor1[:,1:], initial_state = initial_state)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Diffusion_model")
    if verbose > 0:
        model.summary()
    
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        #print(y_pred)
        nb_states = y_pred.shape[1]
        max_LPs = tf.math.reduce_max(y_pred, 1, keepdims = True)
        sum_LP_layers = tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred - max_LPs), axis=-1, keepdims = False)/nb_states) + max_LPs[:, 0]
        return - sum_LP_layers # sum over the spatial dimensions axis
    
    # model.compile(loss=MLE_loss, optimizer='adam')
    adam = tf.keras.optimizers.Adam(learning_rate=0.9, beta_1=0.1, beta_2=0.1) # we use a first set of parameters with hight learning_rate and low beta values to accelerate initial learning
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = 20, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    #plt.plot(history.history['loss'])
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.1, beta_2=0.1) # we use a first set of parameters with hight learning_rate and low beta values to accelerate initial learning
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = 30, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    #plt.plot(history.history['loss'])

    adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = nb_epochs, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    #plt.plot(history.history['loss'])

    adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = nb_epochs, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

    LPs = model.predict_on_batch(tf_tracks)
    
    sum_LP = MLE_loss(LPs, LPs)
    
    est_LocErrs, est_ds = model.layers[1].get_parameters()
    return est_LocErrs, est_ds, sum_LP

class Directed_Initial_layer_multi(tf.keras.layers.Layer):
    def __init__(
        self,
        nb_states = 3,
        Fixed_LocErr = True,
        Initial_params = {'LocErr': [0.02, 0.022, 0.022], 'd': [0.1, 0.12, 0.12], 'q': [0.01, 0.012, 0.012], 'l': [0.01, 0.02, 0.012]},
        **kwargs,
    ):
        self.nb_states = nb_states
        self.Fixed_LocErr = Fixed_LocErr
        self.Initial_params = Initial_params
        super().__init__(**kwargs)
       
    def build(self, input_shape):
       
        def inverse_sigmoid(x):
            return tf.math.log(x/(1-x))
        
        
        self.Log_d2 = tf.Variable(2*np.log([self.Initial_params['d']])[:,:,None], dtype = dtype, name = 'Log_d2', constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        self.Log_q2 = tf.Variable(2*np.log([self.Initial_params['q']])[:,:,None], dtype = dtype, name = 'Log_q2', constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        self.Log_l2 = tf.Variable(2*np.log([self.Initial_params['l']])[:,:,None], dtype = dtype, name = 'Log_l2', constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        self.Log_LocErr2 = tf.Variable(2*np.log([self.Initial_params['LocErr']])[:,:,None], dtype = dtype, name = 'Log_LocErr2', trainable = not self.Fixed_LocErr, constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        
        #self.Log_d2 = tf.Variable(2*np.log([[self.Initial_params['d']]]), dtype = 'float32', name = 'Log_d2')
        #self.Log_q2 = tf.Variable(2*np.log([[self.Initial_params['q']]]), dtype = 'float32', name = 'Log_q2')
        #self.Log_l2 = tf.Variable(2*np.log([[self.Initial_params['l']]]), dtype = 'float32', name = 'Log_l2')
        #self.Log_LocErr2 = tf.Variable(2*np.log([[self.Initial_params['LocErr']]]), dtype = 'float32', name = 'Log_LocErr2', trainable = not self.Fixed_LocErr )
        self.built = True
       
    def call(self, inputs, nb_dims):
       
        LocErr2 =  tf.math.exp(self.Log_LocErr2)
        d2 = tf.math.exp(self.Log_d2)
        q2 = tf.math.exp(self.Log_q2)
        l2 = tf.math.exp(self.Log_l2)
        #print(LocErr2**0.5, d2**0.5, q2**0.5, l2**0.5)
        
        #LP = tf.repeat(tf.zeros_like(inputs[:,:1,0, 0]), self.nb_states, axis = 1)
        LP = tf.zeros_like(inputs[:,0,:, 0], dtype = dtype)
        #step 0:
        s2i = LocErr2 + d2
        x2i = l2 + q2
        g2i = (q2 *  s2i + s2i * l2 + l2 * q2) / x2i
        LocErr2_g2i = LocErr2 + g2i
        alphai = l2 / x2i
        gammai = inputs[:, 0]
        h2i = l2
        #step 1:
        ki = (inputs[:, 1] - gammai) / alphai
        h2i = (g2i + LocErr2) / alphai**2
        LP += - nb_dims * tf.math.log(alphai[:,:,0])
        #fs1
        ai = LocErr2 * alphai / LocErr2_g2i + 1
        a2i = ai**2
        bi = (g2i * inputs[:, 1] + LocErr2 * gammai) / LocErr2_g2i
        s2i = (LocErr2 * g2i + g2i * d2 + d2 * LocErr2) / (LocErr2_g2i * a2i)
        #ft1
        betai = x2i / (x2i + q2)
        t2i = betai * q2
        #chi1
        x2i = x2i + q2
        #G1
        h2i_t2i = h2i + t2i
        alphai = ai * betai * h2i / h2i_t2i
        gammai = bi + ai * t2i * ki / h2i_t2i
        g2i = a2i * (t2i *  s2i + s2i * h2i + h2i * t2i) / h2i_t2i
        #Q1
        kim1 = ki / betai
        q2i = (h2i + t2i) / betai**2
        LP += - nb_dims * tf.math.log(betai[:,:,0])
       
        initial_state = [LP, alphai, gammai, g2i, q2i, kim1, x2i]
       
        return inputs, initial_state
    
    def get_parameters(self):
        return np.exp(self.Log_LocErr2)[0]**0.5, np.exp(self.Log_d2)[0]**0.5, np.exp(self.Log_q2)[0]**0.5, np.exp(self.Log_l2)[0]**0.5*2**0.5

class Directed_RNNCell_multi(tf.keras.layers.Layer):
    
    def __init__(self, state_size, parent, nb_dims, **kwargs):
        self.state_size = state_size # [nb_dims, 1,1]
        self.parent = parent
        self.nb_dims = nb_dims
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True

    def call(self, inputs, states): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        prev_output = states
        output = self.directed_motion_function(inputs, prev_output, self.nb_dims)
        return output[0], output # the first oupput is the output of the layer, the second is the hidden states
    
    def directed_motion_function(self, input_i, prev_output, nb_dims):
        
        d2 = tf.math.exp(self.parent.weights[0])
        q2 = tf.math.exp(self.parent.weights[1])
        LocErr2 = tf.math.exp(self.parent.weights[3])
        
        [LP, alphai, gammai, g2i, q2i, kim1, x2i] = prev_output
        
        ki = (input_i - gammai) / alphai
        h2i = (g2i + LocErr2) / alphai**2
        LP += - nb_dims * tf.math.log(alphai[:,:,0])
        #fsi
        ai = LocErr2 * alphai /(LocErr2 + g2i) + 1
        a2i = ai**2
        bi = (g2i * input_i + LocErr2 * gammai) / (LocErr2 + g2i)
        s2i = (LocErr2 * g2i + g2i * d2 + d2 * LocErr2) / ((LocErr2 + g2i) * a2i)
        # int of Q_{i-1}(w_i - k_{i-1}/beta_{i-1}) chi_{i-1}(w_i) K_i(w_i - k_i) fsi(r_{i+1} - (a_i * w_i + b_i)) fdz(w_{i+1} - w_i)    we need to reduce the 5 terms to 3: fusion of Q and K to L and K' and chi_{i-1} and fdz to chi_i and fti
        # fusion of Qi-1 and Ki (Ki * Qi-1 -> Li * Ki)
        h2i_q2i = h2i + q2i
        LP += tf.math.reduce_sum(self.log_gaussian(ki - kim1, h2i_q2i), axis = -1, keepdims=False)
        ki = (h2i * kim1 + q2i * ki) / h2i_q2i
        h2i = h2i * q2i / h2i_q2i
        #fti
        x2i_q2 = x2i + q2
        betai = x2i / x2i_q2
        t2i = betai * q2
        #chii
        x2i = x2i_q2
        #Gi
        h2i_t2i = h2i + t2i
        alphai = ai * betai * h2i / h2i_t2i
        gammai =  bi + ai * t2i * ki / h2i_t2i
        g2i = a2i * (t2i * s2i + s2i * h2i + h2i * t2i) / h2i_t2i
        #LocErr2_g2i = LocErr2 + g2i
        #Qi
        kim1 = ki / betai
        q2i = h2i_t2i / betai**2
        LP += - nb_dims * tf.math.log(betai[:,:,0])
       
        output = [LP, alphai, gammai, g2i, q2i, kim1, x2i]
        #print(prev_output)
       
        return output # the first oupput is the output of the layer, the second is the hidden states

    def log_gaussian(self, top, variance):
        return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)

class Directed_Final_layer_multi(tf.keras.layers.Layer):

    def __init__(self, parent, nb_dims, **kwargs):
        self.parent = parent
        self.nb_dims = nb_dims
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
   
    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True
   
    def call(self, inputs, states): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        prev_output = states
        output = self.directed_motion_function(inputs, prev_output, self.nb_dims)
        return output # the first oupput is the output of the layer, the second is the hidden states
       
    def directed_motion_function(self, inputs, prev_outputs, nb_dims):
       
        [LP, alphai, gammai, g2i, q2i, kim1, x2i] = prev_outputs

        d2 = tf.math.exp(self.parent.weights[0])
        LocErr2 = tf.math.exp(self.parent.weights[3])

        #int over r0:
        #step n-1: (nothing changes for the dr int but the dw int does not have any fdz term)
        #i = nb_locs - 2
        #Ki
        ki = (inputs[:, -2] - gammai) / alphai
        LocErr2_g2i = LocErr2 + g2i
        h2i = LocErr2_g2i / alphai**2
        LP += - nb_dims * tf.math.log(alphai[:,:,0])
        #fsi
        ai = LocErr2 * alphai /LocErr2_g2i + 1
        a2i = ai**2
        bi = (g2i * inputs[:, -2] + LocErr2 * gammai) / LocErr2_g2i
        s2i = (LocErr2 * g2i + g2i * d2 + d2 * LocErr2) / (LocErr2_g2i * a2i)
        # fusion of Qi-1 and Ki (Ki * Qi-1 -> Li * Ki)
        h2i_q2i = h2i + q2i
        LP += tf.math.reduce_sum(self.log_gaussian(ki - kim1, h2i_q2i), axis = -1, keepdims=False)
        ki = (h2i * kim1 + q2i * ki) / h2i_q2i
               
        h2i = h2i * q2i / h2i_q2i
        # int of chi_{i-1}(w_i) K_i(w_i - k_i) fsi(r_{i+1} - (a_i * w_i + b_i))
        #Qn-1
        h2i_x2i = h2i + x2i
        LP += tf.math.reduce_sum(self.log_gaussian(ki, h2i_x2i), axis = -1, keepdims=False)

        gammai = bi + ai * x2i * ki / h2i_x2i
        g2i = a2i * (x2i * s2i + s2i * h2i + h2i * x2i) / h2i_x2i
        LocErr2_g2i = LocErr2 + g2i
       
        #i = nb_locs - 1
        LP += tf.math.reduce_sum(self.log_gaussian(inputs[:, -1] - gammai, LocErr2_g2i), axis = -1, keepdims=False)
        return LP

    def log_gaussian(self, top, variance):
        return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)

def Directed_fit_multi(tracks, nb_states = 3, verbose = 0, Fixed_LocErr = True, nb_epochs = 300, batch_size = 5000, Initial_params = {'LocErr': [0.02, 0.022, 0.022], 'd': [0.1, 0.12, 0.12], 'q': [0.01, 0.012, 0.012], 'l': [0.01, 0.02, 0.012]}):
    '''
    Fit a population of tracks to a model with diffusion plus directed motion with a number of states specified by `nb_states`.
    
    verbose 
    Parameters
    ----------
    tracks : numpy array
        3-dimension array of tracks of same length. dims: track ID, time point, space coordinates (x,y,..)
    nb_states : int
        Number of states assumed by the model. 
    verbose : int
        tensorflow model fit verbose. The default is 0.
    Fixed_LocErr : bool, optional
        Fix the localization error to its initial value if True. The default is True.
    nb_epochs : int
        Number of epochs for the model fitting. The default is 300.
    batch_size : int
        Number of tracks considered ber batch. The default is 5000.
    Initial_params : dict,
        Dictionary of the initial parameters. The values for each key must be a list of floats of length `nb_states`.
        The default is {'LocErr': [0.02, 0.022, 0.022], 'd': [0.1, 0.12, 0.12], 'q': [0.01, 0.012, 0.012], 'l': [0.01, 0.02, 0.012]}.
        The parameters represent the localization error, the diffusion length per step, the change per step of the x, y speed of the
        directed motion and the standard deviation of the initial speed of the particle. 
    Returns
    -------
    est_LocErrs: 
        estiamted localization errors for each state 
    est_ds:
        estimated d, the diffusion length per step of the particle, for each state (the diffusion coefficient D = d**2/(2dt)
    est_qs: 
        estimated diffusion lengths per step of the potential well for each state.
    est_ls:
        estiamted standard deviation of the initial speed of the particle
    sum_LP: log probability of the data set.
    '''
    if len(Initial_params['d']) != nb_states or len(Initial_params['LocErr']) != nb_states or len(Initial_params['q']) != nb_states or len(Initial_params['l']) != nb_states:
        print("the length of the initial parameter list `Initial_params` is not matching the number of states nb_states, instead, this code will use randomly distibuted initial parameters")
        for param in Initial_params:
            Initial_params[param] = np.random.rand(nb_states)*3*Initial_params[param][0]
    
    track_len = tracks.shape[1]
    nb_dims = tracks.shape[2]
    nb_tracks = tracks.shape[0]
    
    tf_tracks = tf.constant(tracks[:nb_tracks,:,None, :nb_dims], dtype = dtype)

    nb_tracks = len(tracks)
    if track_len > 4:
        inputs = tf.keras.Input(shape=(None, 1, nb_dims), dtype = dtype)
        layer1 = Directed_Initial_layer_multi(nb_states = 3, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_params, dtype = dtype)
        tensor1, initial_state = layer1(inputs, nb_dims)
        cell = Directed_RNNCell_multi([tf.TensorShape([nb_states]), tf.TensorShape([nb_states, 1]), tf.TensorShape([nb_states, nb_dims]), tf.TensorShape([nb_states, 1]), tf.TensorShape([nb_states, 1]), tf.TensorShape([nb_states, nb_dims]), tf.TensorShape([nb_states, 1])], layer1, nb_dims, dtype = dtype) # n
        RNN_layer = tf.keras.layers.RNN(cell, return_state = True, dtype = dtype)
        tensor2 = RNN_layer(tensor1[:,2:-2], initial_state = initial_state)
        prev_outputs = tensor2[1:]
        layer3 = Directed_Final_layer_multi(layer1, nb_dims, dtype = dtype)
        #LP, estimated_ds, estimated_qs, estimated_ls, estimated_LocErrs = layer3(inputs[:,:], prev_outputs)
        LP = layer3(inputs[:,:], prev_outputs)
    elif track_len <= 4:
        inputs = tf.keras.Input(shape=(None, 1, nb_dims), dtype = dtype)
        layer1 = Directed_Initial_layer_multi(Fixed_LocErr = Fixed_LocErr, Initial_params = {'LocErr': [0.02, 0.022, 0.022], 'd': [0.1, 0.12, 0.12], 'q': [0.01, 0.012, 0.012], 'l': [0.01, 0.02, 0.012]}, dtype = dtype)
        tensor1, initial_state = layer1(inputs, nb_dims)
        prev_outputs = initial_state
        layer3 = Directed_Final_layer_multi(layer1, nb_dims, dtype = dtype)
        LP = layer3(inputs[:,:], prev_outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=LP, name="Diffusion_model")
    
    if verbose > 0:
        model.summary()
    
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        #print(y_pred)
        nb_states = y_pred.shape[1]
        max_LPs = tf.math.reduce_max(y_pred, 1, keepdims = True)
        sum_LP_layers = tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred - max_LPs), axis=-1, keepdims = False)/nb_states) + max_LPs[:, 0]
        return - sum_LP_layers # sum over the spatial dimensions axis
    
    #plt.figure()
    # model.compile(loss=MLE_loss, optimizer='adam')
    adam = tf.keras.optimizers.Adam(learning_rate=0.9, beta_1=0.1, beta_2=0.1) # we use a first set of parameters with hight learning_rate and low beta values to accelerate initial learning
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = 20, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    #plt.plot(history.history['loss'])
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.1, beta_2=0.1) # we use a first set of parameters with hight learning_rate and low beta values to accelerate initial learning
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = 30, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    #plt.plot(history.history['loss'])

    adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = nb_epochs, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    #plt.plot(history.history['loss'])

    adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = nb_epochs, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    #plt.plot(history.history['loss'])

    #plt.figure()
    #plt.plot(history.history['loss'])
    LPs = model.predict_on_batch(tf_tracks)
    sum_LP = MLE_loss(LPs, LPs)
    
    est_LocErrs, est_ds, est_qs, est_ls = layer1.get_parameters()
    return est_LocErrs, est_ds, est_qs, est_ls, sum_LP

class Confinement_Initial_layer_multi(tf.keras.layers.Layer):
    def __init__(
        self,
        nb_states = 3,
        Fixed_LocErr = True,
        Initial_params = {'LocErr': [0.02, 0.022, 0.022], 'd': [0.1, 0.12, 0.12], 'q': [0.01, 0.012, 0.012], 'l': [0.01, 0.02, 0.012]},
        **kwargs,
    ):
        self.nb_states = nb_states
        self.Fixed_LocErr = Fixed_LocErr
        self.Initial_params = Initial_params
        super().__init__(**kwargs)
       
    def build(self, input_shape):
       
        def inverse_sigmoid(x):
            return np.log(x/(1-x))
       
        d = np.array([self.Initial_params['d']])
        q = np.min([[self.Initial_params['q'], 0.9*d[0]]], 1)
        l = np.array([self.Initial_params['l']])
        LocErr = np.array([self.Initial_params['LocErr']])

        self.Log_d2 = tf.Variable(2*np.log(d)[:,:,None], dtype = dtype, name = 'Log_d2', constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        q = np.min([q, 0.9*d], 0)
        self.invsig_q2 = tf.Variable(inverse_sigmoid(q**2/d**2)[:,:,None], dtype = dtype, name = 'Log_q2', constraint=lambda x: tf.clip_by_value(x, inverse_sigmoid(1e-10), inverse_sigmoid(1-1e-10)))
        self.invsig_l = tf.Variable(inverse_sigmoid(l)[:,:,None], dtype = dtype, name = 'Log_l2', constraint=lambda x: tf.clip_by_value(x, inverse_sigmoid(1e-10), inverse_sigmoid(1-1e-10)))
        self.Log_LocErr2 = tf.Variable(2*np.log(LocErr)[:,:,None], dtype = dtype, name = 'Log_LocErr2', trainable = not self.Fixed_LocErr, constraint=lambda x: tf.clip_by_value(x, clipping_value, np.inf))
        
        #self.Log_d2 = tf.Variable(2*np.log([[self.Initial_params['d']]]), dtype = 'float32', name = 'Log_d2')
        #q = np.min([self.Initial_params['q'], 0.9*self.Initial_params['d']])
        #self.invsig_q2 = tf.Variable([[inverse_sigmoid(q**2/self.Initial_params['d']**2)]], dtype = 'float32', name = 'invsig_q2')
        #self.invsig_l = tf.Variable([[inverse_sigmoid(self.Initial_params['l'])]], dtype = 'float32', name = 'invsig_l')
        #self.Log_LocErr2 = tf.Variable(2*np.log([[self.Initial_params['LocErr']]]), dtype = 'float32', name = 'Log_LocErr2', trainable = not self.Fixed_LocErr)
        #self.built = True
       
    def call(self, inputs, nb_dims):
       
        LocErr2 =  tf.math.exp(self.Log_LocErr2)
        d2 = tf.math.exp(self.Log_d2)
        q2 = self.sigmoid(self.invsig_q2)*d2
        l = self.sigmoid(self.invsig_l)
        #print(LocErr2**0.5, d2**0.5, q2**0.5, l)
       
        q02 = d2/(2*l) # std of the initial well center compared to c0
        g2 =  LocErr2 * d2 /(LocErr2 + d2)/(1-l)**2
        a = LocErr2 / (LocErr2 + d2)
        b = 1 - a
        ap = a / (1-l)
        bp =  b / (1-l)
        #int over r0:
        k2 = LocErr2 + d2
       
        #int over z0:
        s2 = ((1-l)/l)**2 * (k2 + g2)
        zeta = a / l
        eta = b / l * inputs[:,1] - (1-l)/l * inputs[:,0]
        LP = tf.zeros_like(inputs[:,0,:, 0], dtype = dtype) + nb_dims * tf.math.log((1-l)/l)[:,:,0]
        #int over h0:
        x2 = q02 + q2
        alpha = q02  / (zeta * x2)
        gamma = (- eta + q2 / x2 * inputs[:,0])/zeta
        w2 = (s2*q2 + q2*q02 + q02*s2) / (zeta**2 * x2)
        LP += - nb_dims * tf.math.log(zeta)[:,:,0]
        u = inputs[:,0]
       
        # int over z1
        m2p = k2 + w2
        tho = l/(1-l) + alpha * k2 / m2p
        s2 = (k2*w2 + w2*g2 + g2*k2)/(m2p * tho**2)
        zeta = ap / tho
        eta = (bp * inputs[:,2] - gamma * k2/m2p - w2/m2p *  inputs[:,1]) / tho
        m2 = m2p / alpha ** 2
        LP += - nb_dims * tf.math.log(tho * alpha * (1-l))[:,:,0]
        #int over h1:
        e2 = x2 + m2
        LP += tf.math.reduce_sum(self.log_gaussian((inputs[:,1] - gamma)/alpha - u, e2), axis = -1, keepdims=False)
        u = x2 * (inputs[:,1] - gamma)/(alpha*e2) + m2/e2 * u
        phi2 = x2 * m2 / e2
        x2 = phi2 + q2
        w2 = (phi2*q2 + q2*s2 + s2*phi2)/(x2*zeta**2)
        alpha = phi2/(x2 * zeta)
        gamma = (q2/x2 * u - eta)/zeta
        LP += - nb_dims * tf.math.log(zeta[:,:,0])

        prev_inputs = inputs[:,2]
       
        initial_state = [LP, prev_inputs, x2, w2, alpha, u, gamma]
       
        return inputs, initial_state

    def log_gaussian(self, top, variance):
        return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)
   
    def sigmoid(self, x):
        return 1/(1+tf.math.exp(-x))
    
    def get_parameters(self):
        d2 = np.exp(self.Log_d2)
        l = np.array(self.sigmoid(self.invsig_l))[0]
        cor_l = -np.log(1-l)
        cor_d = l/cor_l * d2[0]**0.5
        return np.exp(self.Log_LocErr2)[0]**0.5, cor_d, np.array((self.sigmoid(self.invsig_q2)*d2)**0.5)[0], cor_l

class Confinement_RNNCell_multi(tf.keras.layers.Layer):
   
    def __init__(self, state_size, parent, nb_dims, **kwargs):
        self.state_size = state_size # [nb_dims, 1,1]
        self.parent = parent
        self.nb_dims = nb_dims
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
   
    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True
    
    def call(self, input_i, states): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        prev_output = states
        output = self.Brownian_function(input_i, prev_output, self.nb_dims)
        return output[0], output # the first oupput is the output of the layer, the second is the hidden states
       
    def Brownian_function(self, input_i, prev_output, nb_dims):
       
        d2 = tf.math.exp(self.parent.weights[0])
        q2 = self.sigmoid(self.parent.weights[1])*d2
        l = self.sigmoid(self.parent.weights[2])
        LocErr2 = tf.math.exp(self.parent.weights[3])

        [LP, prev_inputs, x2, w2, alpha, u, gamma] = prev_output
       
        k2 = LocErr2 + d2
        g2 =  LocErr2 * d2 /(LocErr2 + d2)/(1-l)**2
        a = LocErr2 / (LocErr2 + d2)
        b = 1 - a
        ap = a / (1-l)
        bp =  b / (1-l)
       
        # int over z1
        m2p = k2 + w2
        tho = l/(1-l) + alpha * k2 / m2p
        s2 = (k2*w2 + w2*g2 + g2*k2)/(m2p * tho**2)
        zeta = ap / tho
        eta = (bp * input_i - gamma * k2/m2p - w2/m2p *  prev_inputs) / tho
        m2 = m2p / alpha ** 2
        LP += - nb_dims * tf.math.log(tho * alpha * (1-l))[:,:,0]
        #int over h1:
        e2 = x2 + m2
        
        LP += tf.math.reduce_sum(self.log_gaussian((prev_inputs - gamma)/alpha - u, e2), axis = -1, keepdims=False)
        u = x2 * (prev_inputs - gamma)/(alpha*e2) + m2/e2 * u
        phi2 = x2 * m2 / e2
        x2 = phi2 + q2
        w2 = (phi2*q2 + q2*s2 + s2*phi2)/(x2*zeta**2)
        alpha = phi2/(x2 * zeta)
        gamma = (q2/x2 * u - eta)/zeta
        LP += - nb_dims * tf.math.log(zeta)[:,:,0]
       
        prev_inputs = input_i
       
        output = [LP, prev_inputs, x2, w2, alpha, u, gamma]
        #print(prev_output)
       
        return output # the first oupput is the output of the layer, the second is the hidden states

    def log_gaussian(self, top, variance):
        return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)

    def sigmoid(self, x):
        return 1/(1+tf.math.exp(-x))

class Confinement_Final_layer_multi(tf.keras.layers.Layer):

    def __init__(self, parent, nb_dims, **kwargs):
        self.parent = parent
        self.nb_dims = nb_dims
        #self.constraints = positive_Constrain()
        super().__init__(**kwargs)
   
    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True
   
    def call(self, inputs, states): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        prev_output = states
        output = self.Brownian_function(inputs, prev_output, self.nb_dims)
        return output # the first oupput is the output of the layer, the second is the hidden states
       
    def Brownian_function(self, inputs, prev_outputs, nb_dims):
       
        [LP, prev_inputs, x2, w2, alpha, u, gamma] = prev_outputs
       
        d2 = tf.math.exp(self.parent.weights[0]) 
        l = self.sigmoid(self.parent.weights[2])
        LocErr2 = tf.math.exp(self.parent.weights[3])

        #int over r0:
        k2 = LocErr2 + d2
   
        #int over zn-1:
        m2p = k2 + w2
        m2 = m2p / alpha**2
        tho = l/(1-l) + alpha * k2 / m2p
        eta = (inputs/(1-l) - gamma * k2/m2p - w2/m2p * prev_inputs) / tho
        s2 = (k2*w2 + w2*LocErr2/(1-l)**2 + LocErr2/(1-l)**2*k2)/(m2p * tho**2)
        LP += - nb_dims * tf.math.log(tho * alpha * (1-l))[:,:,0]
        #int over hn-1:
        v2 = (x2 + s2)
        mu = (s2*u + x2*eta)/v2
        LP += tf.math.reduce_sum(self.log_gaussian(u - eta, v2), axis = -1, keepdims=False) + tf.math.reduce_sum(self.log_gaussian((prev_inputs - gamma)/alpha - mu, (x2*s2 + s2*m2 + m2*x2)/v2), axis = -1, keepdims=False)
        #LP += - (nb_locs-2) * nb_dims * np.log(1-l)
        return LP

    def log_gaussian(self, top, variance):
        return - 0.5*tf.math.log(tf.constant(2*np.pi, dtype = dtype)*variance) - (top)**2/(2*variance)
    
    def sigmoid(self, x):
        return 1/(1+tf.math.exp(-x))

def Confined_fit_multi(tracks, nb_states = 3, verbose = 0, Fixed_LocErr = True, nb_epochs = 500, batch_size = 5000, Initial_params = {'LocErr': [0.02, 0.022, 0.022], 'd': [0.1, 0.12, 0.12], 'q': [0.01, 0.012, 0.012], 'l': [0.01, 0.02, 0.012]}):
    '''
    Fit a population of tracks to a model with a number of states specified by `nb_states`.
    
    verbose 
    Parameters
    ----------
    tracks : numpy array
        3-dimension array of tracks of same length. dims: track ID, time point, space coordinates (x,y,..)
    nb_states : int
        Number of states assumed by the model. 
    verbose : int
        tensorflow model fit verbose. The default is 0.
    Fixed_LocErr : bool, optional
        Fix the localization error to its initial value if True. The default is True.
    nb_epochs : int
        Number of epochs for the model fitting. The default is 300.
    batch_size : int
        Number of tracks considered ber batch. The default is 5000.
    Initial_params : dict,
        Dictionary of the initial parameters. The values for each key must be a list of floats of length `nb_states`.
        The default is {'LocErr': [0.02, 0.022, 0.022], 'd': [0.1, 0.12, 0.12], 'q': [0.01, 0.012, 0.012], 'l': [0.01, 0.02, 0.012]}.
        The parameters represent the localization errors, the diffusion lengths per step of the particle, 
        the diffusion lengths per step of the potential well and the confinement factors respectively.
    
    Returns
    -------
    est_LocErrs: 
        estiamted localization errors for each state 
    est_ds:
        estimated d, the diffusion length per step of the particle, for each state (the diffusion coefficient D = d**2/(2dt)
    est_qs: 
        estimated diffusion lengths per step of the potential well for each state.
    est_ls:
        estiamted confinement factor for each state
    sum_LP: log probability of the data set.
    '''
    
    if len(Initial_params['d']) != nb_states or len(Initial_params['LocErr']) != nb_states or len(Initial_params['q']) != nb_states or len(Initial_params['l']) != nb_states:
        print("the length of the initial parameter list `Initial_params` is not matching the number of states nb_states, instead, this code will use randomly distibuted initial parameters")
        for param in Initial_params:
            Initial_params[param] = np.random.rand(nb_states)*3*Initial_params[param][0]
    
    nb_tracks, track_len, nb_dims = tracks.shape
    
    tf_tracks = tf.constant(tracks[:nb_tracks,:,None, :nb_dims], dtype = dtype)
    inputs = tf_tracks
    
    inputs = tf.keras.Input(shape=(None, 1, nb_dims), dtype = dtype)
    layer1 = Confinement_Initial_layer_multi(nb_states = nb_states, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_params, dtype = dtype)
    tensor1, initial_state = layer1(inputs, nb_dims)
    cell = Confinement_RNNCell_multi([tf.TensorShape([nb_states]), tf.TensorShape([1, nb_dims]), tf.TensorShape([nb_states, 1]), tf.TensorShape([nb_states, 1]), tf.TensorShape([nb_states, 1]), tf.TensorShape([nb_states, nb_dims]), tf.TensorShape([nb_states, nb_dims])], layer1, nb_dims, dtype = dtype) # n
    RNN_layer = tf.keras.layers.RNN(cell, return_state = True, dtype = dtype)
    tensor2 = RNN_layer(tensor1[:,3:-1], initial_state = initial_state)
    prev_outputs = tensor2[1:]
    layer3 = Confinement_Final_layer_multi(layer1, nb_dims, dtype = dtype)
    LP = layer3(inputs[:,-1], prev_outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=LP, name="Diffusion_model")
    if verbose > 0:
        model.summary()
    
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        #print(y_pred)
        nb_states = y_pred.shape[1]
        max_LPs = tf.math.reduce_max(y_pred, 1, keepdims = True)
        sum_LP_layers = tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred - max_LPs), axis=-1, keepdims = False)/nb_states) + max_LPs[:, 0]
        return - sum_LP_layers # sum over the spatial dimensions axis
    
    #plt.figure()
    # model.compile(loss=MLE_loss, optimizer='adam')
    adam = tf.keras.optimizers.Adam(learning_rate=0.9, beta_1=0.1, beta_2=0.1) # we use a first set of parameters with hight learning_rate and low beta values to accelerate initial learning
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = 20, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    #plt.plot(history.history['loss'])
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.1, beta_2=0.1) # we use a first set of parameters with hight learning_rate and low beta values to accelerate initial learning
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = 30, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    #plt.plot(history.history['loss'])

    adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = nb_epochs, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    #plt.plot(history.history['loss'])
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tf_tracks, tf_tracks, epochs = nb_epochs, batch_size = batch_size, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    
    LPs = model.predict_on_batch(tf_tracks)
    sum_LP = MLE_loss(LPs, LPs)
    est_LocErrs, est_ds, est_qs, est_ls = model.layers[1].get_parameters()
    
    return est_LocErrs, est_ds, est_qs, est_ls, sum_LP

def confined_LP(inputs, nb_dims = 2, Fixed_LocErr = True, Initial_params = {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01}):
    layer1 = Confinement_Initial_layer_multi(Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_params)
    tensor1, initial_state = layer1(inputs, nb_dims)
    cell = Confinement_RNNCell_multi([1, nb_dims, 1, 1, 1, nb_dims, nb_dims], layer1, nb_dims) # n
    RNN_layer = tf.keras.layers.RNN(cell, return_state = True)
    tensor2 = RNN_layer(tensor1[:,3:-1], initial_state = initial_state)
    prev_outputs = tensor2[1:]
    layer3 = Confinement_Final_layer_multi(layer1, nb_dims)
    LP = layer3(inputs[:,-1], prev_outputs)
    return LP

def directed_LP(inputs, track_len, nb_dims = 2, Fixed_LocErr = True, Initial_params = {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01}):
    if track_len > 4:
        layer1 = Directed_Initial_layer_multi(Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_params)
        tensor1, initial_state = layer1(inputs, nb_dims)
        cell = Directed_RNNCell_multi([1, 1, nb_dims, 1, 1, nb_dims, 1], layer1, nb_dims) # n
        RNN_layer = tf.keras.layers.RNN(cell, return_state = True)
        tensor2 = RNN_layer(tensor1[:,2:-2], initial_state = initial_state)
        prev_outputs = tensor2[1:]
        layer3 = Directed_Final_layer_multi(layer1, nb_dims)
        #LP, estimated_ds, estimated_qs, estimated_ls, estimated_LocErrs = layer3(inputs[:,:], prev_outputs)
        LP = layer3(inputs[:,:], prev_outputs)
    elif track_len <= 4:
        inputs = tf.keras.Input(shape=(None, nb_dims))
        layer1 = Directed_Initial_layer_multi(Fixed_LocErr = Fixed_LocErr)
        tensor1, initial_state = layer1(inputs, nb_dims)
        prev_outputs = initial_state
        layer3 = Directed_Final_layer_multi(layer1, nb_dims)
        LP = layer3(inputs[:,:], prev_outputs)
    return LP

def Brownian_LP(inputs, nb_dims = 2, Fixed_LocErr = True, Initial_params = {'LocErr': 0.02, 'd': 0.1}):
    layer1 = Brownian_Initial_layer_multi(Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_params)
    tensor1, initial_state = layer1(inputs)
    cell = Brownian_RNNCell_multi([nb_dims, 1, 1], layer1) # n
    RNN_layer = tf.keras.layers.RNN(cell)
    LP = RNN_layer(tensor1[:,1:], initial_state = initial_state)
    return LP

def multi_fit(tracks, verbose = 1, Fixed_LocErr = True, min_nb_states = 3, max_nb_states = 15, nb_epochs = 1000, batch_size = 2**11,
               Initial_confined_params = {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01},
               Initial_directed_params = {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01},
               ):
    '''
    Fit models with multiple states and vary the number of states to determine which number of states is best suited to the 
    data set and to retrieve the multi-state model parameters.
    Here, in a first fitting step, we estimate the parameters of individual tracks. We then cluster tracks with close parameters
    to form `max_nb_states` states whose parameters are the average of the parameters of their tracks.
    Then, multi-state fitting is performed on the full data set. the log likelihood is computed and stored and the
    state with the list impact on the likelihood is removed. The number of states is further reduced until the 
    number of states of the model reaches the value `min_nb_states`.
    
    Parameters
    ----------
    tracks : numpy array
         array of tracks of dims (track, time point, spatial axis).
    verbose : int, optional
        print model and fitting infos if 1. The default is 1.
    Fixed_LocErr : bool, optional
        If True fixes the the localization error based on a prior estimate, this can be important to do if there is no immobile state. The default is True.
    max_nb_states: int, optional
        maximum number of states
    min_nb_states : int, optional
        Initial and minimal number of states used for the clustering. If there is no decrease of the number of states between the clustering phase and the reduction phase 
    nb_epochs : TYPE, optional
        DESCRIPTION. The default is 1000.
    batch_size: int
        Number of tracks considered per batch to avoid memory issues when dealing with big data sets.
    Initial_confined_params : TYPE, optional
        Initial guess for the first step of the method. The default is {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01}.
    Initial_directed_params : TYPE, optional
        DESCRIPTION. The default is {'LocErr': 0.02, 'd': 0.1, 'q': 0.01, 'l': 0.01}.

    Returns
    -------
    TYPE
        

    '''
    
    all_pd_params = {}
    
    tracks = np.array(tracks)
    
    nb_tracks, track_len, nb_dims = tracks.shape
    
    est_LocErrs, est_ds, est_qs, est_ls, LP = Confined_fit(tracks, verbose = verbose, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_confined_params, nb_epochs = nb_epochs)
    est_LocErrs2, est_ds2, est_qs2, est_ls2, LP2, pred_kis = Directed_fit(tracks, verbose = verbose, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_directed_params, nb_epochs = nb_epochs)
    
    mask = LP < LP2
    #mask = mask
    
    LP[-20:]
    LP2[-20:]

    est_LocErrs[mask] = est_LocErrs2[mask]
    est_ds[mask] = est_ds2[mask]
    est_qs[mask] = est_qs2[mask]
    est_ls[mask] = est_ls2[mask]
    est_ls[mask==False] = - est_ls[mask==False]
    
    if Fixed_LocErr == True:
        params = np.concatenate((np.log(est_ds), np.log(est_qs), est_ls), -1)
        raw_params = np.concatenate((est_ds,est_qs, est_ls), -1)
    
    else:
        est_LocErrs[mask] = est_LocErrs2[mask]
        params = np.concatenate((np.log(est_ds), np.log(est_qs), est_ls, np.log(est_LocErrs)), -1)
        raw_params = np.concatenate((est_ds,est_qs, est_ls, est_LocErrs), -1) 
    
    params_std = np.std(params, 0)
    params_mean = np.mean(params, 0)
    norm_params = (params - params_mean)/params_std
    
    k = max_nb_states
    
    print("Number of estimated clusters : %d" % k)
    gm = GaussianMixture(n_components=k, random_state=0).fit(norm_params)
    labels = gm.predict(norm_params)
    labels_unique = np.unique(labels)
    
    denorm_centers = []
    for i in np.unique(labels):
        denorm_centers.append(np.median(raw_params[labels==i], 0))
    denorm_centers = np.array(denorm_centers)
    n_clusters_ = k
    
    print('reduction phase: phase where we test the clustered states and remove the states that do not significantly increase the likelihood. If no state is removed during this stage, increase the value of the parameter `initial_nb_states` (integer)')
    
    #Conf_params = denorm_centers[:2]
    
    Conf_params = denorm_centers[denorm_centers[:,2]<0]
    Conf_params[:,2] = - Conf_params[:,2]
    Dir_params = denorm_centers[denorm_centers[:,2]>=0]
    
    nb_conf_states = len(Conf_params)
    nb_dir_states = len(Dir_params)
    
    #max_nb_states = np.max([nb_conf_states, nb_dir_states])
    
    tracks_tf = tf.repeat(tf.constant(tracks[:,:,None, :nb_dims], dtype = dtype), nb_conf_states + nb_dir_states, 2)
    inputs = tf.keras.Input(shape=(None, nb_conf_states + nb_dir_states, nb_dims), dtype = dtype)
    
    outputs = []
    
    if nb_conf_states > 0:
        
        if Fixed_LocErr:
            Conf_params = np.concatenate((Conf_params, np.full((nb_conf_states, 1), Initial_confined_params['LocErr'])), 1)
        
        Initial_conf_params = {}
        for i, param in enumerate(['d', 'q', 'l', 'LocErr']):
            Initial_conf_params[param] = Conf_params[:, i]
        
        Conf_inputs = inputs[:,:,:nb_conf_states]
        Conf_layer1 = Confinement_Initial_layer_multi(nb_states = nb_conf_states, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_conf_params, dtype = dtype)
        Conf_tensor1, Conf_initial_state = Conf_layer1(Conf_inputs, nb_dims)
        Conf_cell = Confinement_RNNCell_multi([tf.TensorShape([nb_conf_states]), tf.TensorShape([nb_conf_states, nb_dims]), tf.TensorShape([nb_conf_states, 1]), tf.TensorShape([nb_conf_states, 1]), tf.TensorShape([nb_conf_states, 1]), tf.TensorShape([nb_conf_states, nb_dims]), tf.TensorShape([nb_conf_states, nb_dims])], Conf_layer1, nb_dims, dtype = dtype) # n
        Conf_RNN_layer = RNN(Conf_cell, return_state = True, dtype = dtype)
        Conf_tensor2 = Conf_RNN_layer(Conf_tensor1[:,3:-1], initial_state = Conf_initial_state)
        Conf_prev_outputs = Conf_tensor2[1:]
        Conf_layer3 = Confinement_Final_layer_multi(Conf_layer1, nb_dims, dtype = dtype)
        Conf_LP = Conf_layer3(Conf_inputs[:,-1], Conf_prev_outputs)
        outputs.append(Conf_LP)
    
    if nb_dir_states > 0:
        
        if Fixed_LocErr:
            Dir_params = np.concatenate((Dir_params, np.full((nb_dir_states, 1), Initial_directed_params['LocErr'])), 1)
        
        Initial_dir_params = {}
        for i, param in enumerate(['d', 'q', 'l', 'LocErr']):
            Initial_dir_params[param] = Dir_params[:, i]
        
        Dir_inputs = inputs[:,:, nb_conf_states:]
        Dir_layer1 = Directed_Initial_layer_multi(nb_states = 3, Fixed_LocErr = Fixed_LocErr, Initial_params = Initial_dir_params, dtype = dtype)
        Dir_tensor1, Dir_initial_state = Dir_layer1(Dir_inputs, nb_dims)
        Dir_cell = Directed_RNNCell_multi([tf.TensorShape([nb_dir_states]), tf.TensorShape([nb_dir_states, 1]), tf.TensorShape([nb_dir_states, nb_dims]), tf.TensorShape([nb_dir_states, 1]), tf.TensorShape([nb_dir_states, 1]), tf.TensorShape([nb_dir_states, nb_dims]), tf.TensorShape([nb_dir_states, 1])], Dir_layer1, nb_dims, dtype = dtype) # n
        Dir_RNN_layer = tf.keras.layers.RNN(Dir_cell, return_state = True, dtype = dtype)
        Dir_tensor2 = Dir_RNN_layer(Dir_tensor1[:,2:-2], initial_state = Dir_initial_state)
        Dir_prev_outputs = Dir_tensor2[1:]
        Dir_layer3 = Directed_Final_layer_multi(Dir_layer1, nb_dims, dtype = dtype)
        #LP, estimated_ds, estimated_qs, estimated_ls, estimated_LocErrs = layer3(inputs[:,:], prev_outputs)
        Dir_LP = Dir_layer3(Dir_inputs, Dir_prev_outputs)
        outputs.append(Dir_LP)
    
    state_list = np.array(['Conf']*nb_conf_states + ['Dir']*nb_dir_states)
    outputs = tf.concat(outputs, axis = 1)
    
    class Fractions_Layer(tf.keras.layers.Layer):
        def __init__(self, nb_states, **kwargs):
            #super(Fractions_Layer, self).__init__(**kwargs)
            self.nb_states = nb_states
            super().__init__(**kwargs)
        
        def build(self, input_shape):
            if not self.built:
                self.Fractions = tf.Variable(initial_value=np.ones(self.nb_states)[None]/self.nb_states, trainable=True, name="Fractions", dtype = dtype)
            self.built = True
            #super(Fractions_Layer, self).build(input_shape)  # Be sure to call this at the end
    
        def call(self, x):
            F_P =  tf.math.log(tf.math.softmax(self.Fractions))
            return x + F_P
    
    F_layer = Fractions_Layer(nb_conf_states + nb_dir_states, dtype = dtype)
    F_outputs = F_layer(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=F_outputs, name="Diffusion_model")
    
    if verbose > 0:
        model.summary()
    # model.compile(loss=MLE_loss, optimizer='adam')
    
    def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
        #print(y_pred)
        nb_states = y_pred.shape[1]
        max_LPs = tf.math.reduce_max(y_pred, 1, keepdims = True)
        #sum_LP_layers = tf.math.log(tf.math.reduce_sum(tf.cast(tf.math.exp(y_pred - max_LPs), dtype = 'float64'), axis=-1, keepdims = False)/nb_states) + tf.cast(max_LPs[:, 0], , dtype = 'float64')
        sum_LP_layers = tf.math.log(tf.math.reduce_sum(tf.math.exp(tf.cast(y_pred - max_LPs, dtype = dtype)), axis=-1, keepdims = False)) + tf.cast(max_LPs[:, 0], dtype = dtype)
        return - sum_LP_layers # sum over the spatial dimensions axis
    
    '''
    checkpoint_filepath = 'D:/anomalous/5_states/best_weights.h5'
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=True,
                                                                   monitor='loss',
                                                                   mode='auto',
                                                                   save_best_only=True)
    '''
    #print('fit:')
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.5, beta_1=0.9, beta_2=0.99, epsilon=1e-20)
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tracks_tf, tracks_tf, epochs = 50, batch_size = min(batch_size, nb_tracks), shuffle=False, verbose = verbose) # , callbacks=[model_checkpoint_callback]
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-20)
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tracks_tf, tracks_tf, epochs = 50, batch_size = min(batch_size, nb_tracks), shuffle=False, verbose = verbose) # , callbacks=[model_checkpoint_callback]
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-20)
    model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    history = model.fit(tracks_tf, tracks_tf, epochs = nb_epochs, batch_size = min(batch_size, nb_tracks), shuffle=False, verbose = verbose) # , callbacks=[model_checkpoint_callback]
    
    #history_number = 80 # 32 bit
    history_number = 5 # 64 bit
    
    np.mean(history.history['loss'][-history_number:])
    np.median(history.history['loss'][-history_number:])
    np.std(history.history['loss'][-history_number:])
    
    mask_history_args = np.arange(nb_conf_states + nb_dir_states)
    #adam = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.9, beta_2=0.99)
    #model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
    #history = model.fit(tracks_tf, tracks_tf, epochs = 100, batch_size = nb_tracks, shuffle=False, verbose = verbose, callbacks=[model_checkpoint_callback])
    
    #model.load_weights('D:/anomalous/5_states/best_weights.h5')
    #LPs = model.predict_on_batch(tracks_tf)
    #tf.math.reduce_mean(tf.cast(- MLE_loss(LPs, LPs), dtype = 'float64')).numpy()
    
    LPs = model.predict_on_batch(tracks_tf)
    #sum_LPs = - MLE_loss(LPs, LPs)
    
    #sum_LP = tf.math.reduce_sum(tf.cast(sum_LPs, dtype = 'float64')).numpy()
    nb_states = LPs.shape[1]
    
    likelihoods = np.zeros(k)
    likelihoods[nb_states-1] = np.median(history.history['loss'][-5:])

    Best_weights = model.get_weights()

    Init_layers = []
    for layer in model.layers:
        if 'confinement__initial_layer' in layer.name:
            Init_layers.append(layer)   
    for layer in model.layers:
        if 'directed__initial_layer' in layer.name:
            Init_layers.append(layer)
    
    if len(Init_layers) == 2:
        LocErrC, dC, qC, lC = Init_layers[0].get_parameters()
        LocErrD, dD, qD, lD = Init_layers[1].get_parameters()
        
        LocErr, d, q, l = [np.concatenate((LocErrC, LocErrD), axis = 0), np.concatenate((dC, dD), axis = 0), np.concatenate((qC, qD), axis = 0), np.concatenate((lC, lD), axis = 0)]
        
    else:
        LocErr, d, q, l = Init_layers[0].get_parameters()
    
    final_LocErr, final_d, final_q, final_l = LocErr[mask_history_args], d[mask_history_args], q[mask_history_args], l[mask_history_args]
    final_states = state_list[mask_history_args]
    final_l[final_states=='Dir'] = final_l[final_states=='Dir']*2**0.5
    final_fractions = tf.math.softmax(F_layer.weights).numpy()[:,0].T
    
    pd_params = pd.DataFrame(np.concatenate((np.arange(nb_states)[:,None], final_LocErr, final_d, final_q, final_l, final_states[:,None], final_fractions), axis = 1), columns = ['state', 'LocErr', 'd', 'q', 'l', 'state', 'fraction'])
    all_pd_params[str(nb_states)] = pd_params
    
    while nb_states > min_nb_states:
        print('current number of states: ', nb_states)
        last_nb_states = nb_states
        LPs = model.predict_on_batch(tracks_tf)
        
        All_alternative_LPs = []
        Fractions = tf.math.softmax(F_layer.weights).numpy()[0]

        for i in range(nb_states):
            #For each state compute the impact of removing it on the likelihood
            mask = np.full((nb_states), True, dtype=bool)
            mask[i] = False
            alternative_LPs = LPs[:, mask]
            
            # The fractions don't sum up to one anymore so we need to correct for this:
            alternative_LPs = alternative_LPs - np.log(np.sum(Fractions[:,mask]))  # - np.log(Fractions[:,mask]) + np.log(Fractions[:,mask] / np.sum(Fractions[:,mask]))
            #alt_LPs = - MLE_loss(alternative_LPs, alternative_LPs)
            
            All_alternative_LPs.append(tf.math.reduce_sum(- MLE_loss(alternative_LPs, alternative_LPs)).numpy())
        
        sorted_states = np.argsort(All_alternative_LPs)[::-1]
        i = sorted_states[0]
        
        mask = np.full((nb_states), True, dtype=bool)
        mask[i] = False
        mask_history_args = mask_history_args[mask]
        
        alternative_outputs = []
        for j in range(outputs.shape[1]):
            if mask[j] == True:
                alternative_outputs.append(outputs[:,j:j+1])
        
        # it would be simpler to use the F_outputs directly but the softmax function of the remaining weights would not sum to 1 anymore 
        alternative_outputs = tf.concat(alternative_outputs, axis = -1)
        F_layer = Fractions_Layer(nb_states-1, dtype = dtype)
        alternative_outputs_F = F_layer(alternative_outputs)
        
        alternative_model = tf.keras.Model(inputs=inputs, outputs=alternative_outputs_F, name="Diffusion_model")
        
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-20)
        alternative_model.compile(loss=MLE_loss, optimizer=adam, jit_compile = True)
        history = alternative_model.fit(tracks_tf, tracks_tf, epochs = nb_epochs, batch_size = min(batch_size, nb_tracks), shuffle=False, verbose = verbose) #, callbacks=[model_checkpoint_callback])
        #alternative_model.load_weights('D:/anomalous/5_states/best_weights.h5')
        
        #LPs = alternative_model.predict_on_batch(tracks_tf)
        #sum_LPs = - MLE_loss(LPs, LPs)
        #sum_LP = tf.math.reduce_sum(tf.cast(sum_LPs, dtype = 'float64')).numpy()
        sum_LP = np.median(history.history['loss'][-history_number:])
        
        outputs = alternative_outputs
        model = alternative_model
        Best_weights = model.get_weights()
        nb_states += -1
        likelihoods[nb_states-1] = sum_LP
        
        print('current number of states: ', nb_states, ', current likelihood: ', likelihoods)
        
        Init_layers = []
        for layer in model.layers:
            if 'confinement__initial_layer' in layer.name:
                Init_layers.append(layer)   
        for layer in model.layers:
            if 'directed__initial_layer' in layer.name:
                Init_layers.append(layer)
        
        if len(Init_layers) == 2:
            LocErrC, dC, qC, lC = Init_layers[0].get_parameters()
            LocErrD, dD, qD, lD = Init_layers[1].get_parameters()
            
            LocErr, d, q, l = [np.concatenate((LocErrC, LocErrD), axis = 0), np.concatenate((dC, dD), axis = 0), np.concatenate((qC, qD), axis = 0), np.concatenate((lC, lD), axis = 0)]
            
        else:
            LocErr, d, q, l = Init_layers[0].get_parameters()
        
        final_LocErr, final_d, final_q, final_l = LocErr[mask_history_args], d[mask_history_args], q[mask_history_args], l[mask_history_args]
        final_states = state_list[mask_history_args]
        final_l[final_states=='Dir'] =  final_l[final_states=='Dir']*2**0.5
        final_fractions = tf.math.softmax(F_layer.weights)
        
        pd_params = pd.DataFrame(np.concatenate((np.arange(nb_states)[:,None], final_LocErr, final_d, final_q, final_l, final_states[:,None], final_fractions), axis = 1), columns = ['state', 'LocErr', 'd', 'q', 'l', 'state', 'fraction'])
        all_pd_params[str(nb_states)] = pd_params
        
    return likelihoods, all_pd_params
























