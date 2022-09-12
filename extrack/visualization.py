import numpy as np
from matplotlib import pyplot as plt
from extrack.histograms import len_hist
from matplotlib import cm

def visualize_states_durations(all_tracks,
                               params,
                               dt,
                               cell_dims = [1,None,None],
                               nb_states = 2,
                               max_nb_states = 500,
                               workers = 1,
                               long_tracks = True,
                               nb_steps_lim = 20,
                               steps = False,
                               input_LocErr = None):
    '''
    arguments:
    all_tracks: dict describing the tracks with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = x, y position.
    params: lmfit parameters used for the model.
    dt: time in between frames.
    cell_dims: dimension limits (um). estimated_vals, min_values, max_values should be changed accordingly to describe all states and transitions.
    max_nb_states: maximum number of sequences kept (most likely sequences).
    nb_steps_lim: upper limit of the plot in the x axis (number of steps)
    long_tracks: if True only selects tracks longer than nb_steps_lim
    steps: x axis in seconds if False or in number of steps if False.

    outputs:
    plot of all tracks (preferencially input a single movie)
    '''
    len_hists = len_hist(all_tracks,
                         params,
                         dt,
                         cell_dims=cell_dims,
                         nb_states=nb_states,
                         workers = workers,
                         nb_substeps=1,
                         max_nb_states = max_nb_states,
                         input_LocErr = input_LocErr)
    
    if steps:
        step_type = 'step'
        dt = 1
    else:
        step_type = 's'
    
    plt.figure(figsize = (3,3))
    for k, hist in enumerate(len_hists.T):
        plt.plot(np.arange(1,len(hist)+1)*dt, hist/np.sum(hist), label='state %s'%k)
    
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.xlim([0,nb_steps_lim*dt])
    plt.ylim([0.001,0.5])
    plt.xlabel('state duration (%s)'%(step_type))
    plt.ylabel('fraction')
    plt.tight_layout()
    return len_hists

def visualize_tracks(DATA,
                     track_length_range = [10,np.inf],
                     figsize = (5,5)):
    '''
    DATA: dataframe outputed by extrack.exporters.extrack_2_pandas
    track_length_range: range of tracks ploted. plotting too many tracks may make it crash
    figsize: size of the figure plotted
    '''
    nb_states = 0
    for param in list(DATA.keys()):
        if param.find('pred')+1:
            nb_states += 1
    
    plt.figure(figsize = figsize)
    DATA['X']
    for ID in np.unique(DATA['track_ID'])[::-1]:
        if np.mod(ID, 20)==0:
            print('.', end = '')
        #print(ID)
        track = DATA[DATA['track_ID'] ==ID ]
        if track_length_range[0] < len(track) > track_length_range[0]:
            if nb_states == 2 :
                pred = track['pred_1']
                pred = cm.brg(pred*0.5)
            else:
                pred = track[['pred_2', 'pred_1', 'pred_0']].values
            
            plt.plot(track['X'], track['Y'], 'k:', alpha = 0.2)
            plt.scatter(track['X'], track['Y'], c = pred, s=3)
            plt.gca().set_aspect('equal', adjustable='datalim')
            #plt.scatter(track['X'], track['X'], marker = 'x', c='k', s=5, alpha = 0.5)

def plot_tracks(DATA,
                max_track_length = 50,
                nb_subplots = [5,5],
                figsize = (10,10),
                lim = 0.4 ):
    ''''
    DATA: dataframe outputed by extrack.exporters.extrack_2_pandas.
    max_track_length: maximum track length to be outputed, it will plot the longest tracks respecting this criteria.
    nb_subplots: number of lines and columns of subplots.
    figsize: size of the figure plotted
    '''
    nb_states = 0
    for param in list(DATA.keys()):
        if param.find('pred')+1:
            nb_states += 1
    
    plt.figure(figsize=figsize)
    
    for ID in np.unique(DATA['track_ID'])[::-1]:
        track = DATA[DATA['track_ID'] ==ID]
        if len(track) > max_track_length:
            DATA.drop((DATA[DATA['track_ID'] == ID]).index, inplace=True)
    
    for k, ID in enumerate(np.unique(DATA['track_ID'])[::-1][:np.product(nb_subplots)]):
        plt.subplot(nb_subplots[0], nb_subplots[1], k+1)

        track = DATA[DATA['track_ID'] ==ID ]
        if nb_states == 2 :
            pred = track['pred_1']
            pred = cm.brg(pred*0.5)
        else:
            pred = track[['pred_2', 'pred_1', 'pred_0']].values
        
        plt.plot(track['X'], track['Y'], 'k:', alpha = 0.2)
        plt.scatter(track['X'], track['Y'], c = pred, s=3)
        plt.xlim([np.mean(track['X']) - lim, np.mean(track['X']) + lim])
        plt.ylim([np.mean(track['Y']) - lim, np.mean(track['Y']) + lim])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.yticks(fontsize = 6)
        plt.xticks(fontsize = 6)
    print('')
    plt.tight_layout(h_pad = 1, w_pad = 1)

