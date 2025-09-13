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
    
    for ID in np.unique(DATA['TRACK_ID'])[::-1]:
        if np.mod(ID, 20)==0:
            print('.', end = '')
        #print(ID)
        track = DATA[DATA['TRACK_ID'] ==ID ]
        if track_length_range[0] < len(track) > track_length_range[0]:
            if nb_states == 2 :
                pred = track['pred_1']
                pred = cm.brg(pred*0.5)
            else:
                pred = track[['pred_2', 'pred_1', 'pred_0']].values
            
            plt.plot(track['POSITION_X'], track['POSITION_Y'], 'k:', alpha = 0.2)
            plt.scatter(track['POSITION_X'], track['POSITION_Y'], c = pred, s=3)
            plt.gca().set_aspect('equal', adjustable='datalim')
            #plt.scatter(track['X'], track['X'], marker = 'x', c='k', s=5, alpha = 0.5)

def plot_tracks(DATA,
                max_track_length = 50,
                nb_subplots = [5,5],
                figsize = (10,10),
                lim = 0.4 ):
    '''
    DATA: dataframe outputed by extrack.exporters.extrack_2_pandas.
    max_track_length: maximum track length to be outputed, it will plot the longest tracks respecting this criteria.
    nb_subplots: number of lines and columns of subplots.
    figsize: size of the figure plotted
    lim: limit for x and y axis around the track center
    '''
    
    # Count number of states
    nb_states = 0
    pred_columns = []
    for param in list(DATA.keys()):
        if param.find('pred')+1:
            nb_states += 1
            pred_columns.append(param)
    
    # Sort prediction columns to ensure consistent ordering
    pred_columns.sort()
    
    plt.figure(figsize=figsize)
    
    # Remove tracks longer than max_track_length
    for ID in np.unique(DATA['TRACK_ID'])[::-1]:
        track = DATA[DATA['TRACK_ID'] == ID]
        if len(track) > max_track_length:
            DATA.drop((DATA[DATA['TRACK_ID'] == ID]).index, inplace=True)
    
    # Generate distinct colors for each state
    if nb_states > 0:
        # Use a colormap that provides distinct colors
        if nb_states <= 10:
            colormap = cm.tab10
        elif nb_states <= 20:
            colormap = cm.tab20
        else:
            colormap = cm.hsv
        
        # Generate colors for each state
        state_colors = [colormap(i) for i in range(nb_states)]
    
    # Plot tracks
    for k, ID in enumerate(np.unique(DATA['TRACK_ID'])[::-1][:min(len(np.unique(DATA['TRACK_ID'])), np.prod(nb_subplots))]):
        
        plt.subplot(nb_subplots[0], nb_subplots[1], k+1)
        track = DATA[DATA['TRACK_ID'] == ID]

        if nb_states == 1:
            # Single state, use intensity based on prediction
            pred = track[pred_columns[0]].values
            pred = cm.viridis(pred)
        elif nb_states == 2:
            # Two states, interpolate between two colors
            pred = track[pred_columns[1]].values  # Usually pred_1 for binary classification
            pred = cm.brg(pred*0.5)
        elif nb_states == 3:
            pred = track[pred_columns].values  # Usually pred_1 for binary classification

        else:
            # Multiple states, use RGB mixing or dominant state coloring
            pred_values = track[pred_columns].values
            
            # Method 1: Color by dominant state
            dominant_states = np.argmax(pred_values, axis=1)
            pred = [state_colors[state] for state in dominant_states]
            
            # Alternative Method 2: RGB mixing (uncomment to use instead)
            # if nb_states == 3:
            #     pred = pred_values  # Direct RGB for 3 states
            # else:
            #     # For more than 3 states, use weighted average of first 3 colors
            #     weights = pred_values[:, :min(3, nb_states)]
            #     if nb_states > 3:
            #         weights = weights / np.sum(weights, axis=1, keepdims=True)
            #     pred = weights
        
        # Plot track
        plt.plot(track['POSITION_X'], track['POSITION_Y'], 'k:', alpha = 0.2)
        plt.scatter(track['POSITION_X'], track['POSITION_Y'], c = pred, s=3)
        plt.xlim([np.mean(track['POSITION_X']) - lim, np.mean(track['POSITION_X']) + lim])
        plt.ylim([np.mean(track['POSITION_Y']) - lim, np.mean(track['POSITION_Y']) + lim])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.yticks(fontsize = 6)
        plt.xticks(fontsize = 6)
    
    # Add legend for states
    legend_elements = []
    for i in range(nb_states):
        label = f'State {i}'
        if nb_states == 1:
            color = cm.viridis(1)  # Middle color for representation
        elif nb_states == 2:
            if i == 0:
                color = cm.brg(0.0)
            else:
                color = cm.brg(0.5)
        elif nb_states == 3:
            color = [0,0,0]
            color[i] = 1
        else:
            label = f'State {i}'
            color = state_colors[i]
        
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=5, 
                                            label=label, linestyle='None'))
    
    # Place legend outside the plot area
    plt.figlegend(handles=legend_elements, 
                 loc='center right', 
                 bbox_to_anchor=(0.98, 0.5),
                 fontsize=8)
    
    print('')
    plt.tight_layout(h_pad = 1, w_pad = 1)
    # Adjust layout to make room for legend
    if nb_states > 0:
        plt.subplots_adjust(right=0.85)
