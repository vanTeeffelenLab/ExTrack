import xmltodict
import numpy as np
import pandas as pd

def read_trackmate_xml(paths, # path (string specifying the path of the file or list of paths in case of multiple files.
                       lengths=np.arange(5,40), # track lengths kept.
                       dist_th = 0.5, # maximum distance between consecutive peaks of a track.
                       frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
                       remove_no_disp = True, # removes tracks showing no motion if True.
                       opt_metrics_names = ['t', 'x'], # e.g. ['pred_0', 'pred_1'],
                       opt_metrics_types = [int, 'float64'] # will assume 'float64' type if none, otherwise specify a list of same length as opt_metrics_names: e.g. ['float64','float64']
                       ):
    """
    Converts xml output from trackmate to a list of arrays of tracks
    each element of the list is an array composed of several tracks (dim 0), of a fix number of position (dim 1)
    with x and y coordinates (dim 2)
    path : path to xml file
    lengths : lengths used for the arrays, list of intergers, tracks with n localization will be added 
    to the array of length n if n is in the list, if n > max(lenghs)=k the k first steps of the track will be added.
    dist_th : maximum distance allowed to connect consecutive peaks
    start_frame : first frame considered 
    """
    if type(paths) == type(''):
        paths = [paths]
    traces = {}
    frames = {}
    opt_metrics = {}
    for m in opt_metrics_names:
        opt_metrics[m] = {}
    for l in lengths:
        traces[str(l)] = []
        frames[str(l)] = []
        for m in opt_metrics_names:
            opt_metrics[m][str(l)] = []
    
    if opt_metrics_types == None:
        opt_metrics_types = ['float64']*len(opt_metrics_names)
    
    for path in paths:
        data = xmltodict.parse(open(path, 'r').read(), encoding='utf-8')
        # Checks
        spaceunit = data['Tracks']['@spaceUnits']
        #if spaceunit not in ('micron', 'um', 'µm', 'Âµm'):
        #    raise IOError("Spatial unit not recognized: {}".format(spaceunit))
        #if data['Tracks']['@timeUnits'] != 'ms':
        #    raise IOError("Time unit not recognized")

        # parameters
        framerate = float(data['Tracks']['@frameInterval'])/1000. # framerate in ms
        
        for i, particle in enumerate(data['Tracks']['particle']):
            try:
                track = [(float(d['@x']), float(d['@y']), float(d['@t'])*framerate, int(d['@t']), i) for d in particle['detection']]
                opt_met = np.empty((int(particle['@nSpots']), len(opt_metrics_names)), dtype = 'object')
                for k, d in enumerate(particle['detection']):
                    for j, m in enumerate(opt_metrics_names):
                        opt_met[k, j] = d['@'+opt_metrics_names[j]]

                track = np.array(track)
                if remove_no_disp:
                    no_zero_disp = np.min((track[1:,0] - track[:-1,0])**2) * np.min((track[1:,1] - track[:-1,1])**2)
                else:
                    no_zero_disp = True

                dists = np.sum((track[1:, :2] - track[:-1, :2])**2, axis = 1)**0.5
                if no_zero_disp and track[0, 3] >= frames_boundaries[0] and track[0, 3] <= frames_boundaries[1] and np.all(dists<dist_th):
                    l = len(track)
                    if np.any([l]*len(lengths) == np.array(lengths)) :
                        traces[str(l)].append(track[:, 0:2])
                        frames[str(l)].append(track[:, 3])
                        for k, m in enumerate(opt_metrics_names):
                            opt_metrics[m][str(l)].append(opt_met[:, k])
                    elif l > np.max(lengths):
                        l = np.max(lengths)
                        traces[str(l)].append(track[:l, 0:2])
                        frames[str(l)].append(track[:l, 3])
                        for k, m in enumerate(opt_metrics_names):
                            opt_metrics[m][str(l)].append(opt_met[:l, k])
            except :
                raise ValueError('problem with data on path: ' + path)

    for l in list(traces.keys()):
        if len(traces[l])>0:
            traces[l] = np.array(traces[l])
            frames[l] = np.array(frames[l])
            for k, m in enumerate(opt_metrics_names):
                cur_opt_met = np.array(opt_metrics[m][l])
                try:
                    cur_opt_met = cur_opt_met.astype(opt_metrics_types[k])
                except :
                    print('Error of type with the optional metric:', m)
                opt_metrics[m][l] = cur_opt_met
        else:
            del traces[l], frames[l]
            for k, m in enumerate(opt_metrics_names):
                del opt_metrics[m][l]
                
    return traces, frames, opt_metrics


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
    
    nb_dims = len(colnames)-2
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
        
        if not pd.api.types.is_numeric_dtype(data.dtypes[colnames[0]]):
            raise ValueError('The x values are not numerical. Verify the presence of non numerical values in the file. In particular, verify that the file contains only one row of headers')            
        
        if not pd.api.types.is_numeric_dtype(data.dtypes[colnames[1]]):
            raise ValueError('The y values are not numerical. Verify the presence of non numerical values in the file. In particular, verify that the file contains only one row of headers')            
        
        if not pd.api.types.is_numeric_dtype(data.dtypes[colnames[2]]):
            raise ValueError('The frame values are not numerical. Verify the presence of non numerical values in the file. In particular, verify that the file contains only one row of headers')    
    
        if not (type(colnames[-1]) == str or type(colnames[-1]) == np.str_):
            # in this case we remove the NA values for simplicity
            None_ID = (data[colnames[-1]] == 'None') + pd.isna(data[colnames[-1]])
            data = data.drop(data[np.any(None_ID,1)].index)
            
            new_ID = data[colnames[-1][0]].astype(str)
            
            for k in range(1,len(colnames[-1])):
                new_ID = new_ID + '_' + data[colnames[-1][k]].astype(str)
            data['unique_ID'] = new_ID
            colnames[-1] = 'unique_ID'        
        try:
            # in this case, peaks without an ID are assumed alone and are added a unique ID, only works if ID are integers
            None_ID = (data[colnames[-1]] == 'None') + pd.isna(data[colnames[3]])
            max_ID = np.max(data[colnames[-1]][(data[colnames[-1]] != 'None' ) * (pd.isna(data[colnames[-1]]) == False)].astype(int))
            data.loc[None_ID, colnames[-1]] = np.arange(max_ID+1, max_ID+1 + np.sum(None_ID))
        except:
            None_ID = (data[colnames[-1]] == 'None') + pd.isna(data[colnames[-1]])
            data = data.drop(data[None_ID].index)
        
        # To enable to retrieve the track ID, we modify the track ID name
        
        if colnames[-1] in opt_colnames:
            new_col = pd.DataFrame(data[colnames[-1]].values, columns = [colnames[-1] + '_bis'])
            data = pd.concat((data, new_col), axis = 1)
            colnames[-1] = colnames[-1] + '_bis'
        
        data = data[colnames + opt_colnames]
        zero_disp_tracks = 0
            
        try:
            for ID, track in data.groupby(colnames[-1]):
                
                track = track.sort_values(colnames[-2], axis = 0)
                track_mat = track.values[:,:nb_dims+1].astype('float64')
                dists2 = (track_mat[1:, :nb_dims] - track_mat[:-1, :nb_dims])**2
                if remove_no_disp:
                    if np.mean(dists2==0)>0.05:
                        continue
                dists = np.sum(dists2, axis = 1)**0.5
                if track_mat[0, nb_dims] >= frames_boundaries[0] and track_mat[0, nb_dims] <= frames_boundaries[1] : #and np.all(dists<dist_th):
                    if not np.any(dists>dist_th):
                        
                        if np.any([len(track_mat)]*len(lengths) == np.array(lengths)):
                            l = len(track)
                            tracks[str(l)].append(track_mat[:, :nb_dims])
                            frames[str(l)].append(track_mat[:, nb_dims])
                            for m in opt_colnames:
                                opt_metrics[m][str(l)].append(track[m].values)
                        
                        elif len(track_mat) > np.max(lengths):
                            l = np.max(lengths)
                            tracks[str(l)].append(track_mat[:l, :nb_dims])
                            frames[str(l)].append(track_mat[:l, nb_dims])
                            for m in opt_colnames:
                                opt_metrics[m][str(l)].append(track[m].values[:l]) 
                        
                        elif len(track_mat) < np.max(lengths) and len(track_mat) > np.min(lengths) : # in case where lengths between min(lengths) and max(lentghs) are not all present:
                            l_idx =   np.argmin(np.floor(len(track_mat) / lengths))-1
                            l = lengths[l_idx]
                            tracks[str(l)].append(track_mat[:l, :nb_dims])
                            frames[str(l)].append(track_mat[:l, nb_dims])
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

