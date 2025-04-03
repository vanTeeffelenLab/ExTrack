#def dict_pred_to_df_pred(all_Cs, all_Bs):
import numpy as np
import pickle
import json
import pandas as pd

def save_params(params, path = '.', fmt = 'json', file_name = 'params'):
    save_params = {}
    [save_params.update({param : params[param].value}) for param in params]
    '''
    available formats : json, npy, csv
    '''
    if fmt == 'npy':
        np.save(path + '/' + file_name, save_params)
    elif fmt == 'pkl':
        with open(path + '/' + file_name + ".pkl", "wb") as tf:
            pickle.dump(save_params,tf)
    elif fmt == 'json':
        with open(path + '/' + file_name + ".json", "w") as tf:
            json.dump(save_params,tf)
    elif fmt == 'csv':
        with open(path + '/' + file_name + ".csv", 'w') as tf:
            for key in save_params.keys():
                tf.write("%s,%s\n"%(key,save_params[key]))
    else :
        raise ValueError("format not supported, use one of : 'json', 'pkl', 'npy', 'csv'")

def extrack_2_matrix(all_Css, pred_Bss, dt, all_frames = None):
    row_ID = 0
    nb_pos = 0
    for len_ID in all_Css:
       all_Cs = all_Css[len_ID]
       nb_pos += all_Cs.shape[0]*all_Cs.shape[1]
    
    matrix = np.empty((nb_pos, 4+pred_Bss[list(pred_Bss.keys())[0]].shape[2]))
    #TRACK_ID,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,PRED_0, PRED_1,(PRED_2 etc)
    track_ID = 0
    for len_ID in all_Css:
        all_Cs = all_Css[len_ID]
        pred_Bs = pred_Bss[len_ID]
        if all_frames != None:
            all_frame = all_frames[len_ID]
        else:
            all_frame = np.arange(all_Cs.shape[0]*all_Cs.shape[1]).reshape((all_Cs.shape[0],all_Cs.shape[1])) 
        for track, preds, frames in zip(all_Cs, pred_Bs, all_frame):
            track_IDs = np.full(len(track),track_ID)[:,None]
            frames = frames[:,None]
            cur_track = np.concatenate((track, track_IDs,frames,preds ),1)
            
            matrix[row_ID:row_ID+cur_track.shape[0]] = cur_track
            row_ID += cur_track.shape[0]
            track_ID+=1
    return matrix

# all_tracks = tracks
# pred_Bs = preds

def extrack_2_pandas(all_tracks, pred_Bs, frames = None, opt_metrics = {}):
    '''
    turn outputs form ExTrack to a unique pandas DataFrame
    '''
    if frames is None:
        frames = {}
        for l in all_tracks:
            frames[l] = np.repeat(np.array([np.arange(int(l))]), len(all_tracks[l]), axis = 0)
    
    track_list = []
    frames_list = []
    track_ID_list = []
    opt_metrics_list = []
    for metric in opt_metrics:
        opt_metrics_list.append([])
    
    cur_nb_track = 0
    pred_Bs_list = []
    for l in all_tracks:
        track_list = track_list + list(all_tracks[l].reshape(all_tracks[l].shape[0] * all_tracks[l].shape[1], 2))
        frames_list = frames_list + list(frames[l].reshape(frames[l].shape[0] * frames[l].shape[1], 1))
        track_ID_list = track_ID_list + list(np.repeat(np.arange(cur_nb_track,cur_nb_track+all_tracks[l].shape[0]),all_tracks[l].shape[1]))
        cur_nb_track += all_tracks[l].shape[0]
        
        for j, metric in enumerate(opt_metrics):
            opt_metrics_list[j] = opt_metrics_list[j] + list(opt_metrics[metric][l].reshape(opt_metrics[metric][l].shape[0] * opt_metrics[metric][l].shape[1], 1))
        
        n = pred_Bs[l].shape[2]
        pred_Bs_list = pred_Bs_list + list(pred_Bs[l].reshape(pred_Bs[l].shape[0] * pred_Bs[l].shape[1], n))
    
    all_data = np.concatenate((np.array(track_list), np.array(frames_list), np.array(track_ID_list)[:,None], np.array(pred_Bs_list)), axis = 1)    
    for opt_metric in opt_metrics_list:
        all_data = np.concatenate((all_data, opt_metric), axis = 1)
    
    nb_dims = len(track_list[0])
    colnames = ['POSITION_X', 'POSITION_Y', 'POSITION_Z'][:nb_dims] + ['FRAME', 'TRACK_ID']
    for i in range(np.array(pred_Bs_list).shape[1]):
        colnames = colnames + ['pred_' + str(i)]
    for metric in opt_metrics:
        colnames = colnames + [metric]
    
    df = pd.DataFrame(data = all_data, index = np.arange(len(all_data)), columns = colnames)
    df['FRAME'] = df['FRAME'].astype(int)
    df['TRACK_ID'] = df['TRACK_ID'].astype(int)
    return df


def extrack_2_pandas2(tracks, pred_Bs, frames = None, opt_metrics = {}):
    '''
    turn outputs form ExTrack to a unique pandas DataFrame
    '''
    if frames is None:
        frames = {}
        for l in tracks:
            frames[l] = np.repeat(np.array([np.arange(int(l))]), len(tracks[l]), axis = 0)
    
    n = 0
    for l in tracks:
        n+= tracks[l].shape[0]*tracks[l].shape[1]
    
    nb_dims = tracks[l].shape[2]
    
    nb_states = pred_Bs[list(pred_Bs.keys())[0]].shape[-1]
    
    flat_tracks = np.zeros((n, tracks[l].shape[2]))
    flat_frames = np.zeros((n, 1))
    flat_Track_IDs = np.zeros((n, 1))
    flat_opt_metrics = np.zeros((n, len(opt_metrics.keys())))
    flat_preds = np.zeros((n, nb_states))
    
    track_ID = 0
    k = 0
    for l in tracks:
        for i, (track, f, p) in enumerate(zip(tracks[l], frames[l], pred_Bs[l])):
            track_length = track.shape[0]
            flat_tracks[k:k+track_length] = track
            flat_frames[k:k+track_length] = f[:, None]
            flat_Track_IDs[k:k+track_length] = track_ID
            flat_preds[k:k+track_length] = p
            for j, metric in enumerate(opt_metrics):
                flat_opt_metrics[k:k+track_length, j] = opt_metrics[metric][l][i]
            k+=track_length
            track_ID+=1
    
    arr = np.concatenate((flat_tracks, flat_frames, flat_Track_IDs, flat_opt_metrics, flat_preds), axis = 1)
    columns = ['POSITION_X', 'POSITION_Y', 'POSITION_Z'][:nb_dims] + ['FRAME', 'TRACK_ID'] + list(opt_metrics.keys())
    for i in range(nb_states):
        columns = columns + ['pred_' + str(i)]
    
    df = pd.DataFrame(data = arr, columns = columns)
    df['FRAME'] = df['FRAME'].astype(int)
    df['TRACK_ID'] = df['TRACK_ID'].astype(int)
    return df

def save_extrack_2_CSV(path, all_tracks, pred_Bss, dt, all_frames = None):
    track_ID = 0
    
    preds_header_str_fmt = ''
    preds_str_fmt = ''
    for k in range(pred_Bss[list(pred_Bss.keys())[0]].shape[2]):
        preds_header_str_fmt = preds_header_str_fmt + 'PRED_%s,'%(k)
        preds_str_fmt = preds_str_fmt + ',%s'
    
    with open(path, 'w') as f:
        f.write('TRACK_ID,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,%s\n'%(preds_header_str_fmt))
    
        for len_ID in all_tracks:
            nb_dims = all_tracks[len_ID].shape[2]
            tracks = np.zeros((all_tracks[len_ID].shape[0], all_tracks[len_ID].shape[1], 3)) # create 3D tracks with values 0 in unknown dims
            tracks[:,:, :nb_dims] = all_tracks[len_ID]
            pred_Bs = pred_Bss[len_ID]
            if all_frames != None:
                all_frame = all_frames[len_ID]
            else:
                all_frame = np.arange(tracks.shape[0]*tracks.shape[1]).reshape((tracks.shape[0],tracks.shape[1])) 
            for track, preds, frames in zip(tracks, pred_Bs, all_frame):
                track_ID+=1
                for pos, p, frame in zip(track, preds, frames):
                    preds_str = preds_str_fmt%(tuple(p))
                    f.write('%s,%s,%s,%s,%s,%s%s\n'%(track_ID, pos[0], pos[1], pos[2], dt* frame*1000, frame, preds_str))

def save_extrack_2_xml(all_tracks, pred_Bss, params, path, dt, all_frames = None, opt_metrics = {}):
    track_ID = 0
    for len_ID in all_tracks:
       tracks = all_tracks[len_ID]
       track_ID += len(tracks)
    nb_dims = all_tracks[len_ID].shape[2]
    
    final_params = []
    for param in params:
        if not '_' in param:
            final_params.append(param)
    Extrack_headers = 'ExTrack_results="'
    
    for param in final_params:
        Extrack_headers = Extrack_headers + param + "='" + str(np.round(params[param].value, 8)) +"' " 
    Extrack_headers += '"'
    
    preds_str_fmt = ''
    for k in range(pred_Bss[list(pred_Bss.keys())[0]].shape[2]):
        preds_str_fmt = preds_str_fmt + ' pred_%s="%s"'%(k,'%s')
    
    opt_metrics_fmt = ''
    
    for m in opt_metrics:
        opt_metrics_fmt = opt_metrics_fmt + '%s="%s" '%(m,'%s')
    
    with open(path, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<Tracks nTracks="%s" spaceUnits="µm" frameInterval="%s" timeUnits="ms" %s>\n'%(track_ID, dt, Extrack_headers))
        
        for len_ID in all_tracks:
            tracks = np.zeros((all_tracks[len_ID].shape[0], all_tracks[len_ID].shape[1], 3))
            tracks[:,:, :nb_dims] = all_tracks[len_ID]
            pred_Bs = pred_Bss[len_ID]
            opt_met = np.empty((tracks.shape[0],tracks.shape[1],len(opt_metrics)))
            for i, m in enumerate(opt_metrics):
                opt_met[:,:,i] = opt_metrics[m][len_ID]
            opt_met.shape
            if all_frames != None:
                all_frame = all_frames[len_ID]
            else:
                all_frame = np.arange(tracks.shape[0]*tracks.shape[1]).reshape((tracks.shape[0],tracks.shape[1])) 
            for i, (track, preds, frames) in enumerate(zip(tracks, pred_Bs, all_frame)):
                track_opt_met = opt_met[i]
                f.write('  <particle nSpots="%s">\n'%(len_ID))
                for pos, p, frame, track_opt_met in zip(track, preds, frames, track_opt_met):
                    preds_str = preds_str_fmt%(tuple(p))
                    opt_metrics_str = opt_metrics_fmt%(tuple(track_opt_met))
                    f.write('    <detection t="%s" x="%s" y="%s" z="%s"%s %s/>\n'%(frame,pos[0],pos[1],pos[2], preds_str, opt_metrics_str))
                f.write('  </particle>\n')
        f.write('</Tracks>\n')


def save_extrack_2_input_xml(all_tracks, pred_Bss, params, path, dt, all_frames = None, opt_metrics = {}):
    '''
    xml format for vizualization with TrackMate using the plugin "Load a TrackMate file"
    '''
    track_ID = 0
    for len_ID in all_tracks:
       tracks = all_tracks[len_ID]
       track_ID += len(tracks)
    nb_dims = all_tracks[len_ID].shape[2]
    
    final_params = []
    for param in params:
        if not '_' in param:
            final_params.append(param)
    Extrack_headers = 'ExTrack_results="'
    
    for param in final_params:
        Extrack_headers = Extrack_headers + param + "='" + str(np.round(params[param].value, 8)) +"' " 
    Extrack_headers += '"'
    
    preds_str_fmt = ''
    for k in range(pred_Bss[list(pred_Bss.keys())[0]].shape[2]):
        preds_str_fmt = preds_str_fmt + ' pred_%s="%s"'%(k,'%s')
    
    opt_metrics_fmt = ''
    
    for m in opt_metrics:
        opt_metrics_fmt = opt_metrics_fmt + '%s="%s" '%(m,'%s')
    
    with open(path, 'w', encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<TrackMate version="7.7.2">\n  <Model spatialunits="µm" timeunits="s">\n    <FeatureDeclarations>\n      <SpotFeatures>\n')
        f.write('        <Feature feature="QUALITY" name="Quality" shortname="Quality" dimension="QUALITY" isint="false" />\n        <Feature feature="POSITION_X" name="X" shortname="X" dimension="POSITION" isint="false" />\n        <Feature feature="POSITION_Y" name="Y" shortname="Y" dimension="POSITION" isint="false" />\n        <Feature feature="POSITION_Z" name="Z" shortname="Z" dimension="POSITION" isint="false" />\n        <Feature feature="POSITION_T" name="T" shortname="T" dimension="TIME" isint="false" />\n        <Feature feature="FRAME" name="Frame" shortname="Frame" dimension="NONE" isint="true" />\n        <Feature feature="RADIUS" name="Radius" shortname="R" dimension="LENGTH" isint="false" />\n        <Feature feature="VISIBILITY" name="Visibility" shortname="Visibility" dimension="NONE" isint="true" />\n        <Feature feature="MANUAL_SPOT_COLOR" name="Manual spot color" shortname="Spot color" dimension="NONE" isint="true" />\n        <Feature feature="MEAN_INTENSITY_CH1" name="Mean intensity ch1" shortname="Mean ch1" dimension="INTENSITY" isint="false" />\n        <Feature feature="MEDIAN_INTENSITY_CH1" name="Median intensity ch1" shortname="Median ch1" dimension="INTENSITY" isint="false" />\n        <Feature feature="MIN_INTENSITY_CH1" name="Min intensity ch1" shortname="Min ch1" dimension="INTENSITY" isint="false" />\n        <Feature feature="MAX_INTENSITY_CH1" name="Max intensity ch1" shortname="Max ch1" dimension="INTENSITY" isint="false" />\n        <Feature feature="TOTAL_INTENSITY_CH1" name="Sum intensity ch1" shortname="Sum ch1" dimension="INTENSITY" isint="false" />\n        <Feature feature="STD_INTENSITY_CH1" name="Std intensity ch1" shortname="Std ch1" dimension="INTENSITY" isint="false" />\n        <Feature feature="EXTRACK_P_STUCK" name="Probability stuck" shortname="P stuck" dimension="NONE" isint="false" />\n        <Feature feature="EXTRACK_P_DIFFUSIVE" name="Probability diffusive" shortname="P diffusive" dimension="NONE" isint="false" />\n        <Feature feature="CONTRAST_CH1" name="Contrast ch1" shortname="Ctrst ch1" dimension="NONE" isint="false" />\n        <Feature feature="SNR_CH1" name="Signal/Noise ratio ch1" shortname="SNR ch1" dimension="NONE" isint="false" />\n      </SpotFeatures>\n      <EdgeFeatures>\n        <Feature feature="SPOT_SOURCE_ID" name="Source spot ID" shortname="Source ID" dimension="NONE" isint="true" />\n        <Feature feature="SPOT_TARGET_ID" name="Target spot ID" shortname="Target ID" dimension="NONE" isint="true" />\n        <Feature feature="LINK_COST" name="Edge cost" shortname="Cost" dimension="COST" isint="false" />\n        <Feature feature="EDGE_TIME" name="Edge time" shortname="Edge T" dimension="TIME" isint="false" />\n        <Feature feature="EDGE_X_LOCATION" name="Edge X" shortname="Edge X" dimension="POSITION" isint="false" />\n        <Feature feature="EDGE_Y_LOCATION" name="Edge Y" shortname="Edge Y" dimension="POSITION" isint="false" />\n        <Feature feature="EDGE_Z_LOCATION" name="Edge Z" shortname="Edge Z" dimension="POSITION" isint="false" />\n        <Feature feature="VELOCITY" name="Velocity" shortname="V" dimension="VELOCITY" isint="false" />\n        <Feature feature="DISPLACEMENT" name="Displacement" shortname="Disp." dimension="LENGTH" isint="false" />\n        <Feature feature="MANUAL_COLOR" name="Manual edge color" shortname="Edge color" dimension="NONE" isint="true" />\n        <Feature feature="DIRECTIONAL_CHANGE_RATE" name="Directional change rate" shortname="γ rate" dimension="ANGLE_RATE" isint="false" />\n        <Feature feature="SPEED" name="Speed" shortname="Speed" dimension="VELOCITY" isint="false" />\n        <Feature feature="MANUAL_EDGE_COLOR" name="Manual edge color" shortname="Edge color" dimension="NONE" isint="true" />\n      </EdgeFeatures>\n      <TrackFeatures>\n        <Feature feature="TRACK_INDEX" name="Track index" shortname="Index" dimension="NONE" isint="true" />\n        <Feature feature="TRACK_ID" name="Track ID" shortname="ID" dimension="NONE" isint="true" />\n        <Feature feature="NUMBER_SPOTS" name="Number of spots in track" shortname="N spots" dimension="NONE" isint="true" />\n        <Feature feature="NUMBER_GAPS" name="Number of gaps" shortname="N gaps" dimension="NONE" isint="true" />\n        <Feature feature="LONGEST_GAP" name="Longest gap" shortname="Lgst gap" dimension="NONE" isint="true" />\n        <Feature feature="NUMBER_SPLITS" name="Number of split events" shortname="N splits" dimension="NONE" isint="true" />\n        <Feature feature="NUMBER_MERGES" name="Number of merge events" shortname="N merges" dimension="NONE" isint="true" />\n        <Feature feature="NUMBER_COMPLEX" name="Number of complex points" shortname="N complex" dimension="NONE" isint="true" />\n        <Feature feature="TRACK_DURATION" name="Track duration" shortname="Duration" dimension="TIME" isint="false" />\n        <Feature feature="TRACK_START" name="Track start" shortname="Track start" dimension="TIME" isint="false" />\n        <Feature feature="TRACK_STOP" name="Track stop" shortname="Track stop" dimension="TIME" isint="false" />\n        <Feature feature="TRACK_DISPLACEMENT" name="Track displacement" shortname="Track disp." dimension="LENGTH" isint="false" />\n        <Feature feature="TRACK_X_LOCATION" name="Track mean X" shortname="Track X" dimension="POSITION" isint="false" />\n        <Feature feature="TRACK_Y_LOCATION" name="Track mean Y" shortname="Track Y" dimension="POSITION" isint="false" />\n        <Feature feature="TRACK_Z_LOCATION" name="Track mean Z" shortname="Track Z" dimension="POSITION" isint="false" />\n        <Feature feature="TRACK_MEAN_SPEED" name="Track mean speed" shortname="Mean sp." dimension="VELOCITY" isint="false" />\n        <Feature feature="TRACK_MAX_SPEED" name="Track max speed" shortname="Max speed" dimension="VELOCITY" isint="false" />\n        <Feature feature="TRACK_MIN_SPEED" name="Track min speed" shortname="Min speed" dimension="VELOCITY" isint="false" />\n        <Feature feature="TRACK_MEDIAN_SPEED" name="Track median speed" shortname="Med. speed" dimension="VELOCITY" isint="false" />\n        <Feature feature="TRACK_STD_SPEED" name="Track std speed" shortname="Std speed" dimension="VELOCITY" isint="false" />\n        <Feature feature="TRACK_MEAN_QUALITY" name="Track mean quality" shortname="Mean Q" dimension="QUALITY" isint="false" />\n        <Feature feature="TRACK_MAX_QUALITY" name="Maximal quality" shortname="Max Q" dimension="QUALITY" isint="false" />\n        <Feature feature="TRACK_MIN_QUALITY" name="Minimal quality" shortname="Min Q" dimension="QUALITY" isint="false" />\n        <Feature feature="TRACK_MEDIAN_QUALITY" name="Median quality" shortname="Median Q" dimension="QUALITY" isint="false" />\n        <Feature feature="TRACK_STD_QUALITY" name="Quality standard deviation" shortname="Q std" dimension="QUALITY" isint="false" />\n        <Feature feature="TOTAL_DISTANCE_TRAVELED" name="Total distance traveled" shortname="Total dist." dimension="LENGTH" isint="false" />\n        <Feature feature="MAX_DISTANCE_TRAVELED" name="Max distance traveled" shortname="Max dist." dimension="LENGTH" isint="false" />\n        <Feature feature="CONFINEMENT_RATIO" name="Confinement ratio" shortname="Cfn. ratio" dimension="NONE" isint="false" />\n        <Feature feature="MEAN_STRAIGHT_LINE_SPEED" name="Mean straight line speed" shortname="Mn. v. line" dimension="VELOCITY" isint="false" />\n        <Feature feature="LINEARITY_OF_FORWARD_PROGRESSION" name="Linearity of forward progression" shortname="Fwd. progr." dimension="NONE" isint="false" />\n        <Feature feature="MEAN_DIRECTIONAL_CHANGE_RATE" name="Mean directional change rate" shortname="Mn. γ rate" dimension="ANGLE_RATE" isint="false" />\n      </TrackFeatures>\n    </FeatureDeclarations>\n')
        nspots = 0
        frames = []
        new_all_frames = {}
        all_spot_IDs = {}
        for len_ID in all_tracks:
            new_all_frames[len_ID] = []
            nspots = nspots + all_tracks[len_ID].shape[0] * all_tracks[len_ID].shape[1]
            if all_frames == None:
                frames = frames + list(np.arange(0, all_tracks[len_ID].shape[1]))
                new_all_frames[len_ID] = np.repeat(np.arange(0, all_tracks[len_ID].shape[1])[None], all_tracks[len_ID].shape[0], 0)
            else:
                for frame in all_frames[len_ID]:
                    frames = frames + list(frame) # not tested
        if all_frames == None:
            all_frames = new_all_frames
        frames = np.unique(frames)
        f.write('    <AllSpots nspots="%s">\n      <SpotsInFrame frame="0">\n'%nspots)
        spot_ID = 0
        for len_ID in all_tracks:
            all_spot_IDs[len_ID] = np.zeros(all_frames[len_ID].shape).astype(int)
        for frame in frames:
            for len_ID in all_tracks:
                cur_frames = all_frames[len_ID]
                tracks = all_tracks[len_ID]
                for i, (track, fm) in enumerate(zip(tracks, cur_frames)):
                    print(i)
                    pos_ID = np.where(frame == fm)[0]
                    if len(pos_ID)>0:
                        
                        pos_ID = pos_ID[0]
                        pos = np.zeros((3))
                        pos[:len(track[pos_ID])] = track[pos_ID]
                        all_spot_IDs[len_ID][i, pos_ID] = spot_ID
                        f.write('        <Spot ID="%s" name="ID%s" VISIBILITY="1" RADIUS="0.25" QUALITY="1.0" POSITION_T="%s" POSITION_X="%s" POSITION_Y="%s" FRAME="%s" POSITION_Z="%s" />\n'%(spot_ID, spot_ID, frame*dt, pos[0], pos[1], frame,pos[2]))
                        spot_ID = spot_ID + 1
        f.write('      </SpotsInFrame>\n    </AllSpots>\n    <AllTracks>\n')
        track_ID = 0
        all_track_IDs = []
        for len_ID in all_tracks:
            tracks = all_tracks[len_ID]
            frames = all_frames[len_ID]
            spot_IDss = all_spot_IDs[len_ID]
            for track, frame, spot_IDs in zip(tracks, frames, spot_IDss):
                f.write('      <Track name="Track_%s" TRACK_ID="%s" TRACK_INDEX="%s" NUMBER_SPOTS="%s" NUMBER_GAPS="%s" LONGEST_GAP="0" NUMBER_SPLITS="0" NUMBER_MERGES="0" NUMBER_COMPLEX="0" TRACK_DURATION="%s" TRACK_START="%s" TRACK_STOP="%s" TRACK_DISPLACEMENT="0.0" TRACK_X_LOCATION="0.0" TRACK_Y_LOCATION="0.0" TRACK_Z_LOCATION="0.0" TRACK_MEAN_SPEED="0.0" TRACK_MAX_SPEED="0.0" TRACK_MIN_SPEED="0.0" TRACK_MEDIAN_SPEED="0.0" TRACK_STD_SPEED="0.0" TRACK_MEAN_QUALITY="1.0" TRACK_MAX_QUALITY="1.0" TRACK_MIN_QUALITY="1.0" TRACK_MEDIAN_QUALITY="1.0" TRACK_STD_QUALITY="0.0" TOTAL_DISTANCE_TRAVELED="0.0" MAX_DISTANCE_TRAVELED="0.0" CONFINEMENT_RATIO="0.0" MEAN_STRAIGHT_LINE_SPEED="0.0" LINEARITY_OF_FORWARD_PROGRESSION="0.0" MEAN_DIRECTIONAL_CHANGE_RATE="0.0">\n'%(track_ID, track_ID, track_ID, track.shape[0], frame[-1] - frame[0] + 1 - track.shape[0], (frame[-1] - frame[0])*dt, frame[0]*dt, frame[-1]*dt))
                (pos, fm, spot_ID) = track[0], frame[0], spot_IDs[0]
                previous_spot_ID = spot_ID
                for pos, fm, spot_ID in zip(track[1:], frame[1:], spot_IDs[1:]):
                    f.write('        <Edge SPOT_SOURCE_ID="%s" SPOT_TARGET_ID="%s" LINK_COST="1.0" EDGE_TIME="%s" EDGE_X_LOCATION="%s" EDGE_Y_LOCATION="%s" EDGE_Z_LOCATION="0.0" VELOCITY="0.0" DISPLACEMENT="0.0" DIRECTIONAL_CHANGE_RATE="NaN" SPEED="0.0" />\n'%(previous_spot_ID, spot_ID, dt/2 + (track.shape[0]-1)*dt, pos[0], pos[1]))
                    previous_spot_ID = spot_ID
                all_track_IDs.append(track_ID)
                track_ID = track_ID + 1
                f.write('      </Track>\n')
        f.write('    </AllTracks>\n    <FilteredTracks>\n')
        for track_ID in all_track_IDs:
            f.write('      <TrackID TRACK_ID="%s" />\n'%track_ID)
        f.write('    </FilteredTracks>\n')
        f.write('  </Model>\n  <Settings>\n    <ImageData filename="blank" folder="" width="512" height="512" nslices="1" nframes="10" pixelwidth="0.041015625" pixelheight="0.041015625" voxeldepth="0.0" timeinterval="1.0" />\n    <BasicSettings xstart="0" xend="511" ystart="0" yend="511" zstart="0" zend="0" tstart="0" tend="9" />\n    <DetectorSettings />\n    <InitialSpotFilter feature="QUALITY" value="0.0" isabove="true" />\n    <SpotFilterCollection />\n    <TrackerSettings />\n    <TrackFilterCollection />\n    <AnalyzerCollection>\n      <SpotAnalyzers>\n        <Analyzer key="Manual spot color" />\n        <Analyzer key="Spot intensity" />\n        <Analyzer key="EXTRACK_PROBABILITIES" />\n        <Analyzer key="Spot contrast and SNR" />\n      </SpotAnalyzers>\n      <EdgeAnalyzers>\n        <Analyzer key="Directional change" />\n        <Analyzer key="Edge speed" />\n        <Analyzer key="Edge target" />\n        <Analyzer key="Edge location" />\n        <Analyzer key="Manual edge color" />\n        <Analyzer key="EXTRACK_EDGE_FEATURES" />\n      </EdgeAnalyzers>\n      <TrackAnalyzers>\n        <Analyzer key="Branching analyzer" />\n        <Analyzer key="Track duration" />\n        <Analyzer key="Track index" />\n        <Analyzer key="Track location" />\n        <Analyzer key="Track speed" />\n        <Analyzer key="Track quality" />\n        <Analyzer key="Track motility analysis" />\n      </TrackAnalyzers>\n    </AnalyzerCollection>\n  </Settings>\n  <GUIState state="ConfigureViews" />\n  <DisplaySettings>{\n  "name": "CurrentDisplaySettings",\n  "spotUniformColor": "204, 51, 204, 255",\n  "spotColorByType": "TRACKS",\n  "spotColorByFeature": "TRACK_INDEX",\n  "spotDisplayRadius": 0.1,\n  "spotDisplayedAsRoi": true,\n  "spotMin": 0.0,\n  "spotMax": 10.0,\n  "spotShowName": false,\n  "trackMin": 0.0,\n  "trackMax": 10.0,\n  "trackColorByType": "TRACKS",\n  "trackColorByFeature": "TRACK_INDEX",\n  "trackUniformColor": "204, 204, 51, 255",\n  "undefinedValueColor": "0, 0, 0, 255",\n  "missingValueColor": "89, 89, 89, 255",\n  "highlightColor": "51, 230, 51, 255",\n  "trackDisplayMode": "LOCAL",\n  "colormap": "Jet",\n  "limitZDrawingDepth": false,\n  "drawingZDepth": 10.0,\n  "fadeTracks": true,\n  "fadeTrackRange": 5,\n  "useAntialiasing": true,\n  "spotVisible": true,\n  "trackVisible": true,\n  "font": {\n    "name": "Arial",\n    "style": 1,\n    "size": 12,\n    "pointSize": 12.0,\n    "fontSerializedDataVersion": 1\n  },\n  "lineThickness": 1.0,\n  "selectionLineThickness": 4.0,\n  "trackschemeBackgroundColor1": "128, 128, 128, 255",\n  "trackschemeBackgroundColor2": "192, 192, 192, 255",\n  "trackschemeForegroundColor": "0, 0, 0, 255",\n  "trackschemeDecorationColor": "0, 0, 0, 255",\n  "trackschemeFillBox": false,\n  "spotFilled": false,\n  "spotTransparencyAlpha": 1.0\n}</DisplaySettings>\n</TrackMate>\n')
