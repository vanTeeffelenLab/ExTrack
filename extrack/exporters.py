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

def extrack_2_pandas(tracks, pred_Bs, frames = None, opt_metrics = {}):
    '''
    turn outputs form ExTrack to a unique pandas DataFrame
    '''
    if frames is None:
        frames = {}
        for l in tracks:
            frames[l] = np.repeat(np.array([np.arange(int(l))]), len(tracks[l]), axis = 0)
    
    track_list = []
    frames_list = []
    track_ID_list = []
    opt_metrics_list = []
    for metric in opt_metrics:
        opt_metrics_list.append([])
    
    cur_nb_track = 0
    pred_Bs_list = []
    for l in tracks:
        track_list = track_list + list(tracks[l].reshape(tracks[l].shape[0] * tracks[l].shape[1], 2))
        frames_list = frames_list + list(frames[l].reshape(frames[l].shape[0] * frames[l].shape[1], 1))
        track_ID_list = track_ID_list + list(np.repeat(np.arange(cur_nb_track,cur_nb_track+tracks[l].shape[0]),tracks[l].shape[1]))
        cur_nb_track += tracks[l].shape[0]
        
        for j, metric in enumerate(opt_metrics):
            opt_metrics_list[j] = opt_metrics_list[j] + list(opt_metrics[metric][l].reshape(opt_metrics[metric][l].shape[0] * opt_metrics[metric][l].shape[1], 1))
        
        n = pred_Bs[l].shape[2]
        pred_Bs_list = pred_Bs_list + list(pred_Bs[l].reshape(pred_Bs[l].shape[0] * pred_Bs[l].shape[1], n))
    
    all_data = np.concatenate((np.array(track_list), np.array(frames_list), np.array(track_ID_list)[:,None], np.array(pred_Bs_list)), axis = 1)    
    for opt_metric in opt_metrics_list:
        all_data = np.concatenate((all_data, opt_metric), axis = 1)

    colnames = ['X', 'Y', 'frame', 'track_ID']
    for i in range(np.array(pred_Bs_list).shape[1]):
        colnames = colnames + ['pred_' + str(i)]
    for metric in opt_metrics:
        colnames = colnames + [metric]
    
    df = pd.DataFrame(data = all_data, index = np.arange(len(all_data)), columns = colnames)
    df['frame'] = df['frame'].astype(int)
    df['track_ID'] = df['track_ID'].astype(int)

    return df

def save_extrack_2_CSV(path, all_Css, pred_Bss, dt, all_frames = None):
    track_ID = 0
    
    preds_header_str_fmt = ''
    preds_str_fmt = ''
    for k in range(pred_Bss[list(pred_Bss.keys())[0]].shape[2]):
        preds_header_str_fmt = preds_header_str_fmt + 'PRED_%s,'%(k)
        preds_str_fmt = preds_str_fmt + ',%s'
    
    with open(path, 'w') as f:
        f.write('TRACK_ID,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,%s\n'%(preds_header_str_fmt))
    
        for len_ID in all_Css:
            all_Cs = all_Css[len_ID]
            pred_Bs = pred_Bss[len_ID]
            if all_frames != None:
                all_frame = all_frames[len_ID]
            else:
                all_frame = np.arange(all_Cs.shape[0]*all_Cs.shape[1]).reshape((all_Cs.shape[0],all_Cs.shape[1])) 
            for track, preds, frames in zip(all_Cs, pred_Bs, all_frame):
                track_ID+=1
                for pos, p, frame in zip(track, preds, frames):
                    preds_str = preds_str_fmt%(tuple(p))
                    f.write('%s,%s,%s,%s,%s,%s%s\n'%(track_ID, p[0], p[1], 0.0, dt* frame*1000, frame, preds_str))
def save_extrack_2_xml(all_tracks, pred_Bss, params, path, dt, all_frames = None, opt_metrics = {}):
    track_ID = 0
    for len_ID in all_tracks:
       tracks = all_tracks[len_ID]
       track_ID += len(tracks)
    
    final_params = []
    for param in params:
        if not '_' in param:
            final_params.append(param)
    Extrack_headers = 'ExTrack_results="'
    
    for param in final_params:
        Extrack_headers = Extrack_headers + param + "='" + str(np.round(params[param].value, -np.log10(params[param].value).astype(int)+5)) +"' " 
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
            tracks = all_tracks[len_ID]
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
                    f.write('    <detection t="%s" x="%s" y="%s" z="%s"%s %s/>\n'%(frame,p[0],p[1],0.0, preds_str, opt_metrics_str))
                f.write('  </particle>\n')
        f.write('</Tracks>\n')
