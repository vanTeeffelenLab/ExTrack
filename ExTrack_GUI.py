# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:15:37 2025

@author: Franc

This code enables to use the Graphical User interface of ExTrack. 
To create a stand alone version of ExTrack:
1) pip install pyinstaller
2) pyinstaller --onedir path\ExTrack_GUI.py
3) Copy the .ddl files starting with mkl into the dist\ExTrack_GUI\_internal (the mkl files can be found in C:\Users\Franc\anaconda3\Library\bin in my case)
4) execute dist\ExTrack_GUI.exe to run the stand alone software
"""

import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
print('tkinter',tk)

from tkinter import ttk
import webbrowser
import extrack
print(extrack.__file__)
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

#ttk = tk.ttk

padx = 10 # spacing between cells of the grid in x
pady = 10 # spacing between cells of the grid in y
previous_window = None

def open_analysis_window():
    global previous_window
    path = path_entry.get()
    print(os.path.normpath(path))
    savepath = os.path.normpath(path).rsplit(os.sep, 1)[0]
    (os.sep, 1)[0]
    min_length = int(min_length_entry.get())
    max_length = int(max_length_entry.get())
    analysis_type = analysis_type_var.get()
    LocErr_type = LocErr_type_var.get()
    LocErr_input_name = LocErr_input_entry.get().split(',')
    Optional_input_name = Optional_input_entry.get().split(',')
    headers = [x_pos_entry.get(), y_pos_entry.get(), frame_entry.get(), ID_entry.get()]
    max_dist = float(max_dist_entry.get())
    remove_no_disps = bool(remove_no_disp_entry.get())
    
    root.withdraw()
    previous_window = root
    
    analysis_window = tk.Tk()
    analysis_window.title("Anomalous Analysis - {}".format(analysis_type))
    
    if analysis_type == 'Model Fitting':
        create_fitting_window(analysis_window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps)
    elif analysis_type == 'State Labeling':
        create_prediction_window(analysis_window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps)
    elif analysis_type == 'State Lifetime Histogram':
        create_lifetime_window(analysis_window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps)
    elif analysis_type == 'Position Refinement':
        create_refinement_window(analysis_window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps)

def show_loading_window(root):
    loading_window = tk.Toplevel(root)
    loading_window.title("Loading")
    loading_window.geometry("200x100")
    label = tk.Label(loading_window, text="Loading, please wait...")
    label.pack(pady=10)
    progress = ttk.Progressbar(loading_window, mode='indeterminate')
    progress.pack(pady=10)
    progress.start()
    return loading_window, progress

def go_to_previous_window(window):
    window.destroy()
    if previous_window:
        previous_window.deiconify()

def show_error_url(window, message, url=None):
    window.withdraw()
    error_window = tk.Toplevel(window)
    error_window.title("Error")
    text = tk.Text(error_window, height=10, width=80, wrap="word")
    text.grid(row=0, column=0, padx=20, pady=10)
    text.insert(tk.END, message)
    if url:
        text.insert(tk.END, "https://github.com/FrancoisSimon/aTrack", "link")
        text.tag_config("link", foreground="blue", underline=True)
        text.tag_bind("link", "<Button-1>", lambda e, link=url: webbrowser.open(link))
    text.config(state="disabled")
    previous_button = ttk.Button(error_window, text="Previous", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=1, column=0)

def create_fitting_window(window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps):

    #try:
    print('path', path, type(path))
    print('headers', headers, type(headers), type(headers[0])) 
    print('remove_no_disp', remove_no_disps, type(remove_no_disps))
    print('Optional_input_name', Optional_input_name, type(Optional_input_name))
    print('max_dist', max_dist, type(max_dist))
    
    if os.path.isdir(path):
        path = glob(path + '/*.csv')
        
    if type(path) == str:
        if not path.endswith('.csv') :
            show_error_url(window, "Please select a csv file with an extention '.csv'.\n", url=None)
            return
    
    elif len(path)==0:
        show_error_url(window, "No csv file detected in the informed directory. Make sure the csv files end with the extention '.csv'.\n", url=None)
        return
    
    if LocErr_type == "Fitted parameter":
        tracks, frames, opt_metrics = extrack.readers.read_table(path,
                                                         lengths=np.arange(min_length, max_length+1),
                                                         dist_th=max_dist,
                                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                                         colnames = headers,
                                                         remove_no_disp=remove_no_disps,
                                                         opt_colnames = Optional_input_name)
        input_LocErr = None
      
    else:
        
        tracks, frames, opt_metrics = extrack.readers.read_table(path,
                                                         lengths=np.arange(min_length, max_length+1),
                                                         dist_th=max_dist,
                                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                                         colnames = headers,
                                                         remove_no_disp=remove_no_disps,
                                                         opt_colnames = Optional_input_name + LocErr_input_name)
        
        # then, we retreive input_LocErr from the optional metrics 
        input_LocErr = {}
        for l in tracks:
            input_LocErr[l] = np.zeros(tracks[l].shape[:2] + (len(LocErr_input_name),))
        
        for i, name in enumerate(LocErr_input_name):
            for l in tracks:
                input_LocErr[l][:,:,i] = opt_metrics[name][l]
            del opt_metrics[name]
        
    #except Exception as e:
        #show_error_url(window, "The csv file could not be read correctly.\nVerify that your file has columns named: 'POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'\nFor more details, click here:", "https://github.com/FrancoisSimon/aTrack")
        #return 
    
    global params

    # Initial number of states
    NbStates_label = ttk.Label(window, text="Number of states:")
    NbStates_label.grid(row=0, column=0, sticky = 'e', padx = padx, pady = pady)
    NbStates_entry = ttk.Entry(window, width=13)
    NbStates_entry.grid(row=0, column=1)
    NbStates_entry.insert(tk.END, str(params['num_states']))
    
    open_button = ttk.Button(window, text="Open Parameter Window", command=lambda: ParameterWindow(window, int(NbStates_entry.get())))
                             #ParameterWindow(window, int(NbStates_entry.get())))
    open_button.grid(row=1, column=0, sticky = 'e', padx = padx, pady = pady)
    
    # Frame time
    frametime_label = ttk.Label(window, text="Frame time (in s)")
    frametime_label.grid(row=2, column=0, sticky = 'e', padx = padx, pady = pady)
    frametime_entry = ttk.Entry(window, width=13)
    frametime_entry.grid(row=2, column=1)
    frametime_entry.insert(tk.END, str(params['dt']))
    
    # Window length
    windowlength_label = ttk.Label(window, text="Window length")
    windowlength_label.grid(row=3, column=0, sticky = 'e', padx = padx, pady = pady)
    windowlength_entry = ttk.Entry(window, width=13)
    windowlength_entry.grid(row=3, column=1)
    windowlength_entry.insert(tk.END, str(params['fitting_window_length']))
    
    # Number of substeps
    nb_substeps_label = ttk.Label(window, text="Number of substeps")
    nb_substeps_label.grid(row=4, column=0, sticky = 'e', padx = padx, pady = pady)
    nb_substeps_entry = ttk.Entry(window, width=13)
    nb_substeps_entry.grid(row=4, column=1)
    nb_substeps_entry.insert(tk.END, str(params['nb_substeps']))
    
    # Threshold to fuse sequences of states
    Threshold_label = ttk.Label(window, text="Threshold")
    Threshold_label.grid(row=5, column=0, sticky = 'e', padx = padx, pady = pady)
    Threshold_entry = ttk.Entry(window, width=13)
    Threshold_entry.grid(row=5, column=1)
    Threshold_entry.insert(tk.END, str(params['threshold']))
    
    # Maximum number of states
    Max_nb_sequences_label = ttk.Label(window, text="Maximum number of sequences")
    Max_nb_sequences_label.grid(row=6, column=0, sticky = 'e', padx = padx, pady = pady)
    Max_nb_sequences_entry = ttk.Entry(window, width=13)
    Max_nb_sequences_entry.grid(row=6, column=1)
    Max_nb_sequences_entry.insert(tk.END, str(params['max_nb_sequ']))
    
    # Depth of field
    Depth_of_field_label = ttk.Label(window, text="Depth of field")
    Depth_of_field_label.grid(row=7, column=0, sticky = 'e', padx = padx, pady = pady)
    Depth_of_field_entry = ttk.Entry(window, width=13)
    Depth_of_field_entry.grid(row=7, column=1)
    Depth_of_field_entry.insert(tk.END, str(params['cell_dims']))
    
    # number of iterations of the fitting methods
    nb_iter_label = ttk.Label(window, text="Number of iterations")
    nb_iter_label.grid(row=8, column=0, sticky = 'e', padx = padx, pady = pady)
    nb_iter_entry = ttk.Entry(window, width=13)
    nb_iter_entry.grid(row=8, column=1)
    nb_iter_entry.insert(tk.END, str(params['nb_iters']))
    
    # Savepath Input
    savepath_label = ttk.Label(window, text="Save Path:")
    savepath_label.grid(row=9, column=0, sticky = 'e', padx = padx, pady = pady)
    savepath_entry = ttk.Entry(window, width=50)
    savepath_entry.grid(row=9, column=1)
    savepath_entry.insert(tk.END, os.path.join(savepath, 'saved_fitting_results.csv'))
    savepath_button = ttk.Button(window, text="Browse", command=lambda: browse_savepath(savepath_entry))
    savepath_button.grid(row=9, column=2)
    
    # Run Button
    run_button = ttk.Button(window, text="Start fitting", command=lambda: run_fitting(window,
                                                                                      tracks, 
                                                                                     dt = float(frametime_entry.get()), 
                                                                                     nb_states = int(NbStates_entry.get()),
                                                                                     nb_iterations = int(nb_iter_entry.get()),
                                                                                     nb_substeps = int(nb_substeps_entry.get()), 
                                                                                     frame_len = int(windowlength_entry.get()), 
                                                                                     cell_dims = float(Depth_of_field_entry.get()), 
                                                                                     LocErr_type = LocErr_type,
                                                                                     input_LocErr = input_LocErr, 
                                                                                     threshold = float(Threshold_entry.get()), 
                                                                                     max_nb_states = int(Max_nb_sequences_entry.get()), 
                                                                                     savepath = savepath_entry.get()))
    run_button.grid(row=10, column=1, columnspan=1)
    
    # Previous Button
    previous_button = ttk.Button(window, text="Other analyses", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=10, column=0, columnspan=1)

def run_fitting(window, tracks, dt, nb_states, nb_iterations, nb_substeps, frame_len, cell_dims, LocErr_type, input_LocErr, threshold, max_nb_states, savepath):
    # Run the Brownian motion analysis
    #tracks = tracks[str(length)]
    global params

    params['dt'] = dt
    params['fitting_window_length'] = frame_len
    params['cell_dims'] = cell_dims
    params['max_nb_sequ'] = max_nb_states
    params['threshold'] = threshold
    params['nb_iters'] = nb_iterations
    params['nb_substeps'] = nb_substeps
    
    if params['num_states'] != nb_states:
        get_new_params(nb_states)
    
    if LocErr_type == "Inputing a quality metric for each peak":
        try:
            for l in input_LocErr:
                input_LocErr[l] = 1/input_LocErr[l]**0.5
        except:
            raise ValueError("If you chose to estimate the localization error from a quality metric, the quality metrics must all be numerical and strictly positive")
        
    lmfit_params = params_to_lmfit_params(params, LocErr_type)
    
    print('lmfit_params', lmfit_params)
    #print('tracks', tracks)
    print('input_LocErr', input_LocErr)
    print('nb_states', nb_states, type(nb_states))
    
    for l in tracks:
        print(tracks[l].shape)
    model_fit = extrack.tracking.param_fitting(tracks,
                                      dt,
                                      params = lmfit_params,
                                      nb_states = nb_states,
                                      nb_substeps = nb_substeps,
                                      frame_len = frame_len,
                                      verbose = 1,
                                      workers = 1,
                                      Matrix_type = 1,
                                      method = 'powell',
                                      steady_state = False,
                                      cell_dims = [cell_dims], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                                      input_LocErr = input_LocErr, 
                                      threshold = threshold, 
                                      max_nb_states = max_nb_states)
    
    for k in range(nb_iterations-1):
        model_fit = extrack.tracking.param_fitting(tracks,
                                          dt,
                                          params = model_fit.params,
                                          nb_states = nb_states,
                                          nb_substeps = nb_substeps,
                                          frame_len = frame_len,
                                          verbose = 1,
                                          workers = 1,
                                          Matrix_type = 1,
                                          method = 'bfgs',
                                          steady_state = False,
                                          cell_dims = [cell_dims], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                                          input_LocErr = input_LocErr, 
                                          threshold = threshold,
                                          max_nb_states = max_nb_states)
    
    lmfit_params = model_fit.params
    
    data = pd.DataFrame([], columns = ['exp', 'likelihood'] + list(lmfit_params.keys()))

    vals = [savepath, - model_fit.residual[0]]
    for param in lmfit_params:
        vals.append(lmfit_params[param].value)
    
    data.loc[len(data.index)] = vals
    data.to_csv(savepath)
    
    lmfit_params_to_params(lmfit_params)
    
    print("Fitting analysis completed and results saved to %s"%savepath)
    print(data)


def create_prediction_window(window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps):
    if not path.endswith('.csv'):
        show_error_url(window, "Please select a csv file with an extention .csv\n", url=None)
        return
    #try:
    print('path', path, type(path))
    print('headers', headers, type(headers), type(headers[0])) 
    print('remove_no_disp', remove_no_disps, type(remove_no_disps))
    print('Optional_input_name', Optional_input_name, type(Optional_input_name))
    print('max_dist', max_dist, type(max_dist))
    
    if os.path.isdir(path):
        path = glob(path + '/*.csv')
    
    if LocErr_type == "Fitted parameter":
        tracks, frames, opt_metrics = extrack.readers.read_table(path,
                                                         lengths=np.arange(min_length, max_length+1),
                                                         dist_th=max_dist,
                                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                                         colnames = headers,
                                                         remove_no_disp=remove_no_disps,
                                                         opt_colnames = Optional_input_name)
        input_LocErr = None
      
    else:
        
        tracks, frames, opt_metrics = extrack.readers.read_table(path,
                                                         lengths=np.arange(min_length, max_length+1),
                                                         dist_th=max_dist,
                                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                                         colnames = headers,
                                                         remove_no_disp=remove_no_disps,
                                                         opt_colnames = Optional_input_name + LocErr_input_name)
        
        # then, we retreive input_LocErr from the optional metrics 
        input_LocErr = {}
        for l in tracks:
            input_LocErr[l] = np.zeros(tracks[l].shape[:2] + (len(LocErr_input_name),))
        
        for i, name in enumerate(LocErr_input_name):
            for l in tracks:
                input_LocErr[l][:,:,i] = opt_metrics[name][l]
            del opt_metrics[name]
        
    global params
    
    # Initial number of states
    NbStates_label = ttk.Label(window, text="Number of states:")
    NbStates_label.grid(row=0, column=0, sticky = 'e', padx = padx, pady = pady)
    NbStates_entry = ttk.Entry(window, width=13)
    NbStates_entry.grid(row=0, column=1)
    NbStates_entry.insert(tk.END, str(params['num_states']))
    
    open_button = ttk.Button(window, text="Open Parameter Window", command=lambda: ParameterWindow(window, int(NbStates_entry.get())))
                             #ParameterWindow(window, int(NbStates_entry.get())))
    open_button.grid(row=1, column=0, sticky = 'e', padx = padx, pady = pady)
    
    # Frame time
    frametime_label = ttk.Label(window, text="Frame time (in s)")
    frametime_label.grid(row=2, column=0, sticky = 'e', padx = padx, pady = pady)
    frametime_entry = ttk.Entry(window, width=13)
    frametime_entry.grid(row=2, column=1)
    frametime_entry.insert(tk.END, str(params['dt']))
    
    # Window length
    windowlength_label = ttk.Label(window, text="Window length")
    windowlength_label.grid(row=3, column=0, sticky = 'e', padx = padx, pady = pady)
    windowlength_entry = ttk.Entry(window, width=13)
    windowlength_entry.grid(row=3, column=1)
    windowlength_entry.insert(tk.END, str(params['labeling_window_length']))
    
    # Threshold to fuse sequences of states
    Threshold_label = ttk.Label(window, text="Threshold")
    Threshold_label.grid(row=5, column=0, sticky = 'e', padx = padx, pady = pady)
    Threshold_entry = ttk.Entry(window, width=13)
    Threshold_entry.grid(row=5, column=1)
    Threshold_entry.insert(tk.END, str(params['threshold']))
    
    # Maximum number of states
    Max_nb_sequences_label = ttk.Label(window, text="Maximum number of sequences")
    Max_nb_sequences_label.grid(row=6, column=0, sticky = 'e', padx = padx, pady = pady)
    Max_nb_sequences_entry = ttk.Entry(window, width=13)
    Max_nb_sequences_entry.grid(row=6, column=1)
    Max_nb_sequences_entry.insert(tk.END, str(params['max_nb_sequ_labeling']))
    
    # Depth of field
    Depth_of_field_label = ttk.Label(window, text="Depth of field")
    Depth_of_field_label.grid(row=7, column=0, sticky = 'e', padx = padx, pady = pady)
    Depth_of_field_entry = ttk.Entry(window, width=13)
    Depth_of_field_entry.grid(row=7, column=1)
    Depth_of_field_entry.insert(tk.END, str(params['cell_dims']))
    
    # Savepath Input
    savepath_label = ttk.Label(window, text="Save Path:")
    savepath_label.grid(row=9, column=0, sticky = 'e', padx = padx, pady = pady)
    savepath_entry = ttk.Entry(window, width=50)
    savepath_entry.grid(row=9, column=1)
    savepath_entry.insert(tk.END, os.path.join(savepath, 'saved_track_predictions.csv'))
    savepath_button = ttk.Button(window, text="Browse", command=lambda: browse_savepath(savepath_entry))
    savepath_button.grid(row=9, column=2)
    
    # Run Button
    run_button = ttk.Button(window,
                            text="Start state predictions", 
                            command=lambda: run_predictions(window,
                                                            tracks,
                                                            frames,
                                                            opt_metrics,
                                                            dt = float(frametime_entry.get()), 
                                                            nb_states = int(NbStates_entry.get()),
                                                            frame_len = int(windowlength_entry.get()), 
                                                            cell_dims = float(Depth_of_field_entry.get()), 
                                                            LocErr_type = LocErr_type,
                                                            input_LocErr = input_LocErr, 
                                                            threshold = float(Threshold_entry.get()), 
                                                            max_nb_states = int(Max_nb_sequences_entry.get()),
                                                            savepath = savepath_entry.get()))
    run_button.grid(row=10, column=1, columnspan=1)
    
    # Previous Button
    previous_button = ttk.Button(window, text="Previous", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=10, column=0, columnspan=1)

def run_predictions(window, tracks, frames, opt_metrics, dt, nb_states, frame_len, cell_dims, LocErr_type, input_LocErr, threshold, max_nb_states, savepath):
    # Run the Brownian motion analysis
    #tracks = tracks[str(length)]
    global params
    
    params['dt'] = dt
    params['labeling_window_length'] = frame_len
    params['cell_dims'] = cell_dims
    params['max_nb_sequ_labeling'] = max_nb_states
    params['threshold'] = threshold
    
    print("params['num_states']", params['num_states'])
    print('nb_states', nb_states)
    
    if params['num_states'] != nb_states:
        get_new_params(nb_states)
    
    nb_states
    
    if LocErr_type == "Inputing a quality metric for each peak":
        try:
            for l in input_LocErr:
                input_LocErr[l] = 1/input_LocErr[l]**0.5
        except:
            raise ValueError("If you chose to estimate the localization error from a quality metric, the quality metrics must all be numerical and strictly positive")
    
    lmfit_params = params_to_lmfit_params(params, LocErr_type)
    
    print('lmfit_params', lmfit_params)
    #print('tracks', tracks)
    #print('input_LocErr', input_LocErr)
    print('nb_states', nb_states, type(nb_states))
    
    for l in tracks:
        print(tracks[l].shape)
    
    preds = extrack.tracking.predict_Bs(tracks,
                           dt,
                           lmfit_params,
                           cell_dims=[cell_dims],
                           nb_states=nb_states,
                           frame_len=frame_len,
                           max_nb_states = max_nb_states,
                           threshold = threshold,
                           workers = 1,
                           input_LocErr = input_LocErr,
                           verbose = 0,
                           nb_max = 1)
    
    DATA = extrack.exporters.extrack_2_pandas(tracks, preds, frames = frames, opt_metrics = opt_metrics)    
    DATA.to_csv(savepath)
    
    print("State labeling completed and results saved to %s."%savepath)

def create_lifetime_window(window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps):
    if not path.endswith('.csv'):
        show_error_url(window, "Please select a csv file with an extention .csv\n", url=None)
        return
    #try:
    print('path', path, type(path))
    print('headers', headers, type(headers), type(headers[0])) 
    print('remove_no_disp', remove_no_disps, type(remove_no_disps))
    print('Optional_input_name', Optional_input_name, type(Optional_input_name))
    print('max_dist', max_dist, type(max_dist))
    
    if os.path.isdir(path):
        path = glob(path + '/*.csv')
    
    if LocErr_type == "Fitted parameter":
        tracks, frames, opt_metrics = extrack.readers.read_table(path,
                                                         lengths=np.arange(min_length, max_length+1),
                                                         dist_th=max_dist,
                                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                                         colnames = headers,
                                                         remove_no_disp=remove_no_disps,
                                                         opt_colnames = Optional_input_name)
        input_LocErr = None
    
    else:
        
        tracks, frames, opt_metrics = extrack.readers.read_table(path,
                                                         lengths=np.arange(min_length, max_length+1),
                                                         dist_th=max_dist,
                                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                                         colnames = headers,
                                                         remove_no_disp=remove_no_disps,
                                                         opt_colnames = Optional_input_name + LocErr_input_name)
        
        # then, we retreive input_LocErr from the optional metrics 
        input_LocErr = {}
        for l in tracks:
            input_LocErr[l] = np.zeros(tracks[l].shape[:2] + (len(LocErr_input_name),))
        
        for i, name in enumerate(LocErr_input_name):
            for l in tracks:
                input_LocErr[l][:,:,i] = opt_metrics[name][l]
            del opt_metrics[name]
        
    global params
    
    # Initial number of states
    NbStates_label = ttk.Label(window, text="Number of states:")
    NbStates_label.grid(row=0, column=0, sticky = 'e', padx = padx, pady = pady)
    NbStates_entry = ttk.Entry(window, width=13)
    NbStates_entry.grid(row=0, column=1)
    NbStates_entry.insert(tk.END, str(params['num_states']))
    
    open_button = ttk.Button(window, text="Open Parameter Window", command=lambda: ParameterWindow(window, int(NbStates_entry.get())))
                             #ParameterWindow(window, int(NbStates_entry.get())))
    open_button.grid(row=1, column=0, sticky = 'e', padx = padx, pady = pady)
    
    # Frame time
    frametime_label = ttk.Label(window, text="Frame time (in s)")
    frametime_label.grid(row=2, column=0, sticky = 'e', padx = padx, pady = pady)
    frametime_entry = ttk.Entry(window, width=13)
    frametime_entry.grid(row=2, column=1)
    frametime_entry.insert(tk.END, str(params['dt']))
    
    # Maximum number of states
    Max_nb_sequences_label = ttk.Label(window, text="Maximum number of sequences")
    Max_nb_sequences_label.grid(row=6, column=0, sticky = 'e', padx = padx, pady = pady)
    Max_nb_sequences_entry = ttk.Entry(window, width=13)
    Max_nb_sequences_entry.grid(row=6, column=1)
    Max_nb_sequences_entry.insert(tk.END, str(params['max_nb_sequ_histograms']))
    
    # Depth of field
    Depth_of_field_label = ttk.Label(window, text="Depth of field")
    Depth_of_field_label.grid(row=7, column=0, sticky = 'e', padx = padx, pady = pady)
    Depth_of_field_entry = ttk.Entry(window, width=13)
    Depth_of_field_entry.grid(row=7, column=1)
    Depth_of_field_entry.insert(tk.END, str(params['cell_dims']))
    
    Draw_plot_label = ttk.Label(window, text="Plot lifetime histograms")
    Draw_plot_label.grid(row=8, column=0, padx = padx, pady = pady, sticky = 'e')
    Draw_plot_var = tk.StringVar(window)
    Draw_plot_var.set(params['draw_plot'])
    Draw_plot_dropdown = ttk.OptionMenu(window, Draw_plot_var, Draw_plot_var.get(),
                                             "Yes",
                                             "No",
                                             style='My.TMenubutton')
    #Draw_plot_dropdown.config(width=15)
    Draw_plot_dropdown.grid(row=8, column=1, padx = padx, pady = pady, sticky="e")
    
    # Savepath Input
    savepath_label = ttk.Label(window, text="Save Path:")
    savepath_label.grid(row=9, column=0, sticky = 'e', padx = padx, pady = pady)
    savepath_entry = ttk.Entry(window, width=50)
    savepath_entry.grid(row=9, column=1)
    savepath_entry.insert(tk.END, os.path.join(savepath, 'saved_lifetimes.csv'))
    savepath_button = ttk.Button(window, text="Browse", command=lambda: browse_savepath(savepath_entry))
    savepath_button.grid(row=9, column=2)
    
    # Run Button
    run_button = ttk.Button(window,
                            text="Compute lifetime histogram", 
                            command=lambda: run_lifetime(window,
                                                            tracks,
                                                            dt = float(frametime_entry.get()), 
                                                            nb_states = int(NbStates_entry.get()),
                                                            cell_dims = float(Depth_of_field_entry.get()), 
                                                            LocErr_type = LocErr_type,
                                                            input_LocErr = input_LocErr, 
                                                            max_nb_states = int(Max_nb_sequences_entry.get()),
                                                            draw_plot = Draw_plot_var.get(),
                                                            savepath = savepath_entry.get()))
    run_button.grid(row=10, column=1, columnspan=1)
    
    # Previous Button
    previous_button = ttk.Button(window, text="Previous", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=10, column=0, columnspan=1)

def run_lifetime(window, tracks, dt, nb_states, cell_dims, LocErr_type, input_LocErr, max_nb_states, draw_plot, savepath):
    # Run the Brownian motion analysis
    #tracks = tracks[str(length)]
    global params
    
    print("params['num_states']", params['num_states'])
    print('nb_states', nb_states)
    
    params['dt'] = dt
    params['cell_dims'] = cell_dims
    params['max_nb_sequ_histograms'] = max_nb_states
    params['draw_plot'] = draw_plot
    
    if params['num_states'] != nb_states:
        get_new_params(nb_states)
    
    if LocErr_type == "Inputing a quality metric for each peak":
        #print('input_LocErr', input_LocErr)
        try:
            for l in input_LocErr:
                input_LocErr[l] = 1/input_LocErr[l]**0.5
        except:
            raise ValueError("If you chose to estimate the localization error from a quality metric, the quality metrics must all be numerical and strictly positive")
    
    lmfit_params = params_to_lmfit_params(params, LocErr_type)
    
    print('lmfit_params', lmfit_params)
    #print('tracks', tracks)
    #print('input_LocErr', input_LocErr)
    print('nb_states', nb_states, type(nb_states))
    
    for l in tracks:
        print(tracks[l].shape)
    
    hists = extrack.histograms.len_hist(tracks,
                                        lmfit_params, 
                                        dt, 
                                        cell_dims=[cell_dims], 
                                        nb_states=nb_states, 
                                        max_nb_states = max_nb_states,
                                        workers = 1,
                                        nb_substeps=1,
                                        input_LocErr = input_LocErr
                                        )
    
    columns = ['Segment length']
    for state in range(nb_states):
        columns.append('State %s'%state)
    
    DATA = pd.DataFrame(np.concatenate((np.arange(1,len(hists)+1)[:,None], hists), axis = 1, dtype = 'str'), columns = columns)
    DATA.to_csv(savepath)
    
    print("State labeling completed and results saved to %s."%savepath)
    if draw_plot=='Yes':
        plt.figure(figsize = (4.8,3.5))
        plt.title('Plot of the lifetime histograms of the different states', font = "Arial", fontsize = 12)
        plt.plot(np.arange(1,len(hists)+1)[:,None]*dt, hists)
        plt.ylabel('Counts')
        plt.xlabel('Time in s')
        plt.legend(np.arange(nb_states), title = 'State')
        plt.tight_layout()
        
        plt.figure(figsize = (4.8,3.5))
        plt.title('Log Plot of the lifetime histograms of the different states', font = "Arial", fontsize = 12)
        plt.plot(np.arange(1,len(hists)+1)[:,None]*dt, hists)
        plt.ylabel('Counts')
        plt.xlabel('Time in s')
        plt.yscale('log')
        plt.legend(np.arange(nb_states), title = 'State')
        plt.tight_layout()
        plt.show()

def create_refinement_window(window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps):
    if not path.endswith('.csv'):
        show_error_url(window, "Please select a csv file with an extention .csv\n", url=None)
        return
    #try:
    print('path', path, type(path))
    print('headers', headers, type(headers), type(headers[0])) 
    print('remove_no_disp', remove_no_disps, type(remove_no_disps))
    print('Optional_input_name', Optional_input_name, type(Optional_input_name))
    print('max_dist', max_dist, type(max_dist))
    
    if os.path.isdir(path):
        path = glob(path + '/*.csv')
    
    if LocErr_type == "Fitted parameter":
        tracks, frames, opt_metrics = extrack.readers.read_table(path,
                                                         lengths=np.arange(min_length, max_length+1),
                                                         dist_th=max_dist,
                                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                                         colnames = headers,
                                                         remove_no_disp=remove_no_disps,
                                                         opt_colnames = Optional_input_name)
        input_LocErr = None
      
    else:
        
        tracks, frames, opt_metrics = extrack.readers.read_table(path,
                                                         lengths=np.arange(min_length, max_length+1),
                                                         dist_th=max_dist,
                                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                                         colnames = headers,
                                                         remove_no_disp=remove_no_disps,
                                                         opt_colnames = Optional_input_name + LocErr_input_name)
        
        # then, we retreive input_LocErr from the optional metrics 
        input_LocErr = {}
        for l in tracks:
            input_LocErr[l] = np.zeros(tracks[l].shape[:2] + (len(LocErr_input_name),))
        
        for i, name in enumerate(LocErr_input_name):
            for l in tracks:
                input_LocErr[l][:,:,i] = opt_metrics[name][l]
            del opt_metrics[name]
        
    global params
    
    # Initial number of states
    NbStates_label = ttk.Label(window, text="Number of states:")
    NbStates_label.grid(row=0, column=0, sticky = 'e', padx = padx, pady = pady)
    NbStates_entry = ttk.Entry(window, width=13)
    NbStates_entry.grid(row=0, column=1)
    NbStates_entry.insert(tk.END, str(params['num_states']))
    
    open_button = ttk.Button(window, text="Open Parameter Window", command=lambda: ParameterWindow(window, int(NbStates_entry.get())))
                             #ParameterWindow(window, int(NbStates_entry.get())))
    open_button.grid(row=1, column=0, sticky = 'e', padx = padx, pady = pady)
    
    # Frame time
    frametime_label = ttk.Label(window, text="Frame time (in s)")
    frametime_label.grid(row=2, column=0, sticky = 'e', padx = padx, pady = pady)
    frametime_entry = ttk.Entry(window, width=13)
    frametime_entry.grid(row=2, column=1)
    frametime_entry.insert(tk.END, str(params['dt']))
    
    # Window length
    windowlength_label = ttk.Label(window, text="Window length")
    windowlength_label.grid(row=3, column=0, sticky = 'e', padx = padx, pady = pady)
    windowlength_entry = ttk.Entry(window, width=13)
    windowlength_entry.grid(row=3, column=1)
    windowlength_entry.insert(tk.END, str(params['labeling_window_length']))
    
    # Threshold to fuse sequences of states
    Threshold_label = ttk.Label(window, text="Threshold")
    Threshold_label.grid(row=5, column=0, sticky = 'e', padx = padx, pady = pady)
    Threshold_entry = ttk.Entry(window, width=13)
    Threshold_entry.grid(row=5, column=1)
    Threshold_entry.insert(tk.END, str(params['threshold']))
    
    # Maximum number of sequences of states
    Max_nb_sequences_label = ttk.Label(window, text="Maximum number of sequences")
    Max_nb_sequences_label.grid(row=6, column=0, sticky = 'e', padx = padx, pady = pady)
    Max_nb_sequences_entry = ttk.Entry(window, width=13)
    Max_nb_sequences_entry.grid(row=6, column=1)
    Max_nb_sequences_entry.insert(tk.END, str(params['max_nb_sequ_histograms']))
    
    # Depth of field
    Depth_of_field_label = ttk.Label(window, text="Depth of field")
    Depth_of_field_label.grid(row=7, column=0, sticky = 'e', padx = padx, pady = pady)
    Depth_of_field_entry = ttk.Entry(window, width=13)
    Depth_of_field_entry.grid(row=7, column=1)
    Depth_of_field_entry.insert(tk.END, str(params['cell_dims']))
    
    # Savepath Input
    savepath_label = ttk.Label(window, text="Save Path:")
    savepath_label.grid(row=9, column=0, sticky = 'e', padx = padx, pady = pady)
    savepath_entry = ttk.Entry(window, width=50)
    savepath_entry.grid(row=9, column=1)
    savepath_entry.insert(tk.END, os.path.join(savepath, 'saved_tracks_with_position_refinement.csv'))
    savepath_button = ttk.Button(window, text="Browse", command=lambda: browse_savepath(savepath_entry))
    savepath_button.grid(row=9, column=2)
    
    # Run Button
    run_button = ttk.Button(window,
                            text="Start position refinement",
                            command=lambda: run_refinement(window,
                                                          tracks,
                                                          frames,
                                                          opt_metrics,
                                                          dt = float(frametime_entry.get()), 
                                                          nb_states = int(NbStates_entry.get()),
                                                          frame_len = int(windowlength_entry.get()), 
                                                          cell_dims = float(Depth_of_field_entry.get()), 
                                                          LocErr_type = LocErr_type,
                                                          input_LocErr = input_LocErr,
                                                          threshold = float(Threshold_entry.get()), 
                                                          max_nb_states = int(Max_nb_sequences_entry.get()),
                                                          savepath = savepath_entry.get()))
    run_button.grid(row=10, column=1, columnspan=1)
    
    # Previous Button
    previous_button = ttk.Button(window, text="Previous", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=10, column=0, columnspan=1)

def run_refinement(window, tracks, frames, opt_metrics, dt, nb_states, frame_len, cell_dims, LocErr_type, input_LocErr, threshold, max_nb_states, savepath):
    # Run the Brownian motion analysis
    #tracks = tracks[str(length)]
    global params
    
    params['dt'] = dt
    params['labeling_window_length'] = frame_len
    params['cell_dims'] = cell_dims
    params['max_nb_sequ_histograms'] = max_nb_states
    params['threshold'] = threshold
    
    print("params['num_states']", params['num_states'])
    print('nb_states', nb_states)
    
    if params['num_states'] != nb_states:
        get_new_params(nb_states)
    
    if LocErr_type == "Inputing a quality metric for each peak":
        #print('input_LocErr', input_LocErr)
        try:
            for l in input_LocErr:
                input_LocErr[l] = 1/input_LocErr[l]**0.5
        except:
            raise ValueError("If you chose to estimate the localization error from a quality metric, the quality metrics must all be numerical and strictly positive")
    
    lmfit_params = params_to_lmfit_params(params, LocErr_type)
    
    print('lmfit_params', lmfit_params)
    #print('tracks', tracks)
    #print('input_LocErr', input_LocErr)
    print('nb_states', nb_states, type(nb_states))
    
    for l in tracks:
        print(tracks[l].shape)

    if LocErr_type == "Inputing a quality metric for each peak" or LocErr_type == "Inputing the Localization error":
        new_input_LocErr = []
        for l in input_LocErr:
            new_input_LocErr.append(input_LocErr[l])
    else:
        new_input_LocErr = None

    nb_substeps = 1
    LocErr, ds, Fs, TrMat, pBL = extrack.tracking.extract_params(lmfit_params, dt, nb_states, nb_substeps, new_input_LocErr)
    LocErr = LocErr[0]
    
    if LocErr_type == "Inputing a quality metric for each peak" or LocErr_type == "Inputing the Localization error":
        LocErr = input_LocErr
    
    mus, sigs = extrack.refined_localization.position_refinement(tracks,
                                                                 LocErr,
                                                                 ds,
                                                                 Fs,
                                                                 TrMat,
                                                                 frame_len = frame_len,
                                                                 threshold = threshold, 
                                                                 max_nb_states = max_nb_states)
        
    n = 0
    for l in tracks:
        n+= tracks[l].shape[0]*tracks[l].shape[1]
    
    nb_dims = tracks[l].shape[2]
    
    flat_tracks = np.zeros((n, tracks[l].shape[2]))
    flat_frames = np.zeros((n, 1))
    flat_Track_IDs = np.zeros((n, 1))
    flat_opt_metrics = np.zeros((n, len(opt_metrics.keys())))
    flat_refined_values = np.zeros((n, nb_dims+1))
    if LocErr_type == "Inputing a quality metric for each peak" or LocErr_type == "Inputing the Localization error":
        flat_LocErr = np.zeros((n, nb_dims))

    track_ID = 0
    k = 0
    for l in tracks:
        for i, (track, f, m, s) in enumerate(zip(tracks[l], frames[l], mus[l], sigs[l])):
            track_length = track.shape[0]
            flat_tracks[k:k+track_length] = track
            flat_frames[k:k+track_length] = f[:, None]
            flat_Track_IDs[k:k+track_length] = track_ID
            flat_refined_values[k:k+track_length] = np.concatenate(( m, s[:, None]), axis = 1)
            if LocErr_type == "Inputing a quality metric for each peak" or LocErr_type == "Inputing the Localization error":
                flat_LocErr[k:k+track_length] = LocErr[l][i]
            for j, metric in enumerate(opt_metrics):
                flat_opt_metrics[k:k+track_length, j] = opt_metrics[metric][l][i]
            k+=track_length
            track_ID+=1
    
    arr = np.concatenate((flat_tracks, flat_frames, flat_Track_IDs, flat_opt_metrics, flat_refined_values), axis = 1)
    columns = ['POSITION_X', 'POSITION_Y', 'POSITION_Z'][:nb_dims] + ['FRAME', 'TRACK_ID'] + list(opt_metrics.keys()) + ['Refined_position_X', 'Refined_position_Y', 'Refined_position_Z'][:nb_dims] + ['Refined_localization_error']
    
    dataframe = pd.DataFrame(arr, columns = columns)
    dataframe.to_csv(savepath)
    
    print('Position refinement finished and saved at "%s"'%savepath)


'''
predict_Bs(all_tracks,
               dt,
               params,
               cell_dims=[1],
               nb_states=4,
               frame_len=5,
               max_nb_states = 200,
               threshold = 0.1,
               workers = 1,
               input_LocErr = None,
               verbose = 0)
'''
def params_to_lmfit_params(params, LocErr_type):
    print('params[num_states]', params['num_states'], type(params['num_states']))
    
    if LocErr_type == "Fitted parameter":
        LocErr_type = 1        
        slope_offsets_estimates = None
    
    elif LocErr_type == "Inputing the Localization error":
        LocErr_type = None
        slope_offsets_estimates = None
    
    elif LocErr_type == "Inputing a quality metric for each peak":
        LocErr_type = 4
        slope_offsets_estimates = [1, 0.5*params['loc_error'][0]]
    
    mask = (1 - np.identity(params['num_states'])).astype(bool)
    
    lmfit_params = extrack.tracking.generate_params(nb_states = params['num_states'],
                                   LocErr_type = LocErr_type,
                                   nb_dims = 2, # only matters if LocErr_type == 2,
                                   LocErr_bounds = [params['loc_error'][0]/10, params['loc_error'][0]*10], # the initial guess on LocErr will be the geometric mean of the boundaries
                                   D_max = 10, # maximal diffusion coefficient allowed
                                   Fractions_bounds = [0.001, 0.99],
                                   estimated_LocErr = params['loc_error'],
                                   estimated_Ds = params['diff_coeffs'], # D will be arbitrary spaced from 0 to D_max if None, otherwise input 1D array/list of Ds for each state from state 0 to nb_states - 1.
                                   estimated_Fs = params['fractions'], # fractions will be equal if None, otherwise input 1D array/list of fractions for each state from state 0 to nb_states - 1.
                                   estimated_transition_rates = params['transition_probs'][mask], # transition rate per step. [0.1,0.05,0.03,0.07,0.2,0.2]
                                   slope_offsets_estimates = slope_offsets_estimates # need to specify the list [slop, offset] if LocErr_type = 4,
                                   )
    
    return lmfit_params

def lmfit_params_to_params(lmfit_params):
    global params
    lmfit_params
    
    nb_states = params['num_states']
    try:
        params['loc_error'] = np.round(np.array([lmfit_params['LocErr'].value]), 6)
    except:
        1
    params['bleaching_rate'] = lmfit_params['pBL'].value

    diff_coefs = []
    fractions = []
    transition_probabilities = np.zeros((nb_states, nb_states))
    for k in range(nb_states):
        diff_coefs = diff_coefs + [lmfit_params['D%s'%k].value]
        fractions = fractions + [lmfit_params['F%s'%k].value]
        for j in range(nb_states):
            if k!=j:
                transition_probabilities[k, j] = lmfit_params['p%s%s'%(k, j)].value
    transition_probabilities[np.arange(nb_states), np.arange(nb_states)] = np.clip(1-np.sum(transition_probabilities, 1), 1e-10, 1)
    
    params['diff_coeffs'] = diff_coefs
    params['fractions'] = fractions
    params['transition_probs'] = transition_probabilities

def get_new_params(nb_states=3):

    global params

    # Localization error
    loc_error = 0.03
    
    # Bleaching rate
    bleaching_rate = 0.02
    
    # Diffusion coefficients
    diff_coefs = []
    for k in range(nb_states):
        diff_coefs = diff_coefs + [np.round((k/(nb_states-1))**2, 4)]
        
    # Fractions
    fractions = []
    for k in range(nb_states-1):
        fractions = fractions + [np.round(1/nb_states, 3)]
    fractions = fractions + [np.round(1-np.sum(fractions), 3)]
    
    # Transition probabilities (matrix)
    transition_probabilities = np.zeros((nb_states,nb_states))
    for i in range(nb_states):
        for j in range(nb_states):
            if i == j:
                transition_probabilities[i,j] = 0.9
            else:
                transition_probabilities[i,j] = 0.1/(nb_states-1)
    '''
    params = {"num_states": nb_states,
              "loc_error": np.array([loc_error]),
              "diff_coeffs": diff_coefs,
              "fractions": fractions,
              "transition_probs": transition_probabilities,
              "bleaching_rate": bleaching_rate}
    '''
    params["num_states"] = nb_states
    params["loc_error"] = np.array([loc_error])
    params["diff_coeffs"] = diff_coefs
    params["fractions"] = fractions
    params["transition_probs"] = transition_probabilities
    params["bleaching_rate"] = bleaching_rate
    
class ParameterWindow:
    def __init__(self, master, nb_states):
        """
        A popup window that allows the user to edit the parameters:
         - Number of states
         - Localization error
         - Diffusion coefficients
         - Fractions
         - Transition probabilities
         - Bleaching rate
         
        When the user clicks OK, the parameters are passed to 'callback' as a dict.
        """
        self.window = tk.Toplevel(master)
        self.window.title("Parameter Window")
        self.nb_states = nb_states
        
        global params
        
        if nb_states != params['num_states']:
            get_new_params(nb_states)
        
        # Number of states
        ttk.Label(self.window, text="Number of states:").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Label(self.window, text=str(nb_states), anchor = 'center').grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Localization error
        ttk.Label(self.window, text="Localization error:").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.loc_error_entry = ttk.Entry(self.window, width=10)
        self.loc_error_entry.grid(row=1, column=1, padx=5, pady=5)
        self.loc_error_entry.insert(0, str(np.round(params['loc_error'][0], 5)))
        
        # Bleaching rate
        ttk.Label(self.window, text="Bleaching rate:").grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.bleach_entry = ttk.Entry(self.window, width=10)
        self.bleach_entry.grid(row=2, column=1, padx=5, pady=5)
        self.bleach_entry.insert(0, str(np.round(params['bleaching_rate'], 5)))
        
        # Diffusion coefficients
        for k in range(nb_states):
            ttk.Label(self.window, text="State %s"%k).grid(row=4, column=1+k, padx=5, pady=5, sticky="e")
        ttk.Label(self.window, text="Diffusion coefficients:").grid(row=5, column=0, padx=5, pady=5, sticky="ew")
        self.diff_entries = []
        for k in range(nb_states):
            self.diff_entries = self.diff_entries + [ttk.Entry(self.window, width=10)]
            self.diff_entries[k].grid(row=5, column=1+k, padx=5, pady=5)
            self.diff_entries[k].insert(0, str(np.round(params['diff_coeffs'][k], 5)))
        
        # Fractions
        ttk.Label(self.window, text="Fractions:").grid(row=6, column=0, padx=5, pady=5, sticky="ew")
        self.frac_entries = []
        for k in range(nb_states):
            self.frac_entries = self.frac_entries + [ttk.Entry(self.window, width=10)]
            self.frac_entries[k].grid(row=6, column=1+k, padx=5, pady=5)
            self.frac_entries[k].insert(0, str(np.round(params['fractions'][k], 4)))
        
        # Transition probabilities (matrix)
        ttk.Label(self.window, text="Transition probabilities:").grid(row=8, column=0, padx=5, pady=5, sticky="e")
        for k in range(nb_states):
            ttk.Label(self.window, text="to state %s"%k).grid(row=8, column=1+k, padx=5, pady=5)
            ttk.Label(self.window, text="from state %s"%k).grid(row=9+k, column=0, padx=5, pady=5, sticky="e")
        
        self.transition_entries = []
        for i in range(nb_states):
            for j in range(nb_states):
                self.transition_entries = self.transition_entries + [ttk.Entry(self.window, width=10)]
                self.transition_entries[i*nb_states+j].grid(row=9+i, column=1+j, padx=5, pady=5)
                self.transition_entries[i*nb_states+j].insert(0, str(np.round(params['transition_probs'][i,j], 5)))
        
        # OK button to validate & send parameters back
        ttk.Button(self.window, text="OK", command=self.ok_clicked).grid(row=12, column=0, columnspan=3, pady=10)
                
    def ok_clicked(self):
        # Collect parameters
        nb_states = self.nb_states
        global params
        
        diff_coeffs = []
        fractions = []
        transition_probs = np.zeros((nb_states, nb_states))
        for k in range(nb_states):
            diff_coeffs.append(float(self.diff_entries[k].get()))
            fractions.append(float(self.frac_entries[k].get()))
            for j in range(nb_states):
                transition_probs[k, j] = float(self.transition_entries[k*nb_states+j].get())
        
        params["num_states"] = int(self.nb_states)
        params["loc_error"] = np.array([float(self.loc_error_entry.get())])
        params["diff_coeffs"] = diff_coeffs
        params["fractions"] = fractions
        params["transition_probs"] = transition_probs
        params["bleaching_rate"] = float(self.bleach_entry.get())
        # Close this parameter window
        self.window.destroy()

padx = 10
width = 19
# Create the first window
root = tk.Tk()
root.title("Anomalous Analysis Setup")

style = ttk.Style()
style.configure('My.TMenubutton', background='#f3f4f6', foreground='black', borderwidth=1, relief="raised")
style.map('My.TMenubutton', background=[('active', '#e8e9eb'), ('pressed', '#d2d3d5')])
#root["bg"] = "2170e3"
LocErr_type = "Fitted parameter"
LocErr_type = "Inputing a quality metric for each peak"
params = {'num_states': 2,
 'cell_dims': 1.0,
 'dt': 0.1,
 'loc_error': np.array([0.03]),
 'diff_coeffs': [0.0, 1.0],
 'fractions': [0.5, 0.5],
 'transition_probs': np.array([[0.9, 0.1],
                               [0.1, 0.9]]),
 'bleaching_rate': 0.02,
 'dt': 0.1, "fitting_window_length": 6, "labeling_window_length": 10, 'nb_iters': 3, 'max_nb_sequ': 200, 'threshold': 0.1, 'nb_substeps':  1, 'max_nb_sequ_labeling': 50, 'max_nb_sequ_histograms': 300, 'draw_plot': 'Yes'}
nb_states = 2

def browser():
    path_entry.delete(0,'end')
    path_entry.insert(tk.END, filedialog.askopenfilename(initialdir=os.path.expanduser('~'), title="Select File"))

def browse_savepath(entry_widget):
    filename = filedialog.asksaveasfilename(
        initialdir=os.path.expanduser('~'),
        title="Select File",
        filetypes=[("CSV files", "*.csv")],
        defaultextension=".csv"
    )
    if filename:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(tk.END, filename)

# Path Input
path_label = ttk.Label(root, text="Path:")
path_label.grid(row=0, column=0, padx = padx, pady = pady, sticky = 'e')
path_entry = ttk.Entry(root, width=52)
path_entry.grid(row=0, column=1, columnspan=3, padx = padx, pady = pady, sticky = 'e')
#path_entry.insert(tk.END, os.getcwd())
path_entry.insert(tk.END,  r'D:\Maria_DATA\Tracks\4.csv')
#path_button = ttk.Button(root, text="Browse", command=lambda: path_entry.insert(tk.END, filedialog.askopenfilename()))
#path_button = ttk.Button(root, text="Browse", command=lambda: (path_entry.insert(tk.END, filedialog.askopenfilename(initialdir=os.path.expanduser('~'), title="Select File"))))
path_button = ttk.Button(root, text="Browse", command=browser)
path_button.grid(row=0, column=4, padx = padx, pady = pady, sticky = 'e')

# minimum Length Input
min_length_label = ttk.Label(root, text="Minimum length:")
min_length_label.grid(row=1, column=0, padx = padx, pady = pady, sticky = 'e')
min_length_entry = ttk.Entry(root, width=width)
min_length_entry.grid(row=1, column=1, padx = padx, pady = pady, sticky = 'e')
min_length_entry.insert(tk.END, "5")

# Maximum Length Input
max_length_label = ttk.Label(root, text="Maximum length:")
max_length_label.grid(row=1, column=2, padx = padx, pady = pady, sticky = 'e')
max_length_entry = ttk.Entry(root, width=width)
max_length_entry.grid(row=1, column=3, padx = padx, pady = pady, sticky = 'e')
max_length_entry.insert(tk.END, "5")

headers_label = ttk.Label(root, text="Headers:")
headers_label.grid(row=3, column=0, padx = padx, pady = pady, sticky = 'e')

x_pos_label = ttk.Label(root, text="x", width=10)
x_pos_label.grid(row=2, column=1, padx = padx, pady = pady, sticky = 'e')
x_pos_entry = ttk.Entry(root, width=width)
x_pos_entry.grid(row=3, column=1, padx = padx, pady = pady, sticky = 'e')
x_pos_entry.insert(tk.END, "POSITION_X")

y_pos_label = ttk.Label(root, text="y", width=10)
y_pos_label.grid(row=2, column=2, padx = padx, pady = pady, sticky = 'e')
y_pos_entry = ttk.Entry(root, width=width)
y_pos_entry.grid(row=3, column=2, padx = padx, pady = pady, sticky = 'e')
y_pos_entry.insert(tk.END, "POSITION_Y")

frame_label = ttk.Label(root, text="frame", width=12)
frame_label.grid(row=2, column=3, padx = padx, pady = pady, sticky = 'e')
frame_entry = ttk.Entry(root, width=width)
frame_entry.grid(row=3, column=3, padx = padx, pady = pady, sticky = 'e')
frame_entry.insert(tk.END, "FRAME")

ID_label = ttk.Label(root, text="Track ID", width=13)
ID_label.grid(row=2, column=4, padx = padx, pady = pady, sticky = 'e')
ID_entry = ttk.Entry(root, width=width)
ID_entry.grid(row=3, column=4, padx = padx, pady = pady, sticky = 'e')
ID_entry.insert(tk.END, "TRACK_ID")

# Analysis Type Input
analysis_type_label = ttk.Label(root, text="Analysis Type:")
analysis_type_label.grid(row=4, column=0, padx = padx, pady = pady, sticky = 'e')
analysis_type_var = tk.StringVar(root)
analysis_type_var.set("Model Fitting")
analysis_type_dropdown = ttk.OptionMenu(root, analysis_type_var, analysis_type_var.get(),
                                        "Model Fitting",
                                        "State Labeling",
                                        "State Lifetime Histogram",
                                        "Position Refinement",
                                        style='My.TMenubutton')
analysis_type_dropdown.config(width=32)
analysis_type_dropdown.grid(row=4, column=1, columnspan=2, padx = padx, pady = pady, sticky="e")

LocErr_type_label = ttk.Label(root, text="Type of localization error")
LocErr_type_label.grid(row=5, column=0, padx = padx, pady = pady, sticky = 'e')
LocErr_type_var = tk.StringVar(root)
LocErr_type_var.set("Fitted parameter")
LocErr_type_dropdown = ttk.OptionMenu(root, LocErr_type_var, LocErr_type_var.get(),
                                         "Fitted parameter",
                                         "Inputing the Localization error",
                                         "Inputing a quality metric for each peak", #  must verify LocErr = a/quality + b
                                         style='My.TMenubutton')
LocErr_type_dropdown.config(width=32)
LocErr_type_dropdown.grid(row=5, column=1, columnspan=2, padx = padx, pady = pady, sticky="e")

LocErr_input_entry = ttk.Entry(root, width=41)
LocErr_input_entry.grid(row=5, column=3, columnspan=2, padx = padx, pady = pady, sticky = 'e')
LocErr_input_entry.insert(tk.END, "QUALITY")

LocErr_type_label = ttk.Label(root, text="Additional metrics")
LocErr_type_label.grid(row=6, column=0, padx = padx, pady = pady, sticky = 'e')
Optional_input_entry = ttk.Entry(root, width=88)
Optional_input_entry.grid(row=6, column=1, columnspan=4, padx = padx, pady = pady, sticky = 'e')
Optional_input_entry.insert(tk.END, "CONTRAST_CH1,SNR_CH1")

max_dist_label = ttk.Label(root, text="Maximum distance")
max_dist_label.grid(row=7, column=0, padx = padx, pady = pady, sticky = 'e')
max_dist_entry = ttk.Entry(root, width=width)
max_dist_entry.grid(row=7, column=1, padx = padx, pady = pady, sticky = 'e')
max_dist_entry.insert(tk.END, '1.')

remove_no_disp_label = ttk.Label(root, text="Remove no displacements")
remove_no_disp_label.grid(row=7, column=2, columnspan=2, padx = padx, pady = pady, sticky = 'e')
remove_no_disp_entry = ttk.Entry(root, width=width)
remove_no_disp_entry.grid(row=7, column=4, padx = padx, pady = pady, sticky = 'e')
remove_no_disp_entry.insert(tk.END, 'True')

# Next Button
next_button = ttk.Button(root, text="Next", command=open_analysis_window, width=31)
next_button.grid(row=8, column=3, columnspan=2, padx = padx, pady = pady, sticky = 'e')

root.mainloop()

import numpy; print(numpy.__file__)
