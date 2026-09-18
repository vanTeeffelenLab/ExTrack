# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:15:37 2025
@author: Franc
This code enables to use the Graphical User interface of ExTrack. 
To create a stand alone version of ExTrack:
1) pip install pyinstaller
2) pyinstaller --onedir path\ExTrack_GUI.py
3) Copy the .ddl files starting with mkl into the dist\ExTrack_GUI\_internal (the mkl files can be found in C:\ Users\Franc\anaconda3\Library\bin in my case)
4) execute dist\ExTrack_GUI.exe to run the stand alone software
test commit
"""

import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
print('tkinter',tk)

from tkinter import ttk
import webbrowser
import copy
import extrack

import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

#ttk = tk.ttk

padx = 10 # spacing between cells of the grid in x
pady = 10 # spacing between cells of the grid in y
previous_window = None

# Small persistent settings (currently the last dataset path), kept in the user
# folder so they survive from one instance of the GUI to the next. The location
# can be overridden with the EXTRACK_GUI_CONFIG environment variable (used by
# the tests to stay away from the real file).
import json
CONFIG_PATH = os.environ.get('EXTRACK_GUI_CONFIG',
                             os.path.join(os.path.expanduser('~'), '.extrack_gui.json'))

def load_gui_config():
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            config = json.load(f)
        return config if type(config) == dict else {}
    except Exception: # no config yet, or an unreadable one: start fresh
        return {}

def save_gui_config(**updates):
    config = load_gui_config()
    config.update(updates)
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    except Exception as error: # never let a settings write break the GUI
        print('Could not save the GUI settings to %s: %s'%(CONFIG_PATH, error))

def initialdir_from(current_path):
    """the folder containing the current path, for the Browse dialogs to start in"""
    current_path = current_path.strip()
    if os.path.isdir(current_path):
        return current_path
    parent = os.path.dirname(current_path)
    if os.path.isdir(parent):
        return parent
    return os.path.expanduser('~')

def default_save_folder(path):
    """
    Where the analyses propose to write their results: the PARENT of the folder
    holding the dataset. A batch reads a folder of replicates, so writing next
    to them would mix results into the data; one level up keeps the two apart
    and gathers the replicates of one experiment in a single place. Falls back
    to the dataset folder when there is no usable parent (a drive root).
    """
    path = os.path.normpath(path.strip())
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    parent = os.path.dirname(folder)
    if parent and parent != folder and os.path.isdir(parent):
        return parent
    if os.path.isdir(folder):
        return folder
    return os.path.expanduser('~')

def open_analysis_window():
    global previous_window
    path = path_entry.get()
    save_gui_config(last_path=path) # remembered for the next instance of the GUI
    print(os.path.normpath(path))
    savepath = default_save_folder(path)

    min_length = int(min_length_entry.get())
    max_length = int(max_length_entry.get())
    analysis_type = analysis_type_var.get()
    LocErr_type = LocErr_type_var.get()
    LocErr_input_name = LocErr_input_entry.get().split(',')
    if LocErr_input_name == ['']:
        LocErr_input_name = []
    Optional_input_name = Optional_input_entry.get().split(',')
    if Optional_input_name == ['']:
        Optional_input_name = []
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
    elif analysis_type in BATCH_STAGES:
        create_batch_window(analysis_window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps, analysis_type)

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

def equilibrium_fractions(transition_probs, nb_substeps = 1):
    """
    Steady-state occupancy of each state implied by the transition probabilities,
    using the exact convention of the model (extrack.tracking.extract_params,
    Matrix_type = 1): per-substep probabilities 1 - exp(-rate/nb_substeps) off
    the diagonal, the remainder on it. Solved exactly as the left eigenvector of
    the transition matrix, so arbitrarily slow rates converge too.
    """
    nb_states = len(transition_probs)
    TrMat = 1 - np.exp(-np.asarray(transition_probs, dtype = float) / nb_substeps)
    TrMat[np.arange(nb_states), np.arange(nb_states)] = 0
    TrMat[np.arange(nb_states), np.arange(nb_states)] = 1 - np.sum(TrMat, 1)
    M = TrMat.T - np.identity(nb_states)
    M[-1] = 1                                # replaces one redundant equation by sum = 1
    b = np.zeros(nb_states)
    b[-1] = 1
    return np.linalg.solve(M, b)

def load_dataset_or_error(window, path, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps):
    """
    Read the dataset behind the four analysis windows. Any problem -- a path that
    is not set correctly, a file that does not read, no track passing the length
    filters -- raises an error panel (show_error_url) instead of a console
    traceback, and returns None so the caller can simply return.
    Returns (tracks, frames, opt_metrics, input_LocErr) on success.
    """
    if os.path.isdir(path):
        path = glob(path + '/*.csv')
        if len(path) == 0:
            show_error_url(window, "No csv file detected in the informed directory. Make sure the csv files end with the extention '.csv'.\n", url=None)
            return None
    elif not os.path.exists(path):
        show_error_url(window, "The informed dataset path does not exist:\n%s\nPlease verify the Path field of the previous window.\n"%path, url=None)
        return None
    elif not path.endswith('.csv'):
        show_error_url(window, "Please select a csv file with an extention '.csv'.\n", url=None)
        return None
    try:
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
            if LocErr_input_name == []:
                raise ValueError('If selecting localization errors "Inputing the Localization error" or "Inputing a quality metric for each peak", you must provide the name of the column that informs on the localization error of each peaks in the 3rd column of the "Type of localization error" row, exemple: QUALITY.')
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
        if sum(len(tracks[l]) for l in tracks) == 0:
            raise ValueError('No track was loaded: verify the headers and the minimum/maximum track lengths.')
    except Exception as error:
        show_error_url(window, "The dataset could not be read correctly.\nVerify that the headers for the x positions, y positions, frame number and track ID are correctly informed, as well as the minimum and maximum track lengths. If selecting localization errors \"Inputing the Localization error\" or \"Inputing a quality metric for each peak\",  you must provide the name of the column that informs on the localization error of each peaks in the 3rd column of the 'Type of localization error' row, exemple: QUALITY.\n\nError: %s\n"%error, url=None)
        return None
    return tracks, frames, opt_metrics, input_LocErr

# Explanations shown by the '?' cells next to each hyperparameter.
HYPERPARAM_INFO = {
    'nb_states': (
        "Number of motion states of the model (e.g. 2 for an immobile and a mobile "
        "state). Each state has its own diffusion coefficient and fraction, with "
        "transition probabilities between states, all set in the Parameter Window. "
        "The computation time increases steeply with the number of states (see the "
        "fusion model box)."),
    'dt': (
        "Time between two consecutive frames of the movie, in seconds. It converts "
        "the fitted diffusion coefficients (um2/s) and the per-step transition "
        "probabilities into physical units."),
    'window_length': (
        "Number of consecutive time points over which the sequences of states are "
        "treated exactly. Longer windows are more accurate but more costly: the "
        "multi-transition model scales as nb_states ** window_length, the "
        "mono-transition model as window_length * nb_states**2."),
    'nb_substeps': (
        "Number of transition steps considered between two consecutive positions. "
        "1 assumes at most one transition per frame; higher values refine the "
        "timing of the transitions at a multiplied cost. The mono-transition "
        "fusion model requires 1."),
    'threshold': (
        "Sequences of states whose means and standard deviations differ by less "
        "than this fraction of sigma are fused (multi-transition model only). "
        "Lower values are more accurate but keep more sequences; it is increased "
        "by 20% automatically whenever the number of sequences exceeds the "
        "maximum."),
    'max_nb_sequences': (
        "Cap on the number of sequences of states kept per track (multi-transition "
        "model only). When the cap is exceeded, the fusion threshold is increased "
        "by 20% until the number of sequences fits. Lower values are faster but "
        "coarser."),
    'depth_of_field': (
        "Depth of the observable volume in micrometers (e.g. ~0.3 for TIRF, ~0.8 "
        "for HILO). It models the probability that a particle leaves the field of "
        "view, which corrects the bias of the observed tracks towards slow "
        "particles."),
    'nb_iters': (
        "Number of times the fit is repeated, each round restarting from the "
        "previous optimum (first round with the Powell method, later rounds with "
        "BFGS). More iterations improve convergence at a proportional cost."),
    'draw_plot': (
        "If Yes, a plot of the results is displayed at the end of the analysis."),
    'initial_fractions': (
        "Fraction of the particles in each state at the FIRST time point of the "
        "tracks. This is the initial occupancy, not the steady state: along the "
        "tracks the occupancies relax towards the equilibrium fractions set by "
        "the transition probabilities."),
    'batch_files': (
        "Every listed file is processed independently, each starting from the "
        "parameters of the Parameter Window: fitting first (its fitted "
        "parameters feed the later stages of that same file), then depending on "
        "the chosen analysis: state labeling, lifetime histograms and position "
        "refinement. One csv per stage is written in the save folder, prefixed "
        "by the input file name. A file that fails is reported and the batch "
        "continues with the next one. Plots are disabled in batch mode, the "
        "labeling/histogram sequence caps reuse the values of the single-file "
        "windows, and the global parameters are restored at the end of the "
        "batch."),
    'equilibrium_fractions': (
        "Steady-state occupancy of each state implied by the transition "
        "probabilities below (read-only, updated live as they are edited). If "
        "the tracking starts at steady state, the initial fractions should be "
        "close to these values."),
}

def add_param_info(window, row, key, column=2):
    """
    A small '?' cell to the right of a hyperparameter value. Clicking it expands
    the explanation from HYPERPARAM_INFO next to it; clicking again collapses it.
    """
    info_label = tk.Label(window, text=HYPERPARAM_INFO[key], justify='left',
                          wraplength=340, relief='groove', borderwidth=1,
                          padx=6, pady=4, bg='#f3f4f6')
    expanded = {'on': False}
    def toggle():
        if expanded['on']:
            info_label.grid_remove()
        else:
            info_label.grid(row=row, column=column+1, padx=padx, pady=2, sticky='w')
        expanded['on'] = not expanded['on']
    info_button = ttk.Button(window, text='?', width=2, command=toggle)
    info_button.grid(row=row, column=column, sticky='w', padx=2)
    return info_button

def show_progress_window(window, message):
    """
    Transient window shown while an analysis runs ('Fitting on-going...'). The
    analyses run in the interface's own thread, so the window is drawn once
    (update) before the computation starts and refreshed at the end by
    finish_progress_window.
    """
    progress_window = tk.Toplevel(window)
    progress_window.title("ExTrack")
    label = tk.Label(progress_window, text=message, wraplength=380,
                     justify='center', padx=25, pady=20)
    label.pack(expand=True)
    progress_window.lift()
    progress_window.update()
    return progress_window, label

def finish_progress_window(progress_window, label, message):
    """Turn the on-going window into a completion message with an OK button."""
    label.config(text=message)
    ok_button = ttk.Button(progress_window, text="OK",
                           command=progress_window.destroy)
    ok_button.pack(pady=(0, 12))
    progress_window.lift()
    progress_window.update()

# Name of the file gathering the fitting results of a whole batch, written in
# the save folder next to the per-replicate outputs.
BATCH_SUMMARY_NAME = 'batch_fitting_summary.csv'

# A batch writes several files per replicate, so it gets a folder of its own
# rather than dropping them straight into the parent of the dataset folder: one
# experiment can then hold several dataset folders without their outputs mixing.
BATCH_RESULTS_DIRNAME = 'Results'

def batch_save_folder(savepath):
    """The save folder a batch proposes: a Results directory inside `savepath`,
    which default_save_folder has already set to the parent of the folder the
    datasets are read from. Created by run_batch, not here, so that merely
    opening the batch window leaves no folder behind."""
    return os.path.join(savepath, BATCH_RESULTS_DIRNAME)

# Suffix of the one track file a batch writes per replicate: the state
# predictions and the refined positions in the same table, so that a replicate
# leaves one track file behind instead of one per stage.
BATCH_TRACKS_SUFFIX = '_tracks.csv'

def merge_track_tables(labeled, refined):
    """
    One track table out of the labeling and the refinement outputs: the state
    predictions beside the refined positions and their localization error, with
    the positions, frames, track IDs and optional metrics they have in common
    written once. Either side may be None -- the batch mode did not run that
    stage -- and the other is then returned unchanged.

    Both tables are built by flattening the same tracks dictionary in the same
    order, so their rows correspond one to one. That is verified on FRAME and
    TRACK_ID rather than assumed, with a merge on those two keys as the fall
    back if they ever stop lining up.
    """
    if labeled is None or refined is None:
        return refined if labeled is None else labeled
    keys = [k for k in ['TRACK_ID', 'FRAME'] if k in labeled.columns and k in refined.columns]
    if len(keys) != 2:
        raise ValueError('the labeling and refinement tables share no TRACK_ID/FRAME columns to merge on: %s vs %s'
                         %(list(labeled.columns), list(refined.columns)))
    extra = [c for c in refined.columns if c not in labeled.columns] # the Refined_* columns
    lined_up = len(labeled) == len(refined) and all(
        np.array_equal(labeled[k].values.astype(float), refined[k].values.astype(float)) for k in keys)
    if lined_up:
        merged = labeled.copy()
        for c in extra:
            merged[c] = refined[c].values
        return merged
    print('Batch: the labeling and refinement rows do not line up; merging on %s'%keys)
    right = refined[keys + extra].copy()
    for k in keys: # the refinement table carries the keys as floats, the labeling one as ints
        right[k] = right[k].astype(labeled[k].dtype)
    return labeled.merge(right, on = keys, how = 'left')

def collect_fitting_summary(fitted, save_folder, filename = BATCH_SUMMARY_NAME):
    """
    Gather the one-row fitting result of every replicate of a batch into a single
    table and write it to the save folder. `fitted` is the list of (file name,
    one-row dataframe returned by _run_fitting_core) of the files that fitted.
    This file REPLACES the per-replicate fitting csv, which a batch no longer
    writes. Returns (DataFrame, path written), or (None, None) if nothing fitted.
    """
    rows = []
    for name, row in fitted:
        if row is None or len(row) == 0:
            print('Batch: no fitting result to summarise for %s'%name)
            continue
        row = row.drop(columns = ['exp'], errors = 'ignore') # the single-file savepath
        row.insert(0, 'dataset', name)
        rows.append(row)
    if len(rows) == 0:
        return None, None
    summary = pd.concat(rows, ignore_index = True)
    path = os.path.join(save_folder, filename)
    try:
        summary.to_csv(path, index = False)
    except Exception as error:
        print('Batch: could not write the fitting summary: %s'%error)
        return summary, None
    return summary, path

def format_summary_cell(value):
    """Table cells: 4 significant digits for the fitted values, text as it comes."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return str(value)
    return '%.4g'%value

def show_fitting_summary(window, summary, max_rows = 12):
    """
    The fitted parameters of every replicate, as a table inside the window that
    announces the end of the batch. It scrolls in both directions: a fit of n
    states carries n**2 + 3n + 3 columns, and a batch has one row per file it
    could read.
    """
    frame = ttk.Frame(window)
    frame.pack(fill = 'both', expand = True, padx = 10, pady = (0, 10))
    columns = [str(c) for c in summary.columns]
    tree = ttk.Treeview(frame, columns = columns, show = 'headings',
                        height = min(max_rows, max(1, len(summary))))
    cells = [[format_summary_cell(v) for v in row] for row in summary.values]
    for i, col in enumerate(columns):
        tree.heading(col, text = col)
        widest = max([len(col)] + [len(row[i]) for row in cells])
        tree.column(col, width = min(200, max(60, 8 * widest + 16)),
                    anchor = 'w' if col == 'dataset' else 'center', stretch = False)
    for row in cells:
        tree.insert('', 'end', values = row)
    vsb = ttk.Scrollbar(frame, orient = 'vertical', command = tree.yview)
    hsb = ttk.Scrollbar(frame, orient = 'horizontal', command = tree.xview)
    tree.configure(yscrollcommand = vsb.set, xscrollcommand = hsb.set)
    tree.grid(row = 0, column = 0, sticky = 'nsew')
    vsb.grid(row = 0, column = 1, sticky = 'ns')
    hsb.grid(row = 1, column = 0, sticky = 'ew')
    frame.rowconfigure(0, weight = 1)
    frame.columnconfigure(0, weight = 1)
    # the window was already realised around the message alone: let it grow to
    # the table rather than clipping its last row
    window.update_idletasks()
    window.geometry('')
    return tree

# The stages each batch analysis runs on every file of the folder, in order.
# Fitting always comes first: its fitted parameters feed the later stages.
BATCH_STAGES = {"Batch Fitting": ['fitting'],
                "Batch Fitting + Labeling": ['fitting', 'labeling'],
                "Batch All": ['fitting', 'labeling', 'histogram', 'refinement']}

def read_dataset_file(path, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps):
    """
    Read one dataset file, csv (read_table, using the informed headers) or
    TrackMate xml (read_trackmate_xml). Raises on any problem; the batch loop
    catches per file. Returns (tracks, frames, opt_metrics, input_LocErr).
    """
    per_peak = LocErr_type != "Fitted parameter"
    if per_peak and LocErr_input_name == []:
        raise ValueError('If selecting localization errors "Inputing the Localization error" or "Inputing a quality metric for each peak", you must provide the name of the column that informs on the localization error of each peaks')
    opt_names = Optional_input_name + (LocErr_input_name if per_peak else [])
    if path.endswith('.csv'):
        tracks, frames, opt_metrics = extrack.readers.read_table(path,
                                                         lengths=np.arange(min_length, max_length+1),
                                                         dist_th=max_dist,
                                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                                         colnames = headers,
                                                         remove_no_disp=remove_no_disps,
                                                         opt_colnames = opt_names)
    else:
        tracks, frames, opt_metrics = extrack.readers.read_trackmate_xml(path,
                                                         lengths=np.arange(min_length, max_length+1),
                                                         dist_th=max_dist,
                                                         frames_boundaries=[-np.inf, np.inf],
                                                         remove_no_disp=remove_no_disps,
                                                         opt_metrics_names = opt_names,
                                                         opt_metrics_types = ['float64']*len(opt_names))
    input_LocErr = None
    if per_peak:
        input_LocErr = {}
        for l in tracks:
            input_LocErr[l] = np.zeros(tracks[l].shape[:2] + (len(LocErr_input_name),))
        for i, name in enumerate(LocErr_input_name):
            for l in tracks:
                input_LocErr[l][:,:,i] = opt_metrics[name][l]
            del opt_metrics[name]
    if sum(len(tracks[l]) for l in tracks) == 0:
        raise ValueError('No track was loaded: verify the headers and the minimum/maximum track lengths.')
    return tracks, frames, opt_metrics, input_LocErr

def browse_savefolder(entry_widget):
    folder = filedialog.askdirectory(initialdir=initialdir_from(entry_widget.get()), title="Select Folder")
    if folder:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(tk.END, folder)

# The two ways ExTrack can fuse the sequences of states, mapped to the
# `sequence_scheme` argument of extrack.tracking.param_fitting / predict_Bs.
FUSION_MODELS = {"Multi-transition": "sequences",
                 "Mono-transition": "ages"}

fusion_model_info = (
    "Multi-transition: every sequence of states within the window is considered, "
    "so several transitions per window can be resolved. More accurate, but it "
    "scales poorly with the number of states: time proportional to "
    "nb_states ** window_length.\n"
    "Mono-transition: only the time since the last transition is kept. Time "
    "proportional to window_length * nb_states**2, so it stays fast with many "
    "states or long windows, at the cost of a coarser approximation.")
HYPERPARAM_INFO['fusion_model'] = fusion_model_info

def add_fusion_model_selector(window, row, default):
    """
    Dropdown to pick the fusion model, with a '?' cell on the same row that
    expands the multi- vs mono-transition trade-off, like the other
    hyperparameters. Returns the tk.StringVar holding the selection;
    FUSION_MODELS maps it to the sequence_scheme argument of
    param_fitting / predict_Bs.
    """
    fusion_label = ttk.Label(window, text="Fusion model")
    fusion_label.grid(row=row, column=0, sticky = 'e', padx = padx, pady = pady)
    fusion_var = tk.StringVar(window)
    fusion_var.set(default)
    fusion_dropdown = ttk.OptionMenu(window, fusion_var, fusion_var.get(),
                                     *FUSION_MODELS.keys(),
                                     style='My.TMenubutton')
    fusion_dropdown.config(width=15)
    fusion_dropdown.grid(row=row, column=1)
    add_param_info(window, row, 'fusion_model')
    return fusion_var

def create_fitting_window(window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps):

    #try:
    print('path', path, type(path))
    print('headers', headers, type(headers), type(headers[0])) 
    print('remove_no_disp', remove_no_disps, type(remove_no_disps))
    print('Optional_input_name', Optional_input_name, type(Optional_input_name))
    print('max_dist', max_dist, type(max_dist))
    
    loaded = load_dataset_or_error(window, path, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps)
    if loaded is None: # an error panel explains what went wrong
        return
    tracks, frames, opt_metrics, input_LocErr = loaded 
    
    global params

    # Initial number of states
    NbStates_label = ttk.Label(window, text="Number of states:")
    NbStates_label.grid(row=0, column=0, sticky = 'e', padx = padx, pady = pady)
    NbStates_entry = ttk.Entry(window, width=13)
    NbStates_entry.grid(row=0, column=1)
    NbStates_entry.insert(tk.END, str(params['num_states']))
    add_param_info(window, 0, 'nb_states')
    
    open_button = ttk.Button(window, text="Open Parameter Window", command=lambda: ParameterWindow(window, int(NbStates_entry.get())))
                             #ParameterWindow(window, int(NbStates_entry.get())))
    open_button.grid(row=1, column=0, sticky = 'e', padx = padx, pady = pady)
    
    # Frame time
    frametime_label = ttk.Label(window, text="Frame time (in s)")
    frametime_label.grid(row=2, column=0, sticky = 'e', padx = padx, pady = pady)
    frametime_entry = ttk.Entry(window, width=13)
    frametime_entry.grid(row=2, column=1)
    frametime_entry.insert(tk.END, str(params['dt']))
    add_param_info(window, 2, 'dt')
    
    # Window length
    windowlength_label = ttk.Label(window, text="Window length")
    windowlength_label.grid(row=3, column=0, sticky = 'e', padx = padx, pady = pady)
    windowlength_entry = ttk.Entry(window, width=13)
    windowlength_entry.grid(row=3, column=1)
    windowlength_entry.insert(tk.END, str(params['fitting_window_length']))
    add_param_info(window, 3, 'window_length')
    
    # Number of substeps
    nb_substeps_label = ttk.Label(window, text="Number of substeps")
    nb_substeps_label.grid(row=4, column=0, sticky = 'e', padx = padx, pady = pady)
    nb_substeps_entry = ttk.Entry(window, width=13)
    nb_substeps_entry.grid(row=4, column=1)
    nb_substeps_entry.insert(tk.END, str(params['nb_substeps']))
    add_param_info(window, 4, 'nb_substeps')
    
    # Threshold to fuse sequences of states
    Threshold_label = ttk.Label(window, text="Threshold")
    Threshold_label.grid(row=5, column=0, sticky = 'e', padx = padx, pady = pady)
    Threshold_entry = ttk.Entry(window, width=13)
    Threshold_entry.grid(row=5, column=1)
    Threshold_entry.insert(tk.END, str(params['threshold']))
    add_param_info(window, 5, 'threshold')
    
    # Maximum number of states
    Max_nb_sequences_label = ttk.Label(window, text="Maximum number of sequences")
    Max_nb_sequences_label.grid(row=6, column=0, sticky = 'e', padx = padx, pady = pady)
    Max_nb_sequences_entry = ttk.Entry(window, width=13)
    Max_nb_sequences_entry.grid(row=6, column=1)
    Max_nb_sequences_entry.insert(tk.END, str(params['max_nb_sequ']))
    add_param_info(window, 6, 'max_nb_sequences')
    
    # Depth of field
    Depth_of_field_label = ttk.Label(window, text="Depth of field")
    Depth_of_field_label.grid(row=7, column=0, sticky = 'e', padx = padx, pady = pady)
    Depth_of_field_entry = ttk.Entry(window, width=13)
    Depth_of_field_entry.grid(row=7, column=1)
    Depth_of_field_entry.insert(tk.END, str(params['cell_dims']))
    add_param_info(window, 7, 'depth_of_field')
    
    # number of iterations of the fitting methods
    nb_iter_label = ttk.Label(window, text="Number of iterations")
    nb_iter_label.grid(row=8, column=0, sticky = 'e', padx = padx, pady = pady)
    nb_iter_entry = ttk.Entry(window, width=13)
    nb_iter_entry.grid(row=8, column=1)
    nb_iter_entry.insert(tk.END, str(params['nb_iters']))
    add_param_info(window, 8, 'nb_iters')
    
    # Fusion model (multi- vs mono-transition) and its explanation box
    fusion_model_var = add_fusion_model_selector(window, 9, params['fusion_model'])

    # Savepath Input
    savepath_label = ttk.Label(window, text="Save Path:")
    savepath_label.grid(row=11, column=0, sticky = 'e', padx = padx, pady = pady)
    savepath_entry = ttk.Entry(window, width=50)
    savepath_entry.grid(row=11, column=1)
    savepath_entry.insert(tk.END, os.path.join(savepath, 'saved_fitting_results.csv'))
    savepath_button = ttk.Button(window, text="Browse", command=lambda: browse_savepath(savepath_entry))
    savepath_button.grid(row=11, column=2)
    
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
                                                                                     savepath = savepath_entry.get(),
                                                                                     fusion_model = fusion_model_var.get()))
    run_button.grid(row=12, column=1, columnspan=1)

    # Previous Button
    previous_button = ttk.Button(window, text="Other analyses", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=12, column=0, columnspan=1)

def run_fitting(window, tracks, dt, nb_states, nb_iterations, nb_substeps, frame_len, cell_dims, LocErr_type, input_LocErr, threshold, max_nb_states, savepath, fusion_model = 'Multi-transition'):
    progress_window, progress_label = show_progress_window(window, "Fitting on-going...")
    try:
        result = _run_fitting_core(window, tracks, dt, nb_states, nb_iterations, nb_substeps, frame_len, cell_dims, LocErr_type, input_LocErr, threshold, max_nb_states, savepath, fusion_model)
    except Exception as error:
        finish_progress_window(progress_window, progress_label, "Fitting failed:\n%s"%error)
        raise
    if result is None: # the analysis was refused before starting; an error window explains why
        progress_window.destroy()
        return
    finish_progress_window(progress_window, progress_label, "Fitting finished.\nResults saved to:\n%s"%savepath)

def _run_fitting_core(window, tracks, dt, nb_states, nb_iterations, nb_substeps, frame_len, cell_dims, LocErr_type, input_LocErr, threshold, max_nb_states, savepath, fusion_model = 'Multi-transition'):
    # Run the Brownian motion analysis
    #tracks = tracks[str(length)]
    global params

    fusion_scheme = FUSION_MODELS[fusion_model]
    if fusion_scheme == 'ages' and nb_substeps != 1:
        show_error_url(window, "The mono-transition fusion model does not support substeps.\nSet 'Number of substeps' to 1 or select the multi-transition model.", url=None)
        return
    params['fusion_model'] = fusion_model

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
                                      verbose = 0,
                                      workers = 1,
                                      Matrix_type = 1,
                                      method = 'powell',
                                      steady_state = False,
                                      cell_dims = [cell_dims], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                                      input_LocErr = input_LocErr,
                                      threshold = threshold,
                                      max_nb_states = max_nb_states,
                                      sequence_scheme = fusion_scheme)
    print('likelihood iteration 0:', - model_fit.residual[0])
    for k in range(nb_iterations-1):
        model_fit = extrack.tracking.param_fitting(tracks,
                                          dt,
                                          params = model_fit.params,
                                          nb_states = nb_states,
                                          nb_substeps = nb_substeps,
                                          frame_len = frame_len,
                                          verbose = 0,
                                          workers = 1,
                                          Matrix_type = 1,
                                          method = 'bfgs',
                                          steady_state = False,
                                          cell_dims = [cell_dims], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                                          input_LocErr = input_LocErr,
                                          threshold = threshold,
                                          max_nb_states = max_nb_states,
                                          sequence_scheme = fusion_scheme)
        print('likelihood iteration %s:'%(k+1), - model_fit.residual[0])
    lmfit_params = model_fit.params
    
    TrMat = np.zeros((nb_states, nb_states))
    for i in range(nb_states):
        for j in range(nb_states):
            if i!=j:
                TrMat[i,j] = model_fit.params['p%s%s'%(i,j)].value/100
        TrMat[i,i] = 1-np.sum(TrMat[i])
    
    A0 = np.ones((1,nb_states))/nb_states
    for k in range(200000):
        A0 = A0 @ TrMat
    
    equilibrium_Fraction_names = []
    for s in range(nb_states):
        equilibrium_Fraction_names.append('equilibrium_F%s'%s)
    
    data = pd.DataFrame([], columns = ['exp', 'likelihood'] + list(lmfit_params.keys()) + equilibrium_Fraction_names)

    vals = [savepath, - model_fit.residual[0]]
    for param in lmfit_params:
        vals.append(lmfit_params[param].value)
    
    for Fi in A0[0]:
        vals.append(Fi)    
    
    data.loc[len(data.index)] = vals
    if savepath is not None: # only the batch passes None; its replicates share one summary file
        data.to_csv(savepath)
    
    lmfit_params_to_params(lmfit_params)
    
    print("Fitting analysis completed%s"%(" and results saved to %s"%savepath if savepath is not None else ""))
    print(data)
    return data


def create_prediction_window(window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps):
    
    loaded = load_dataset_or_error(window, path, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps)
    if loaded is None: # an error panel explains what went wrong
        return
    tracks, frames, opt_metrics, input_LocErr = loaded
        
    global params
    
    # Initial number of states
    NbStates_label = ttk.Label(window, text="Number of states:")
    NbStates_label.grid(row=0, column=0, sticky = 'e', padx = padx, pady = pady)
    NbStates_entry = ttk.Entry(window, width=13)
    NbStates_entry.grid(row=0, column=1)
    NbStates_entry.insert(tk.END, str(params['num_states']))
    add_param_info(window, 0, 'nb_states')
    
    open_button = ttk.Button(window, text="Open Parameter Window", command=lambda: ParameterWindow(window, int(NbStates_entry.get())))
                             #ParameterWindow(window, int(NbStates_entry.get())))
    open_button.grid(row=1, column=0, sticky = 'e', padx = padx, pady = pady)
    
    # Frame time
    frametime_label = ttk.Label(window, text="Frame time (in s)")
    frametime_label.grid(row=2, column=0, sticky = 'e', padx = padx, pady = pady)
    frametime_entry = ttk.Entry(window, width=13)
    frametime_entry.grid(row=2, column=1)
    frametime_entry.insert(tk.END, str(params['dt']))
    add_param_info(window, 2, 'dt')
    
    # Window length
    windowlength_label = ttk.Label(window, text="Window length")
    windowlength_label.grid(row=3, column=0, sticky = 'e', padx = padx, pady = pady)
    windowlength_entry = ttk.Entry(window, width=13)
    windowlength_entry.grid(row=3, column=1)
    windowlength_entry.insert(tk.END, str(params['labeling_window_length']))
    add_param_info(window, 3, 'window_length')
    
    # Threshold to fuse sequences of states
    Threshold_label = ttk.Label(window, text="Threshold")
    Threshold_label.grid(row=5, column=0, sticky = 'e', padx = padx, pady = pady)
    Threshold_entry = ttk.Entry(window, width=13)
    Threshold_entry.grid(row=5, column=1)
    Threshold_entry.insert(tk.END, str(params['threshold']))
    add_param_info(window, 5, 'threshold')
    
    # Maximum number of states
    Max_nb_sequences_label = ttk.Label(window, text="Maximum number of sequences")
    Max_nb_sequences_label.grid(row=6, column=0, sticky = 'e', padx = padx, pady = pady)
    Max_nb_sequences_entry = ttk.Entry(window, width=13)
    Max_nb_sequences_entry.grid(row=6, column=1)
    Max_nb_sequences_entry.insert(tk.END, str(params['max_nb_sequ_labeling']))
    add_param_info(window, 6, 'max_nb_sequences')
    
    # Depth of field
    Depth_of_field_label = ttk.Label(window, text="Depth of field")
    Depth_of_field_label.grid(row=7, column=0, sticky = 'e', padx = padx, pady = pady)
    Depth_of_field_entry = ttk.Entry(window, width=13)
    Depth_of_field_entry.grid(row=7, column=1)
    Depth_of_field_entry.insert(tk.END, str(params['cell_dims']))
    add_param_info(window, 7, 'depth_of_field')
    
    Draw_plot_label = ttk.Label(window, text="Plot labeled tracks")
    Draw_plot_label.grid(row=8, column=0, padx = padx, pady = pady, sticky = 'e')
    Draw_plot_var = tk.StringVar(window)
    Draw_plot_var.set(params['draw_plot'])
    Draw_plot_dropdown = ttk.OptionMenu(window, Draw_plot_var, Draw_plot_var.get(),
                                             "Yes",
                                             "No",
                                             style='My.TMenubutton')
    # gridded like the other hyperparameter values (plain column 1, no sticky
    # east and no extra padding) so the menu lines up with the entry column
    Draw_plot_dropdown.config(width=10)
    Draw_plot_dropdown.grid(row=8, column=1)
    add_param_info(window, 8, 'draw_plot')
    
    # Fusion model (multi- vs mono-transition) and its explanation box
    fusion_model_var = add_fusion_model_selector(window, 9, params['fusion_model'])

    # Savepath Input
    savepath_label = ttk.Label(window, text="Save Path:")
    savepath_label.grid(row=11, column=0, sticky = 'e', padx = padx, pady = pady)
    savepath_entry = ttk.Entry(window, width=50)
    savepath_entry.grid(row=11, column=1)
    savepath_entry.insert(tk.END, os.path.join(savepath, 'saved_track_predictions.csv'))
    savepath_button = ttk.Button(window, text="Browse", command=lambda: browse_savepath(savepath_entry))
    savepath_button.grid(row=11, column=2)
    
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
                                                            savepath = savepath_entry.get(),
                                                            Draw_plot = Draw_plot_var.get(),
                                                            fusion_model = fusion_model_var.get()))
    run_button.grid(row=12, column=1, columnspan=1)

    # Previous Button
    previous_button = ttk.Button(window, text="Previous", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=12, column=0, columnspan=1)

def run_predictions(window, tracks, frames, opt_metrics, dt, nb_states, frame_len, cell_dims, LocErr_type, input_LocErr, threshold, max_nb_states, savepath, Draw_plot, fusion_model = 'Multi-transition'):
    progress_window, progress_label = show_progress_window(window, "State labeling on-going...")
    try:
        result = _run_predictions_core(window, tracks, frames, opt_metrics, dt, nb_states, frame_len, cell_dims, LocErr_type, input_LocErr, threshold, max_nb_states, savepath, Draw_plot, fusion_model)
    except Exception as error:
        finish_progress_window(progress_window, progress_label, "State labeling failed:\n%s"%error)
        raise
    if result is None: # the analysis was refused before starting; an error window explains why
        progress_window.destroy()
        return
    finish_progress_window(progress_window, progress_label, "State labeling finished.\nResults saved to:\n%s"%savepath)

def _run_predictions_core(window, tracks, frames, opt_metrics, dt, nb_states, frame_len, cell_dims, LocErr_type, input_LocErr, threshold, max_nb_states, savepath, Draw_plot, fusion_model = 'Multi-transition'):
    # Run the Brownian motion analysis
    #tracks = tracks[str(length)]
    global params

    fusion_scheme = FUSION_MODELS[fusion_model]
    params['fusion_model'] = fusion_model

    params['dt'] = dt
    params['labeling_window_length'] = frame_len
    params['cell_dims'] = cell_dims
    params['max_nb_sequ_labeling'] = max_nb_states
    params['threshold'] = threshold
    
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
    
    #print('tracks', tracks)
    #print('input_LocErr', input_LocErr)
    
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
                           nb_max = 1,
                           sequence_scheme = fusion_scheme)
        
    if Draw_plot == 'Yes':
        track_list = []
        pred_list = []
        for l in tracks:
            track_list = track_list + list(tracks[l])
            pred_list = pred_list + list(preds[l])
            
        stds = np.zeros(100)

        for k in range(100):
            ID = np.random.randint(len(track_list))
            stds[k] = np.mean(np.std(track_list[ID], 0))
        
        lim = 10*np.mean(stds)
        nb_rows = 8
        
        def rgb_cm(pred, nb_states):
            pred2color = np.zeros((1, nb_states, 3))
            for state in range(nb_states):
                x = state/(nb_states-1)
                r = np.clip(1-2*x, 0, 1)
                if x <0.5:
                    g = 2*x
                else:
                    g = 1 - 2*(x-0.5)
                b = np.clip(2*x - 1, 0, 1)
                pred2color[0, state] = [r, g, b]
            return np.sum(pred[:,:,None]*pred2color, 1)
        
        plt.figure(figsize = (10,10))
        # Distinct tracks only. Drawing ID = np.random.randint(len(track_list))
        # independently for every one of the nb_rows**2 slots samples WITH
        # replacement, so the same track was shown several times: with certainty
        # when fewer than nb_rows**2 tracks are loaded (example_tracks.csv loads
        # 35 for 64 slots), and with high probability otherwise (birthday
        # effect). When the data set is smaller than the grid, only that many
        # tracks are drawn.
        nb_shown = min(nb_rows**2, len(track_list))
        shown_IDs = np.random.choice(len(track_list), size = nb_shown, replace = False)
        for k, ID in enumerate(shown_IDs):
                i, j = k // nb_rows, k % nb_rows
                track = track_list[ID]
                track = track - np.mean(track, 0, keepdims = True) + [[lim*i, lim*j]]
                pred = pred_list[ID]
                plt.plot(track[:, 0], track[:, 1], ':k')
                if nb_states == 2:
                    current_colors = plt.cm.brg(pred[:,0]/2)
                if nb_states == 3:
                    current_colors = rgb_cm(pred, nb_states)
                if nb_states > 3:
                    current_colors = rgb_cm(pred, nb_states)
                plt.scatter(track[:, 0], track[:, 1], c = current_colors, s = 8)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
    DATA = extrack.exporters.extrack_2_pandas(tracks, preds, frames = frames, opt_metrics = opt_metrics)
    if savepath is not None: # only the batch passes None, and merges this into one track file
        DATA.to_csv(savepath)

    print("State labeling completed%s"%(" and results saved to %s."%savepath if savepath is not None else "."))
    return DATA

def create_lifetime_window(window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps):
    
    loaded = load_dataset_or_error(window, path, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps)
    if loaded is None: # an error panel explains what went wrong
        return
    tracks, frames, opt_metrics, input_LocErr = loaded
        
    global params
    
    # Initial number of states
    NbStates_label = ttk.Label(window, text="Number of states:")
    NbStates_label.grid(row=0, column=0, sticky = 'e', padx = padx, pady = pady)
    NbStates_entry = ttk.Entry(window, width=13)
    NbStates_entry.grid(row=0, column=1)
    NbStates_entry.insert(tk.END, str(params['num_states']))
    add_param_info(window, 0, 'nb_states')
    
    open_button = ttk.Button(window, text="Open Parameter Window", command=lambda: ParameterWindow(window, int(NbStates_entry.get())))
                             #ParameterWindow(window, int(NbStates_entry.get())))
    open_button.grid(row=1, column=0, sticky = 'e', padx = padx, pady = pady)
    
    # Frame time
    frametime_label = ttk.Label(window, text="Frame time (in s)")
    frametime_label.grid(row=2, column=0, sticky = 'e', padx = padx, pady = pady)
    frametime_entry = ttk.Entry(window, width=13)
    frametime_entry.grid(row=2, column=1)
    frametime_entry.insert(tk.END, str(params['dt']))
    add_param_info(window, 2, 'dt')
    
    # Maximum number of states
    Max_nb_sequences_label = ttk.Label(window, text="Maximum number of sequences")
    Max_nb_sequences_label.grid(row=6, column=0, sticky = 'e', padx = padx, pady = pady)
    Max_nb_sequences_entry = ttk.Entry(window, width=13)
    Max_nb_sequences_entry.grid(row=6, column=1)
    Max_nb_sequences_entry.insert(tk.END, str(params['max_nb_sequ_histograms']))
    add_param_info(window, 6, 'max_nb_sequences')
    
    # Depth of field
    Depth_of_field_label = ttk.Label(window, text="Depth of field")
    Depth_of_field_label.grid(row=7, column=0, sticky = 'e', padx = padx, pady = pady)
    Depth_of_field_entry = ttk.Entry(window, width=13)
    Depth_of_field_entry.grid(row=7, column=1)
    Depth_of_field_entry.insert(tk.END, str(params['cell_dims']))
    add_param_info(window, 7, 'depth_of_field')
    
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
    add_param_info(window, 8, 'draw_plot')
    
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
    progress_window, progress_label = show_progress_window(window, "Lifetime histograms on-going...")
    try:
        result = _run_lifetime_core(window, tracks, dt, nb_states, cell_dims, LocErr_type, input_LocErr, max_nb_states, draw_plot, savepath)
    except Exception as error:
        finish_progress_window(progress_window, progress_label, "Lifetime histograms failed:\n%s"%error)
        raise
    if result is None: # the analysis was refused before starting; an error window explains why
        progress_window.destroy()
        return
    finish_progress_window(progress_window, progress_label, "Lifetime histograms finished.\nResults saved to:\n%s"%savepath)

def _run_lifetime_core(window, tracks, dt, nb_states, cell_dims, LocErr_type, input_LocErr, max_nb_states, draw_plot, savepath):
    # Run the Brownian motion analysis
    #tracks = tracks[str(length)]
    global params
    
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
    if savepath is not None:
        DATA.to_csv(savepath)
    
    print("Lifetime histograms completed%s"%(" and results saved to %s."%savepath if savepath is not None else "."))
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
    return DATA

def create_refinement_window(window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps):
    
    loaded = load_dataset_or_error(window, path, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps)
    if loaded is None: # an error panel explains what went wrong
        return
    tracks, frames, opt_metrics, input_LocErr = loaded
        
    global params
    
    # Initial number of states
    NbStates_label = ttk.Label(window, text="Number of states:")
    NbStates_label.grid(row=0, column=0, sticky = 'e', padx = padx, pady = pady)
    NbStates_entry = ttk.Entry(window, width=13)
    NbStates_entry.grid(row=0, column=1)
    NbStates_entry.insert(tk.END, str(params['num_states']))
    add_param_info(window, 0, 'nb_states')
    
    open_button = ttk.Button(window, text="Open Parameter Window", command=lambda: ParameterWindow(window, int(NbStates_entry.get())))
                             #ParameterWindow(window, int(NbStates_entry.get())))
    open_button.grid(row=1, column=0, sticky = 'e', padx = padx, pady = pady)
    
    # Frame time
    frametime_label = ttk.Label(window, text="Frame time (in s)")
    frametime_label.grid(row=2, column=0, sticky = 'e', padx = padx, pady = pady)
    frametime_entry = ttk.Entry(window, width=13)
    frametime_entry.grid(row=2, column=1)
    frametime_entry.insert(tk.END, str(params['dt']))
    add_param_info(window, 2, 'dt')
    
    # Window length
    windowlength_label = ttk.Label(window, text="Window length")
    windowlength_label.grid(row=3, column=0, sticky = 'e', padx = padx, pady = pady)
    windowlength_entry = ttk.Entry(window, width=13)
    windowlength_entry.grid(row=3, column=1)
    windowlength_entry.insert(tk.END, str(params['labeling_window_length']))
    add_param_info(window, 3, 'window_length')
    
    # Threshold to fuse sequences of states
    Threshold_label = ttk.Label(window, text="Threshold")
    Threshold_label.grid(row=5, column=0, sticky = 'e', padx = padx, pady = pady)
    Threshold_entry = ttk.Entry(window, width=13)
    Threshold_entry.grid(row=5, column=1)
    Threshold_entry.insert(tk.END, str(params['threshold']))
    add_param_info(window, 5, 'threshold')
    
    # Maximum number of sequences of states
    Max_nb_sequences_label = ttk.Label(window, text="Maximum number of sequences")
    Max_nb_sequences_label.grid(row=6, column=0, sticky = 'e', padx = padx, pady = pady)
    Max_nb_sequences_entry = ttk.Entry(window, width=13)
    Max_nb_sequences_entry.grid(row=6, column=1)
    Max_nb_sequences_entry.insert(tk.END, str(params['max_nb_sequ_histograms']))
    add_param_info(window, 6, 'max_nb_sequences')
    
    # Depth of field
    Depth_of_field_label = ttk.Label(window, text="Depth of field")
    Depth_of_field_label.grid(row=7, column=0, sticky = 'e', padx = padx, pady = pady)
    Depth_of_field_entry = ttk.Entry(window, width=13)
    Depth_of_field_entry.grid(row=7, column=1)
    Depth_of_field_entry.insert(tk.END, str(params['cell_dims']))
    add_param_info(window, 7, 'depth_of_field')
    
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
    progress_window, progress_label = show_progress_window(window, "Position refinement on-going...")
    try:
        result = _run_refinement_core(window, tracks, frames, opt_metrics, dt, nb_states, frame_len, cell_dims, LocErr_type, input_LocErr, threshold, max_nb_states, savepath)
    except Exception as error:
        finish_progress_window(progress_window, progress_label, "Position refinement failed:\n%s"%error)
        raise
    if result is None: # the analysis was refused before starting; an error window explains why
        progress_window.destroy()
        return
    finish_progress_window(progress_window, progress_label, "Position refinement finished.\nResults saved to:\n%s"%savepath)

def _run_refinement_core(window, tracks, frames, opt_metrics, dt, nb_states, frame_len, cell_dims, LocErr_type, input_LocErr, threshold, max_nb_states, savepath):
    # Run the Brownian motion analysis
    #tracks = tracks[str(length)]
    global params
    
    params['dt'] = dt
    params['labeling_window_length'] = frame_len
    params['cell_dims'] = cell_dims
    params['max_nb_sequ_histograms'] = max_nb_states
    params['threshold'] = threshold
    
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
    if savepath is not None: # only the batch passes None, and merges this into one track file
        dataframe.to_csv(savepath)
    
    print('Position refinement finished%s'%(' and saved at "%s"'%savepath if savepath is not None else ''))
    return dataframe


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
def create_batch_window(window, path, savepath, min_length, max_length, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps, batch_mode):
    """
    The window of the three batch analyses. `path` must be a folder; every csv
    and TrackMate xml file it contains is processed with the stages of
    BATCH_STAGES[batch_mode]. Results go to a BATCH_RESULTS_DIRNAME folder
    inside `savepath`, which default_save_folder has set to the parent of the
    dataset folder, so that they do not land among the replicates.

    Each replicate leaves one BATCH_TRACKS_SUFFIX track file -- state
    predictions, refined positions and their localization error, and the
    optional metrics, in one table -- plus its lifetime histograms when that
    stage runs. The fitted parameters are not written per replicate: they all go
    to the single BATCH_SUMMARY_NAME file.
    """
    if not os.path.isdir(path):
        show_error_url(window, "Batch analyses take a FOLDER path, and the informed path is not one:\n%s\nGo back and pick the folder holding the csv/xml files with Browse > 'Select a folder...', or type it in the Path field.\n"%path, url=None)
        return
    files = sorted(glob(os.path.join(path, '*.csv')) + glob(os.path.join(path, '*.xml')))
    if len(files) == 0:
        show_error_url(window, "No csv or xml file detected in the informed folder:\n%s\n"%path, url=None)
        return
    stages = BATCH_STAGES[batch_mode]

    global params

    # Initial number of states
    NbStates_label = ttk.Label(window, text="Number of states:")
    NbStates_label.grid(row=0, column=0, sticky = 'e', padx = padx, pady = pady)
    NbStates_entry = ttk.Entry(window, width=13)
    NbStates_entry.grid(row=0, column=1)
    NbStates_entry.insert(tk.END, str(params['num_states']))
    add_param_info(window, 0, 'nb_states')

    open_button = ttk.Button(window, text="Open Parameter Window", command=lambda: ParameterWindow(window, int(NbStates_entry.get())))
    open_button.grid(row=1, column=0, sticky = 'e', padx = padx, pady = pady)

    # Frame time
    frametime_label = ttk.Label(window, text="Frame time (in s)")
    frametime_label.grid(row=2, column=0, sticky = 'e', padx = padx, pady = pady)
    frametime_entry = ttk.Entry(window, width=13)
    frametime_entry.grid(row=2, column=1)
    frametime_entry.insert(tk.END, str(params['dt']))
    add_param_info(window, 2, 'dt')

    # Window length used by the fit
    windowlength_label = ttk.Label(window, text="Window length (fitting)")
    windowlength_label.grid(row=3, column=0, sticky = 'e', padx = padx, pady = pady)
    windowlength_entry = ttk.Entry(window, width=13)
    windowlength_entry.grid(row=3, column=1)
    windowlength_entry.insert(tk.END, str(params['fitting_window_length']))
    add_param_info(window, 3, 'window_length')

    # Window length used by the labeling and refinement stages
    label_windowlength_entry = None
    if 'labeling' in stages or 'refinement' in stages:
        label_windowlength_label = ttk.Label(window, text="Window length (labeling)")
        label_windowlength_label.grid(row=4, column=0, sticky = 'e', padx = padx, pady = pady)
        label_windowlength_entry = ttk.Entry(window, width=13)
        label_windowlength_entry.grid(row=4, column=1)
        label_windowlength_entry.insert(tk.END, str(params['labeling_window_length']))
        add_param_info(window, 4, 'window_length')

    # Number of substeps
    nb_substeps_label = ttk.Label(window, text="Number of substeps")
    nb_substeps_label.grid(row=5, column=0, sticky = 'e', padx = padx, pady = pady)
    nb_substeps_entry = ttk.Entry(window, width=13)
    nb_substeps_entry.grid(row=5, column=1)
    nb_substeps_entry.insert(tk.END, str(params['nb_substeps']))
    add_param_info(window, 5, 'nb_substeps')

    # Threshold to fuse sequences of states
    Threshold_label = ttk.Label(window, text="Threshold")
    Threshold_label.grid(row=6, column=0, sticky = 'e', padx = padx, pady = pady)
    Threshold_entry = ttk.Entry(window, width=13)
    Threshold_entry.grid(row=6, column=1)
    Threshold_entry.insert(tk.END, str(params['threshold']))
    add_param_info(window, 6, 'threshold')

    # Maximum number of sequences (fitting)
    Max_nb_sequences_label = ttk.Label(window, text="Maximum number of sequences")
    Max_nb_sequences_label.grid(row=7, column=0, sticky = 'e', padx = padx, pady = pady)
    Max_nb_sequences_entry = ttk.Entry(window, width=13)
    Max_nb_sequences_entry.grid(row=7, column=1)
    Max_nb_sequences_entry.insert(tk.END, str(params['max_nb_sequ']))
    add_param_info(window, 7, 'max_nb_sequences')

    # Depth of field
    Depth_of_field_label = ttk.Label(window, text="Depth of field")
    Depth_of_field_label.grid(row=8, column=0, sticky = 'e', padx = padx, pady = pady)
    Depth_of_field_entry = ttk.Entry(window, width=13)
    Depth_of_field_entry.grid(row=8, column=1)
    Depth_of_field_entry.insert(tk.END, str(params['cell_dims']))
    add_param_info(window, 8, 'depth_of_field')

    # number of iterations of the fitting methods
    nb_iter_label = ttk.Label(window, text="Number of iterations")
    nb_iter_label.grid(row=9, column=0, sticky = 'e', padx = padx, pady = pady)
    nb_iter_entry = ttk.Entry(window, width=13)
    nb_iter_entry.grid(row=9, column=1)
    nb_iter_entry.insert(tk.END, str(params['nb_iters']))
    add_param_info(window, 9, 'nb_iters')

    # Fusion model (multi- vs mono-transition) and its explanation cell
    fusion_model_var = add_fusion_model_selector(window, 10, params['fusion_model'])

    # The files that will be processed
    nb_csv = sum(1 for f in files if f.endswith('.csv'))
    files_text = "%d files detected (%d csv, %d xml): %s"%(len(files), nb_csv, len(files)-nb_csv,
                 ', '.join(os.path.basename(f) for f in files[:6]) + (', ...' if len(files) > 6 else ''))
    files_label = ttk.Label(window, text=files_text, wraplength=430, justify='left')
    files_label.grid(row=11, column=0, columnspan=2, padx = padx, pady = pady, sticky='w')
    add_param_info(window, 11, 'batch_files')

    # Save folder
    savepath_label = ttk.Label(window, text="Save Folder:")
    savepath_label.grid(row=12, column=0, sticky = 'e', padx = padx, pady = pady)
    savepath_entry = ttk.Entry(window, width=50)
    savepath_entry.grid(row=12, column=1)
    savepath_entry.insert(tk.END, batch_save_folder(savepath)) # <parent of the dataset folder>/Results
    savepath_button = ttk.Button(window, text="Browse", command=lambda: browse_savefolder(savepath_entry))
    savepath_button.grid(row=12, column=2)

    # Run Button
    run_button = ttk.Button(window, text="Start %s"%batch_mode.lower(),
                            command=lambda: run_batch(window,
                                                      files,
                                                      stages,
                                                      dt = float(frametime_entry.get()),
                                                      nb_states = int(NbStates_entry.get()),
                                                      nb_iterations = int(nb_iter_entry.get()),
                                                      nb_substeps = int(nb_substeps_entry.get()),
                                                      fit_frame_len = int(windowlength_entry.get()),
                                                      label_frame_len = int(label_windowlength_entry.get()) if label_windowlength_entry is not None else int(windowlength_entry.get()),
                                                      cell_dims = float(Depth_of_field_entry.get()),
                                                      LocErr_type = LocErr_type,
                                                      LocErr_input_name = LocErr_input_name,
                                                      Optional_input_name = Optional_input_name,
                                                      headers = headers,
                                                      max_dist = max_dist,
                                                      remove_no_disps = remove_no_disps,
                                                      min_length = min_length,
                                                      max_length = max_length,
                                                      threshold = float(Threshold_entry.get()),
                                                      max_nb_states = int(Max_nb_sequences_entry.get()),
                                                      fusion_model = fusion_model_var.get(),
                                                      save_folder = savepath_entry.get(),
                                                      batch_mode = batch_mode))
    run_button.grid(row=13, column=1, columnspan=1)

    # Previous Button
    previous_button = ttk.Button(window, text="Other analyses", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=13, column=0, columnspan=1)

def run_batch(window, files, stages, dt, nb_states, nb_iterations, nb_substeps, fit_frame_len, label_frame_len, cell_dims, LocErr_type, LocErr_input_name, Optional_input_name, headers, max_dist, remove_no_disps, min_length, max_length, threshold, max_nb_states, fusion_model, save_folder, batch_mode):
    """
    Run the selected stages on every file of the batch. Each file starts from
    the same user-set parameters (snapshotted before the batch and restored at
    the end); within one file, fitting runs first and its fitted parameters
    feed the later stages, exactly as running the single-file analyses in that
    order would. A file that fails is reported and the batch moves on.

    `save_folder` is created if it does not exist. The labeling and refinement
    tables of a replicate are merged into one track file instead of being saved
    separately, and the fitting results are not saved per replicate at all: they
    are gathered in one summary file, which is also shown as a table in the
    window that closes the batch.
    """
    global params

    if FUSION_MODELS[fusion_model] == 'ages' and nb_substeps != 1:
        show_error_url(window, "The mono-transition fusion model does not support substeps.\nSet 'Number of substeps' to 1 or select the multi-transition model.", url=None)
        return

    # the proposed folder does not exist yet the first time an experiment is
    # analysed, and neither does one the user typed by hand
    try:
        os.makedirs(save_folder, exist_ok = True)
    except Exception as error:
        show_error_url(window, "The save folder could not be created:" + os.linesep
                       + "%s"%save_folder + os.linesep + "%s"%error + os.linesep, url=None)
        return

    stage_titles = {'fitting': 'fitting', 'labeling': 'state labeling',
                    'histogram': 'lifetime histograms', 'refinement': 'position refinement'}
    progress_window, progress_label = show_progress_window(window, "%s on-going..."%batch_mode)

    snapshot = copy.deepcopy(params)
    results = []
    fitted = [] # (file name, one-row fitting table) of every replicate that fitted
    for file_ID, file in enumerate(files):
        name = os.path.basename(file)
        base = os.path.join(save_folder, os.path.splitext(name)[0])
        # every file starts from the same user-set parameters
        params.clear()
        params.update(copy.deepcopy(snapshot))
        try:
            progress_label.config(text="%s on-going...\nFile %d/%d: %s\nStage: loading"%(batch_mode, file_ID+1, len(files), name))
            progress_window.update()
            tracks, frames, opt_metrics, input_LocErr = read_dataset_file(
                file, min_length, max_length, LocErr_type, LocErr_input_name,
                Optional_input_name, headers, max_dist, remove_no_disps)
            tables = {} # what each stage produced, before deciding what to write
            for stage in stages:
                progress_label.config(text="%s on-going...\nFile %d/%d: %s\nStage: %s"%(batch_mode, file_ID+1, len(files), name, stage_titles[stage]))
                progress_window.update()
                # the cores of the quality-metric localization errors transform
                # input_LocErr in place, so every stage gets its own copy
                LocErr_copy = None if input_LocErr is None else dict((l, input_LocErr[l].copy()) for l in input_LocErr)
                if stage == 'fitting':
                    out = _run_fitting_core(window, tracks, dt, nb_states, nb_iterations,
                                            nb_substeps, fit_frame_len, cell_dims, LocErr_type,
                                            LocErr_copy, threshold, max_nb_states,
                                            None, fusion_model) # summarised, not saved per file
                elif stage == 'labeling':
                    out = _run_predictions_core(window, tracks, frames, opt_metrics, dt, nb_states,
                                                label_frame_len, cell_dims, LocErr_type, LocErr_copy,
                                                threshold, params['max_nb_sequ_labeling'],
                                                None, 'No', fusion_model) # goes to the track file
                elif stage == 'histogram':
                    out = _run_lifetime_core(window, tracks, dt, nb_states, cell_dims, LocErr_type,
                                             LocErr_copy, params['max_nb_sequ_histograms'], 'No',
                                             base + '_lifetime_histograms.csv')
                elif stage == 'refinement':
                    out = _run_refinement_core(window, tracks, frames, opt_metrics, dt, nb_states,
                                               label_frame_len, cell_dims, LocErr_type, LocErr_copy,
                                               threshold, params['max_nb_sequ_histograms'],
                                               None) # goes to the track file
                if out is None:
                    raise RuntimeError('the %s stage was refused'%stage_titles[stage])
                tables[stage] = out

            outputs = []
            # the predictions and the refined positions describe the same
            # detections, so they leave one track file rather than two
            track_table = merge_track_tables(tables.get('labeling'), tables.get('refinement'))
            if track_table is not None:
                track_path = base + BATCH_TRACKS_SUFFIX
                track_table.to_csv(track_path, index = False)
                outputs.append(track_path)
                print('Batch: %s written'%track_path)
            if 'histogram' in tables:
                outputs.append(base + '_lifetime_histograms.csv')
            if 'fitting' in tables:
                fitted.append((name, tables['fitting']))
            results.append((name, 'done', outputs))
        except Exception as error:
            print('Batch: %s failed: %s'%(name, error))
            results.append((name, 'failed', str(error)))

    # the batch leaves the global parameters as it found them
    params.clear()
    params.update(snapshot)

    # one table for the whole batch, written next to the per-replicate results
    summary, summary_path = collect_fitting_summary(fitted, save_folder)

    nb_done = sum(1 for r in results if r[1] == 'done')
    message = "%s finished.\nProcessed %d/%d files.\nResults saved to:\n%s"%(batch_mode, nb_done, len(files), save_folder)
    failed = [r for r in results if r[1] == 'failed']
    if failed:
        message += "\nFailed: " + ", ".join("%s (%s)"%(r[0], r[2][:60]) for r in failed)
    if summary_path:
        message += "\nFitting results of the %d replicates gathered in:\n%s"%(len(summary), os.path.basename(summary_path))
    if summary is not None:
        show_fitting_summary(progress_window, summary) # between the message and OK
    finish_progress_window(progress_window, progress_label, message)
    return results

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
                                   D_max = 10*params['diff_coeffs'][-1], # maximal diffusion coefficient allowed
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
         - Initial fractions (the equilibrium fractions implied by the
           transition probabilities are shown next to them, read-only)
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
        
        # Initial fractions: the occupancies at the first time point of the tracks
        ttk.Label(self.window, text="Initial fractions:").grid(row=6, column=0, padx=5, pady=5, sticky="ew")
        self.frac_entries = []
        for k in range(nb_states):
            self.frac_entries = self.frac_entries + [ttk.Entry(self.window, width=10)]
            self.frac_entries[k].grid(row=6, column=1+k, padx=5, pady=5)
            self.frac_entries[k].insert(0, str(np.round(params['fractions'][k], 4)))
        add_param_info(self.window, 6, 'initial_fractions', column=1+nb_states)

        # Equilibrium fractions implied by the transition probabilities (read-only)
        ttk.Label(self.window, text="Equilibrium fractions:").grid(row=7, column=0, padx=5, pady=5, sticky="ew")
        self.equilibrium_labels = []
        for k in range(nb_states):
            self.equilibrium_labels.append(ttk.Label(self.window, text='', anchor='center'))
            self.equilibrium_labels[k].grid(row=7, column=1+k, padx=5, pady=5)
        add_param_info(self.window, 7, 'equilibrium_fractions', column=1+nb_states)
        
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
                self.transition_entries[i*nb_states+j].bind('<KeyRelease>', self.update_equilibrium)
        self.update_equilibrium()
        
        # OK button to validate & send parameters back
        ttk.Button(self.window, text="OK", command=self.ok_clicked).grid(row=12, column=0, columnspan=3, pady=10)
                
    def update_equilibrium(self, event=None):
        """refresh the read-only equilibrium fractions from the transition entries"""
        try:
            transition_probs = np.zeros((self.nb_states, self.nb_states))
            for i in range(self.nb_states):
                for j in range(self.nb_states):
                    transition_probs[i, j] = float(self.transition_entries[i*self.nb_states+j].get())
            equilibrium = equilibrium_fractions(transition_probs, params.get('nb_substeps', 1))
            for k in range(self.nb_states):
                self.equilibrium_labels[k].config(text=str(np.round(equilibrium[k], 4)))
        except Exception: # incomplete or non numerical entries while typing
            for k in range(self.nb_states):
                self.equilibrium_labels[k].config(text='?')

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
 'dt': 0.1, "fitting_window_length": 6, "labeling_window_length": 10, 'nb_iters': 3, 'max_nb_sequ': 200, 'threshold': 0.1, 'nb_substeps':  1, 'max_nb_sequ_labeling': 50, 'max_nb_sequ_histograms': 300, 'draw_plot': 'Yes', 'LocErr_input_name': '', 'Optional_input_names': '', 'fusion_model': 'Multi-transition', }
nb_states = 2

# The dataset path can be either a single file -- what the four single-dataset
# analyses expect -- or a folder, which is what the batch analyses take (they
# run over every csv/xml it contains). No file dialog offers both kinds, so
# Browse is a small menu with one entry per kind; set_dataset_path is what the
# two entries share.
# the single-dataset analyses read csv only (the batch ones also read xml, but
# they take a folder, so they never go through this dialog)
DATASET_FILETYPES = [("CSV files", "*.csv"), ("All files", "*.*")]

def set_dataset_path(path):
    """Put a browsed path in the Path field and remember it for the next instance."""
    if not path: # cancelling keeps the current path instead of clearing it
        return
    path = os.path.normpath(path)
    path_entry.delete(0, 'end')
    path_entry.insert(tk.END, path)
    save_gui_config(last_path=path)
    update_path_kind_label()

def browser():
    """Browse for a single dataset file."""
    set_dataset_path(filedialog.askopenfilename(
        initialdir=initialdir_from(path_entry.get()),
        title="Select a dataset file",
        filetypes=DATASET_FILETYPES))

def browse_folder():
    """Browse for a folder of dataset files -- what the batch analyses take."""
    set_dataset_path(filedialog.askdirectory(
        initialdir=initialdir_from(path_entry.get()),
        title="Select a folder of dataset files",
        mustexist=True))

def path_kind_text():
    """
    The line shown under the Path field: what the current path is, and how many
    files a folder holds. The file/folder distinction decides which analyses can
    run, so it is worth seeing before clicking Next rather than in an error panel
    afterwards.
    """
    path = path_entry.get().strip()
    if path == '':
        return 'no path set'
    if os.path.isdir(path):
        nb_csv = len(glob(os.path.join(path, '*.csv')))
        nb_xml = len(glob(os.path.join(path, '*.xml')))
        if nb_csv + nb_xml == 0:
            return 'folder: no csv or xml file in it'
        return 'folder: %d csv, %d xml -- batch analyses run over all of them'%(nb_csv, nb_xml)
    if os.path.isfile(path):
        return 'single file'
    return 'this path does not exist'

def update_path_kind_label(*args):
    """Refresh that line; bound to every edit of the Path field."""
    path_kind_label.config(text=path_kind_text())

def browse_savepath(entry_widget):
    filename = filedialog.asksaveasfilename(
        initialdir=initialdir_from(entry_widget.get()),
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
# the entry and, under it, the line saying what the current path is; they share
# one cell through a frame so the rest of the grid keeps its row numbers
path_frame = ttk.Frame(root)
path_frame.grid(row=0, column=1, columnspan=3, padx = padx, pady = pady, sticky = 'e')
path_entry = ttk.Entry(path_frame, width=64)
path_entry.pack(fill='x')
path_kind_label = ttk.Label(path_frame, text='', foreground='#4b5563')
path_kind_label.pack(anchor='w')
# start from the path of the previous session when it still makes sense:
# the saved path itself if it still exists, otherwise its folder, otherwise cwd
_last_path = load_gui_config().get('last_path', '')
if not (type(_last_path) == str and os.path.exists(_last_path)):
    _parent = os.path.dirname(_last_path) if type(_last_path) == str else ''
    _last_path = _parent if os.path.isdir(_parent) else os.getcwd()
path_entry.insert(tk.END, _last_path)
#path_entry.insert(tk.END,  r'D:\Maria_DATA\Tracks\4.csv')
#path_button = ttk.Button(root, text="Browse", command=lambda: path_entry.insert(tk.END, filedialog.askopenfilename()))
#path_button = ttk.Button(root, text="Browse", command=lambda: (path_entry.insert(tk.END, filedialog.askopenfilename(initialdir=os.path.expanduser('~'), title="Select File"))))
# Browse offers both kinds of path: a file for the single-dataset analyses, a
# folder for the batch ones.
path_button = ttk.Menubutton(root, text="Browse", style='My.TMenubutton', width=10)
path_menu = tk.Menu(path_button, tearoff=0)
path_menu.add_command(label="Select a file...", command=browser)
path_menu.add_command(label="Select a folder...   (batch analyses)", command=browse_folder)
path_button['menu'] = path_menu
path_button.grid(row=0, column=4, padx = padx, pady = pady, sticky = 'ne')

# the kind line follows a typed path as well as a browsed one
path_entry.bind('<KeyRelease>', update_path_kind_label)
update_path_kind_label()

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
max_length_entry.insert(tk.END, "15")

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
                                        "Batch Fitting",
                                        "Batch Fitting + Labeling",
                                        "Batch All",
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
LocErr_input_entry.insert(tk.END, params['LocErr_input_name'])

LocErr_type_label = ttk.Label(root, text="Additional metrics")
LocErr_type_label.grid(row=6, column=0, padx = padx, pady = pady, sticky = 'e')
Optional_input_entry = ttk.Entry(root, width=88)
Optional_input_entry.grid(row=6, column=1, columnspan=4, padx = padx, pady = pady, sticky = 'e')
Optional_input_entry.insert(tk.END, params['Optional_input_names'])

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

