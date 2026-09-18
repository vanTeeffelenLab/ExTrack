#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Browsing a folder as well as a file, and what the batch analyses then do with
it: the save folder they propose, the single summary file they write for all
the replicates, and the table of fitted parameters shown at the end.

  1. the Browse menu offers a file entry and a folder entry, and browse_folder
     fills the Path field the way browser does (the dialogs are stubbed);
  2. path_kind_text tells a file from a folder, and counts what a folder holds;
  3. default_save_folder is the PARENT of the folder holding the dataset, for a
     folder path and for a file path alike, and the batch window proposes a
     Results directory inside it, without creating it yet;
  4. collect_fitting_summary gathers one row per replicate from the tables the
     fitting stage returns, and merge_track_tables puts the predictions and the
     refined positions in one table;
  5. end to end: a two-replicate Batch Fitting creates that Results directory,
     leaves the dataset folder untouched, writes batch_fitting_summary.csv there
     and nothing else, and shows it as a table in the final window.
"""

import os
import runpy
import shutil
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.chdir(REPO)

CONFIG = os.path.join(tempfile.mkdtemp(prefix='extrack_cfg_'), 'gui_config.json')
os.environ['EXTRACK_GUI_CONFIG'] = CONFIG

import tkinter as tk               # noqa: E402
from tkinter import ttk            # noqa: E402
from tkinter import filedialog     # noqa: E402
tk.Misc.mainloop = lambda self, n=0: None

fails = []


def check(name, ok, detail=''):
    print(('  OK   ' if ok else '  FAIL ') + name + ('   ' + detail if detail else ''))
    if not ok:
        fails.append(name)


ns = runpy.run_path(os.path.join(REPO, 'ExTrack_GUI.py'))
ns['root'].withdraw()
import extrack                     # noqa: E402
extrack.tracking.set_numba('auto', threads=8)

HEADERS = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID']


def write_csv(path, nb_tracks, T, seed):
    rng = np.random.default_rng(seed)
    d = np.array([0.0, np.sqrt(2 * 1.0 * 0.1)])
    rows = []
    for tid in range(nb_tracks):
        st = rng.integers(0, 2)
        xy = np.zeros(2)
        for t in range(T):
            if t:
                if rng.random() < 0.1:
                    st = 1 - st
                xy = xy + rng.normal(0, max(d[st], 1e-6), 2)
            obs = xy + rng.normal(0, 0.03, 2)
            rows.append((obs[0], obs[1], t, tid))
    pd.DataFrame(rows, columns=HEADERS).to_csv(path, index=False)


# an experiment folder holding a folder of replicates, so that 'the parent of
# the dataset folder' is a real place and not the temp root
EXPERIMENT = tempfile.mkdtemp(prefix='extrack_exp_')
DATASETS = os.path.join(EXPERIMENT, 'replicates')
os.mkdir(DATASETS)
write_csv(os.path.join(DATASETS, 'rep1.csv'), 25, 7, 11)
write_csv(os.path.join(DATASETS, 'rep2.csv'), 25, 7, 12)

# ---------------------------------------------------------------------------
print('')
print('1) Browse offers a file and a folder')
menu = ns['path_menu']
labels = [menu.entrycget(i, 'label') for i in range(menu.index('end') + 1)]
check('the Browse menu has one entry per kind of path',
      len(labels) == 2 and 'file' in labels[0].lower() and 'folder' in labels[1].lower(),
      str(labels))
check('   and the folder entry says it is what the batch analyses take',
      'batch' in labels[1].lower(), labels[1])

real_open = filedialog.askopenfilename
real_dir = filedialog.askdirectory
seen = {}
try:
    filedialog.askdirectory = lambda **k: seen.update(dir=k) or DATASETS
    ns['browse_folder']()
    check('browse_folder puts the folder in the Path field',
          ns['path_entry'].get() == os.path.normpath(DATASETS))
    check('   and remembers it for the next instance of the GUI',
          ns['load_gui_config']().get('last_path') == os.path.normpath(DATASETS))
    check('   asking only for existing folders',
          seen.get('dir', {}).get('mustexist') is True)

    filedialog.askdirectory = lambda **k: ''      # cancelling
    ns['browse_folder']()
    check('cancelling the folder dialog keeps the current path',
          ns['path_entry'].get() == os.path.normpath(DATASETS))

    filedialog.askopenfilename = lambda **k: seen.update(open=k) or os.path.join(DATASETS, 'rep1.csv')
    ns['browser']()
    check('browsing a file still works and replaces the folder',
          ns['path_entry'].get() == os.path.join(DATASETS, 'rep1.csv'))
    check('   and the file dialog filters on what the analyses can read',
          seen.get('open', {}).get('filetypes') == ns['DATASET_FILETYPES'])
finally:
    filedialog.askopenfilename = real_open
    filedialog.askdirectory = real_dir

# ---------------------------------------------------------------------------
print('')
print('2) the Path field says which kind of path it holds')


def kind_of(path):
    ns['path_entry'].delete(0, tk.END)
    ns['path_entry'].insert(tk.END, path)
    ns['update_path_kind_label']()
    return ns['path_kind_label'].cget('text')


check('a file reads as a single file',
      kind_of(os.path.join(DATASETS, 'rep1.csv')) == 'single file')
folder_text = kind_of(DATASETS)
check('a folder reads as a folder, with its file counts',
      folder_text.startswith('folder:') and '2 csv' in folder_text and '0 xml' in folder_text,
      folder_text)
check('an empty folder says so', kind_of(EXPERIMENT).startswith('folder: no csv or xml'))
check('a path that does not exist says so',
      kind_of(os.path.join(DATASETS, 'nope.csv')) == 'this path does not exist')
check('an empty field says so', kind_of('') == 'no path set')

# ---------------------------------------------------------------------------
print('')
print('3) the save folder: a Results directory in the parent of the dataset folder')
check('a dataset FOLDER proposes its parent',
      ns['default_save_folder'](DATASETS) == EXPERIMENT)
check('a dataset FILE proposes the parent of its folder too',
      ns['default_save_folder'](os.path.join(DATASETS, 'rep1.csv')) == EXPERIMENT)
check('a trailing separator changes nothing',
      ns['default_save_folder'](DATASETS + os.sep) == EXPERIMENT)
root_dir = os.path.abspath(os.sep)
check('a drive root falls back to itself instead of vanishing',
      ns['default_save_folder'](root_dir) == os.path.normpath(root_dir))

RESULTS = os.path.join(EXPERIMENT, ns['BATCH_RESULTS_DIRNAME'])
check('a batch adds a Results directory to it',
      ns['batch_save_folder'](ns['default_save_folder'](DATASETS)) == RESULTS, RESULTS)

win = tk.Toplevel(ns['root'])
win.withdraw()
ns['create_batch_window'](win, DATASETS, ns['default_save_folder'](DATASETS), 5, 15,
                          'Fitted parameter', [], [], HEADERS, 1.0, True, 'Batch Fitting')
entries = {}
for w in win.winfo_children():
    gi = w.grid_info()
    if gi and w.winfo_class() == 'TEntry':
        entries[int(gi['row'])] = w
check('the batch window proposes it as its Save Folder',
      entries[12].get() == RESULTS, entries[12].get())
check('   and opening the window has not created it yet',
      not os.path.exists(RESULTS))

# ---------------------------------------------------------------------------
print('')
print('4) collect_fitting_summary, from the tables the fitting stage returns')
fake = tempfile.mkdtemp(prefix='extrack_sum_')
fit_rows = [(name, pd.DataFrame([{'exp': None, 'likelihood': 100.0 + i,
                                  'D0': 0.01 * (i + 1), 'D1': 1.0 + i,
                                  'equilibrium_F0': 0.4}]))
            for i, name in enumerate(['a.csv', 'b.csv'])]
summary, summary_path = ns['collect_fitting_summary'](fit_rows, fake)
check('one row per replicate', summary is not None and len(summary) == 2)
check('   named by the dataset, with the path column dropped',
      list(summary['dataset']) == ['a.csv', 'b.csv'] and 'exp' not in summary.columns
      and list(summary.columns)[0] == 'dataset', str(list(summary.columns)))
check('   the fitted values are carried over',
      list(summary['D1']) == [1.0, 2.0] and list(summary['likelihood']) == [100.0, 101.0])
check('   written as %s' % ns['BATCH_SUMMARY_NAME'],
      summary_path == os.path.join(fake, ns['BATCH_SUMMARY_NAME'])
      and os.path.isfile(summary_path))
reread = pd.read_csv(summary_path)
check('   and the file round-trips without an index column',
      list(reread.columns) == list(summary.columns), str(list(reread.columns)))
check('   the caller keeps its own table unchanged',
      list(fit_rows[0][1].columns) == ['exp', 'likelihood', 'D0', 'D1', 'equilibrium_F0'],
      str(list(fit_rows[0][1].columns)))
check('a batch in which nothing fitted writes no summary',
      ns['collect_fitting_summary']([], fake) == (None, None))
check('a replicate with no fitting table is skipped, not fatal',
      ns['collect_fitting_summary']([('gone.csv', None)], fake) == (None, None))

print('')
print('4b) merge_track_tables')
labeled = pd.DataFrame({'POSITION_X': [0.0, 1.0, 2.0], 'POSITION_Y': [3.0, 4.0, 5.0],
                        'FRAME': [0, 1, 0], 'TRACK_ID': [0, 0, 1],
                        'pred_0': [0.9, 0.8, 0.1], 'pred_1': [0.1, 0.2, 0.9],
                        'QUALITY': [7.0, 8.0, 9.0]})
refined = pd.DataFrame({'POSITION_X': [0.0, 1.0, 2.0], 'POSITION_Y': [3.0, 4.0, 5.0],
                        'FRAME': [0.0, 1.0, 0.0], 'TRACK_ID': [0.0, 0.0, 1.0],
                        'QUALITY': [7.0, 8.0, 9.0],
                        'Refined_position_X': [0.1, 1.1, 2.1],
                        'Refined_position_Y': [3.1, 4.1, 5.1],
                        'Refined_localization_error': [0.02, 0.03, 0.04]})
merged = ns['merge_track_tables'](labeled, refined)
check('the shared columns are written once',
      list(merged.columns) == list(labeled.columns) + ['Refined_position_X',
                                                       'Refined_position_Y',
                                                       'Refined_localization_error'],
      str(list(merged.columns)))
check('   the predictions and the refined values land on the same rows',
      list(merged['pred_0']) == [0.9, 0.8, 0.1]
      and list(merged['Refined_position_X']) == [0.1, 1.1, 2.1])
check('   FRAME and TRACK_ID keep the labeling table integer dtypes',
      merged['FRAME'].dtype == labeled['FRAME'].dtype
      and merged['TRACK_ID'].dtype == labeled['TRACK_ID'].dtype)
check('   the optional metrics are kept', list(merged['QUALITY']) == [7.0, 8.0, 9.0])
check('one side missing returns the other unchanged',
      ns['merge_track_tables'](labeled, None) is labeled
      and ns['merge_track_tables'](None, refined) is refined
      and ns['merge_track_tables'](None, None) is None)
# rows out of order: the positional path must not be taken
shuffled = refined.iloc[[2, 0, 1]].reset_index(drop=True)
merged2 = ns['merge_track_tables'](labeled, shuffled)
check('rows that do not line up are merged on TRACK_ID and FRAME instead',
      list(merged2['Refined_position_X']) == [0.1, 1.1, 2.1]
      and list(merged2['pred_0']) == [0.9, 0.8, 0.1],
      str(list(merged2['Refined_position_X'])))

# ---------------------------------------------------------------------------
print('')
print('5) end to end: two replicates, one summary, one table')
win2 = tk.Toplevel(ns['root'])
win2.withdraw()
files = sorted([os.path.join(DATASETS, f) for f in os.listdir(DATASETS)])
results = ns['run_batch'](win2, files, ns['BATCH_STAGES']['Batch Fitting'], dt=0.1,
                          nb_states=2, nb_iterations=1, nb_substeps=1,
                          fit_frame_len=5, label_frame_len=5, cell_dims=1.0,
                          LocErr_type='Fitted parameter', LocErr_input_name=[],
                          Optional_input_name=[], headers=HEADERS, max_dist=1.0,
                          remove_no_disps=True, min_length=5, max_length=15,
                          threshold=0.1, max_nb_states=50,
                          fusion_model='Multi-transition', save_folder=RESULTS,
                          batch_mode='Batch Fitting')
check('both replicates fitted', [r[1] for r in results] == ['done', 'done'])
check('the batch created the Results directory it was pointed at',
      os.path.isdir(RESULTS))
check('   and left the dataset folder untouched',
      sorted(os.listdir(DATASETS)) == ['rep1.csv', 'rep2.csv'],
      str(sorted(os.listdir(DATASETS))))
summary_file = os.path.join(RESULTS, ns['BATCH_SUMMARY_NAME'])
check('the summary file is written in the save folder', os.path.isfile(summary_file))
table = pd.read_csv(summary_file)
check('   with one row per replicate, named after the dataset files',
      len(table) == 2 and list(table['dataset']) == ['rep1.csv', 'rep2.csv'],
      str(list(table.get('dataset', []))))
check('   and the fitted parameters of the run',
      all(c in table.columns for c in ['likelihood', 'D0', 'D1', 'LocErr', 'equilibrium_F0']),
      str(list(table.columns)))
check('   and it is the only thing a fitting-only batch writes',
      sorted(os.listdir(RESULTS)) == [ns['BATCH_SUMMARY_NAME']],
      str(sorted(os.listdir(RESULTS))))

top = [w for w in win2.winfo_children() if isinstance(w, tk.Toplevel)
       and w.winfo_exists() and w.title() == 'ExTrack'][0]
message = str([c for c in top.winfo_children() if isinstance(c, tk.Label)][0].cget('text'))
check('the final window still opens on the batch report',
      message.startswith('Batch Fitting finished.'))
check('   and points at the summary file',
      ns['BATCH_SUMMARY_NAME'] in message, message.replace(os.linesep, ' | '))

trees = []
for frame in top.winfo_children():
    if isinstance(frame, ttk.Frame):
        trees += [c for c in frame.winfo_children() if isinstance(c, ttk.Treeview)]
check('the fitting results are shown as a table', len(trees) == 1)
if trees:
    tree = trees[0]
    rows = tree.get_children('')
    check('   one line per replicate', len(rows) == 2)
    check('   with the dataset names in the first column',
          [tree.item(r, 'values')[0] for r in rows] == ['rep1.csv', 'rep2.csv'],
          str([tree.item(r, 'values')[0] for r in rows]))
    check('   and the numbers rounded for reading',
          all(len(str(v)) <= 12 for r in rows for v in tree.item(r, 'values')[1:]),
          str(tree.item(rows[0], 'values')[1:4]))
    check('   the columns match the summary file',
          list(tree['columns']) == list(table.columns), str(list(tree['columns'])))

check('format_summary_cell keeps text and rounds numbers',
      ns['format_summary_cell']('rep1.csv') == 'rep1.csv'
      and ns['format_summary_cell'](0.123456789) == '0.1235'
      and ns['format_summary_cell'](np.nan) == 'nan')

# ---------------------------------------------------------------------------
shutil.rmtree(EXPERIMENT, ignore_errors=True)
shutil.rmtree(fake, ignore_errors=True)
if os.path.isfile(CONFIG):
    os.remove(CONFIG)
os.rmdir(os.path.dirname(CONFIG))

print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), '; '.join(fails[:6])))
    raise SystemExit(1)
print('all checks passed')
