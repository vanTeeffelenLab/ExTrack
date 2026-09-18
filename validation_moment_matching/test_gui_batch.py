#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The three batch analysis types of the GUI.

A temp folder is built with two good csv files, one corrupt csv and the
tutorial's TrackMate xml, then:

  1. the main window's dropdown offers the three batch types, and the batch
     window refuses a non-folder path and an empty folder with the error panel;
  2. 'Batch All' (driven through run_batch) processes every readable file --
     two output csvs per file, one merged track file (predictions + refined
     positions) and the lifetime histograms -- reports the corrupt file as
     failed without stopping, and restores the global parameters afterwards;
  3. 'Batch Fitting' (driven through the window's own Start button) writes the
     batch summary and nothing per file;
  4. mono-transition + substeps 2 is refused before anything runs.
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

import tkinter as tk               # noqa: E402
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


FOLDER = tempfile.mkdtemp(prefix='extrack_batch_')
write_csv(os.path.join(FOLDER, 'set_a.csv'), 30, 7, 1)
write_csv(os.path.join(FOLDER, 'set_b.csv'), 25, 8, 2)
with open(os.path.join(FOLDER, 'corrupt.csv'), 'w') as f:
    f.write('this is not, a valid, tracks file\n1,2,3\n')
shutil.copy(os.path.join(REPO, 'Tutorials', 'example_tracks.xml'),
            os.path.join(FOLDER, 'example_tracks.xml'))

# how many tracks the xml yields at these lengths decides what to expect of it
try:
    xml_tracks, _, _ = extrack.readers.read_trackmate_xml(
        os.path.join(FOLDER, 'example_tracks.xml'), lengths=np.arange(5, 16),
        dist_th=1.0, frames_boundaries=[-np.inf, np.inf], remove_no_disp=True,
        opt_metrics_names=[], opt_metrics_types=[])
    xml_ok = sum(len(v) for v in xml_tracks.values()) > 0
except Exception:
    xml_ok = False
print('(the xml yields tracks at lengths 5-15: %s)' % xml_ok)

# ---------------------------------------------------------------------------
print('')
print('1) dropdown entries and path guards')
menu = ns['analysis_type_dropdown']['menu']
entries = [menu.entrycget(i, 'label') for i in range(menu.index('end') + 1)]
check('the dropdown offers the three batch types',
      all(e in entries for e in ['Batch Fitting', 'Batch Fitting + Labeling', 'Batch All'])
      and len(entries) == 7, str(entries))
check('BATCH_STAGES runs the advertised stages',
      ns['BATCH_STAGES'] == {'Batch Fitting': ['fitting'],
                             'Batch Fitting + Labeling': ['fitting', 'labeling'],
                             'Batch All': ['fitting', 'labeling', 'histogram', 'refinement']})


def error_texts(win):
    out = []
    for w in win.winfo_children():
        if isinstance(w, tk.Toplevel) and w.winfo_exists() and w.title() == 'Error':
            for c in w.winfo_children():
                if isinstance(c, tk.Text):
                    out.append(c.get('1.0', 'end'))
    return out


win = tk.Toplevel(ns['root'])
win.withdraw()
ns['create_batch_window'](win, os.path.join(FOLDER, 'set_a.csv'), FOLDER, 5, 15,
                          'Fitted parameter', [], [], HEADERS, 1.0, True, 'Batch All')
errs = error_texts(win)
check('a file path (not a folder) raises the panel',
      len(errs) == 1 and 'FOLDER' in errs[0])

empty = tempfile.mkdtemp(prefix='extrack_batch_empty_')
win2 = tk.Toplevel(ns['root'])
win2.withdraw()
ns['create_batch_window'](win2, empty, empty, 5, 15, 'Fitted parameter', [], [],
                          HEADERS, 1.0, True, 'Batch All')
check('an empty folder raises the panel',
      len(error_texts(win2)) == 1 and 'No csv or xml file' in error_texts(win2)[0])
os.rmdir(empty)

# ---------------------------------------------------------------------------
print('')
print('2) Batch All over the folder')
win3 = tk.Toplevel(ns['root'])
win3.withdraw()
files = sorted([os.path.join(FOLDER, f) for f in os.listdir(FOLDER)])
before = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in ns['params'].items()}
results = ns['run_batch'](win3, files, ns['BATCH_STAGES']['Batch All'], dt=0.1,
                          nb_states=2, nb_iterations=1, nb_substeps=1,
                          fit_frame_len=5, label_frame_len=5, cell_dims=1.0,
                          LocErr_type='Fitted parameter', LocErr_input_name=[],
                          Optional_input_name=[], headers=HEADERS, max_dist=1.0,
                          remove_no_disps=True, min_length=5, max_length=15,
                          threshold=0.1, max_nb_states=50,
                          fusion_model='Multi-transition', save_folder=FOLDER,
                          batch_mode='Batch All')
status = {name: st for name, st, _ in results}
SUFFIXES = [ns['BATCH_TRACKS_SUFFIX'], '_lifetime_histograms.csv']
GONE = ['_fitting_results.csv', '_labeling.csv', '_refined_positions.csv']
for stem in ['set_a', 'set_b']:
    outs = [os.path.isfile(os.path.join(FOLDER, stem + sfx)) for sfx in SUFFIXES]
    check('%s.csv: both stage outputs written and reported done' % stem,
          all(outs) and status.get(stem + '.csv') == 'done', str(outs))
    check('   and no per-stage file beside them',
          not any(os.path.isfile(os.path.join(FOLDER, stem + sfx)) for sfx in GONE))

# the merged track file carries the predictions AND the refined positions
merged = pd.read_csv(os.path.join(FOLDER, 'set_a' + ns['BATCH_TRACKS_SUFFIX']))
check('the track file has positions, frames and track IDs',
      all(c in merged.columns for c in ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID']),
      str(list(merged.columns)))
check('   the state predictions',
      [c for c in merged.columns if c.startswith('pred_')] == ['pred_0', 'pred_1'])
check('   and the refined positions with their localization error',
      all(c in merged.columns for c in ['Refined_position_X', 'Refined_position_Y',
                                        'Refined_localization_error']))
check('   one row per detection of every track, no index column',
      len(merged) > 0 and len(merged) == merged['TRACK_ID'].nunique() * 7
      and not any(str(c).startswith('Unnamed') for c in merged.columns),
      '%d rows, %d tracks' % (len(merged), merged['TRACK_ID'].nunique()))
check('   the refined positions sit near the raw ones',
      float(np.abs(merged['POSITION_X'] - merged['Refined_position_X']).max()) < 1.0,
      '%.3f' % float(np.abs(merged['POSITION_X'] - merged['Refined_position_X']).max()))
check('   and the predictions are a distribution over the states',
      bool(np.allclose(merged['pred_0'] + merged['pred_1'], 1.0)))

check('the corrupt file is reported failed and the batch continued',
      status.get('corrupt.csv') == 'failed'
      and not os.path.isfile(os.path.join(FOLDER, 'corrupt' + ns['BATCH_TRACKS_SUFFIX'])))
check('the xml file was %s' % ('processed' if xml_ok else 'reported failed'),
      status.get('example_tracks.xml') == ('done' if xml_ok else 'failed'))
if xml_ok:
    check('   with its 2 outputs',
          all(os.path.isfile(os.path.join(FOLDER, 'example_tracks' + sfx))
              for sfx in SUFFIXES))

after = ns['params']
same = all(np.array_equal(before[k], after[k]) if isinstance(before[k], np.ndarray)
           else before[k] == after[k] for k in before) and set(before) == set(after)
check('the global parameters are restored after the batch', same)

tops = [w for w in win3.winfo_children() if isinstance(w, tk.Toplevel)
        and w.winfo_exists() and w.title() == 'ExTrack']
txt = str([c for c in tops[0].winfo_children() if isinstance(c, tk.Label)][0].cget('text')) if tops else ''
nb_done = sum(1 for s in status.values() if s == 'done')
check("the progress window announces 'Batch All finished.' with the counts",
      len(tops) == 1 and txt.startswith('Batch All finished.')
      and ('%d/%d files' % (nb_done, len(files))) in txt and 'corrupt.csv' in txt)
for t in tops:
    t.destroy()

# ---------------------------------------------------------------------------
print('')
print('3) Batch Fitting through the window itself')
FOLDER2 = tempfile.mkdtemp(prefix='extrack_batch2_')
write_csv(os.path.join(FOLDER2, 'only.csv'), 25, 7, 3)
win4 = tk.Toplevel(ns['root'])
win4.withdraw()
ns['create_batch_window'](win4, FOLDER2, FOLDER2, 5, 15, 'Fitted parameter', [], [],
                          HEADERS, 1.0, True, 'Batch Fitting')
check('the batch window builds (files label + Start button)',
      any('1 files detected' in str(w.cget('text')) for w in win4.winfo_children()
          if w.winfo_class() == 'TLabel')
      and any(str(w.cget('text')).startswith('Start batch')
              for w in win4.winfo_children() if w.winfo_class() == 'TButton'))
by_row = {}
for w in win4.winfo_children():
    gi = w.grid_info()
    if gi and w.winfo_class() == 'TEntry':
        by_row[int(gi['row'])] = w
by_row[2].delete(0, tk.END)
by_row[2].insert(0, '0.1')                     # frame time
by_row[9].delete(0, tk.END)
by_row[9].insert(0, '1')                       # one iteration
check('   no labeling window length row in fitting-only mode', 4 not in by_row)
start = [w for w in win4.winfo_children() if w.winfo_class() == 'TButton'
         and str(w.cget('text')).startswith('Start batch')][0]
start.invoke()
# the window proposes <savepath>/Results, which the batch creates on its way
RESULTS2 = os.path.join(FOLDER2, ns['BATCH_RESULTS_DIRNAME'])
check('Start runs the batch: the summary is written',
      os.path.isfile(os.path.join(RESULTS2, ns['BATCH_SUMMARY_NAME'])))
check('   and fitting alone leaves no per-file output',
      sorted(os.listdir(RESULTS2)) == [ns['BATCH_SUMMARY_NAME']],
      str(sorted(os.listdir(RESULTS2))))
one_row = pd.read_csv(os.path.join(RESULTS2, ns['BATCH_SUMMARY_NAME']))
check('   with the single replicate in it',
      len(one_row) == 1 and list(one_row['dataset']) == ['only.csv'])

# ---------------------------------------------------------------------------
print('')
print('4) the substeps guard')
res = ns['run_batch'](win4, [os.path.join(FOLDER2, 'only.csv')],
                      ns['BATCH_STAGES']['Batch Fitting'], dt=0.1, nb_states=2,
                      nb_iterations=1, nb_substeps=2, fit_frame_len=5,
                      label_frame_len=5, cell_dims=1.0,
                      LocErr_type='Fitted parameter', LocErr_input_name=[],
                      Optional_input_name=[], headers=HEADERS, max_dist=1.0,
                      remove_no_disps=True, min_length=5, max_length=15,
                      threshold=0.1, max_nb_states=50,
                      fusion_model='Mono-transition', save_folder=FOLDER2,
                      batch_mode='Batch Fitting')
check('mono-transition + substeps 2 is refused before any file runs',
      res is None and len(error_texts(win4)) == 1)

shutil.rmtree(FOLDER)
shutil.rmtree(FOLDER2)

print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), '; '.join(fails[:6])))
    raise SystemExit(1)
print('all checks passed')
