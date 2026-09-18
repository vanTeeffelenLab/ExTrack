#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The four single-dataset analyses of the GUI, driven the way a user drives them:
the main window's Next button opens the analysis window, its own Start button
runs the analysis, and the file it was told to write must exist and be readable.

This is the end-to-end counterpart of the other GUI tests, which drive the
run_* functions directly. It exists because the numerics under Position
Refinement changed (see test_refinement_multistate.py) and the refinement window
is the one path no other test exercised from the button.

Run on the tutorial dataset, so it also checks the headers the GUI ships with.
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

OUT = tempfile.mkdtemp(prefix='extrack_e2e_')
DATA = os.path.join(REPO, 'Tutorials', 'example_tracks.csv')


def widgets(win, cls):
    return [w for w in win.winfo_children() if w.winfo_class() == cls]


def entries_by_row(win):
    out = {}
    for w in widgets(win, 'TEntry'):
        gi = w.grid_info()
        if gi:
            out[int(gi['row'])] = w
    return out


def set_entry(entry, value):
    entry.delete(0, tk.END)
    entry.insert(0, str(value))


def error_panels(win):
    out = []
    for w in win.winfo_children():
        if isinstance(w, tk.Toplevel) and w.winfo_exists() and w.title() == 'Error':
            for c in w.winfo_children():
                if isinstance(c, tk.Text):
                    out.append(c.get('1.0', 'end'))
    return out


def progress_text(win):
    for w in win.winfo_children():
        if isinstance(w, tk.Toplevel) and w.winfo_exists() and w.title() == 'ExTrack':
            labels = [c for c in w.winfo_children() if isinstance(c, tk.Label)]
            if labels:
                return str(labels[0].cget('text'))
    return ''


def run_analysis(kind, savename, tweaks):
    """
    Go through the main window: set the path and the analysis type, click Next,
    then set the parameters and click Start. `tweaks` maps a grid row of the
    analysis window to the value to type into its entry.
    """
    set_entry(ns['path_entry'], DATA)
    set_entry(ns['min_length_entry'], 5)
    set_entry(ns['max_length_entry'], 15)
    ns['analysis_type_var'].set(kind)
    ns['LocErr_type_var'].set('Fitted parameter')
    set_entry(ns['LocErr_input_entry'], '')
    set_entry(ns['Optional_input_entry'], '')

    win = tk.Toplevel(ns['root'])
    win.withdraw()
    savepath = os.path.join(OUT, savename)
    builder = {'Model Fitting': 'create_fitting_window',
               'State Labeling': 'create_prediction_window',
               'State Lifetime Histogram': 'create_lifetime_window',
               'Position Refinement': 'create_refinement_window'}[kind]
    ns[builder](win, DATA, OUT, 5, 15, 'Fitted parameter', [], [],
                ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'], 1.0, True)
    if error_panels(win):
        return win, savepath, error_panels(win)[0]

    rows = entries_by_row(win)
    for row, value in tweaks.items():
        if row in rows:
            set_entry(rows[row], value)
    # the save path is the widest entry of the window; set it explicitly
    savepath_entry = max(rows.values(), key=lambda w: int(str(w.cget('width'))))
    set_entry(savepath_entry, savepath)

    # the run button is whichever one is not part of the furniture: the labels
    # are not uniform ("Start fitting", "Start state predictions",
    # "Compute lifetime histogram", "Start position refinement")
    furniture = {'Open Parameter Window', 'Browse', 'Previous', 'Other analyses', '?'}
    run = [w for w in widgets(win, 'TButton') if str(w.cget('text')) not in furniture]
    if len(run) != 1:
        return win, savepath, 'expected one run button, found %s' % [str(w.cget('text')) for w in run]
    run[0].invoke()
    return win, savepath, None


# ---------------------------------------------------------------------------
print('')
print('1) Model Fitting')
win, path, err = run_analysis('Model Fitting', 'fitting.csv',
                              {2: 0.02, 3: 5, 9: 1})     # dt, window length, iterations
check('the fitting window ran without an error panel', err is None, str(err))
check('   it wrote its results', os.path.isfile(path))
if os.path.isfile(path):
    table = pd.read_csv(path, index_col=0)
    check('   one row with a likelihood and the fitted parameters',
          len(table) == 1 and 'likelihood' in table.columns
          and all(c in table.columns for c in ['D0', 'D1', 'LocErr', 'F0'])
          and np.isfinite(table['likelihood'][0]),
          'likelihood %.2f, D0 %.4g, D1 %.4g, LocErr %.4g'
          % (table['likelihood'][0], table['D0'][0], table['D1'][0], table['LocErr'][0]))
check('   and announced that it finished', progress_text(win).startswith('Fitting finished'),
      progress_text(win).replace(os.linesep, ' | ')[:90])

print('')
print('2) State Labeling')
win, path, err = run_analysis('State Labeling', 'labeling.csv', {2: 0.02, 3: 5})
check('the labeling window ran without an error panel', err is None, str(err))
check('   it wrote its results', os.path.isfile(path))
if os.path.isfile(path):
    table = pd.read_csv(path, index_col=0)
    preds = [c for c in table.columns if c.startswith('pred_')]
    check('   with one prediction column per state, summing to 1',
          len(preds) == 2 and bool(np.allclose(table[preds].sum(1), 1.0)),
          '%d rows, columns %s' % (len(table), preds))
check('   and announced that it finished',
      progress_text(win).startswith('State labeling finished'),
      progress_text(win).replace(os.linesep, ' | ')[:90])

print('')
print('3) State Lifetime Histogram')
win, path, err = run_analysis('State Lifetime Histogram', 'lifetimes.csv', {2: 0.02})
check('the lifetime window ran without an error panel', err is None, str(err))
check('   it wrote its results', os.path.isfile(path))
if os.path.isfile(path):
    table = pd.read_csv(path, index_col=0)
    check('   with a segment length and one column per state',
          'Segment length' in table.columns
          and sum(c.startswith('State ') for c in table.columns) == 2,
          '%d rows, columns %s' % (len(table), list(table.columns)))
check('   and announced that it finished',
      progress_text(win).startswith('Lifetime histograms finished'),
      progress_text(win).replace(os.linesep, ' | ')[:90])

print('')
print('4) Position Refinement  (the path whose numerics changed)')
win, path, err = run_analysis('Position Refinement', 'refined.csv', {2: 0.02, 3: 5})
check('the refinement window ran without an error panel', err is None, str(err))
check('   it wrote its results', os.path.isfile(path))
if os.path.isfile(path):
    table = pd.read_csv(path, index_col=0)
    cols = ['Refined_position_X', 'Refined_position_Y', 'Refined_localization_error']
    check('   with the refined positions and their localization error',
          all(c in table.columns for c in cols), '%d rows' % len(table))
    if all(c in table.columns for c in cols):
        shift = np.sqrt((table['POSITION_X'] - table['Refined_position_X']) ** 2
                        + (table['POSITION_Y'] - table['Refined_position_Y']) ** 2)
        check('   every value finite',
              bool(np.isfinite(table[cols].values).all()))
        check('   the refined positions sit near the raw ones',
              float(shift.max()) < 1.0 and float(shift.mean()) > 0,
              'mean shift %.4f um, max %.4f um' % (shift.mean(), shift.max()))
        check('   the reported localization error is positive and sane',
              bool((table['Refined_localization_error'] > 0).all())
              and float(table['Refined_localization_error'].max()) < 1.0,
              'median %.4f um, max %.4f um'
              % (table['Refined_localization_error'].median(),
                 table['Refined_localization_error'].max()))
check('   and announced that it finished',
      progress_text(win).startswith('Position refinement finished'),
      progress_text(win).replace(os.linesep, ' | ')[:90])

# ---------------------------------------------------------------------------
print('')
print('5) the guards still fire')
win = tk.Toplevel(ns['root'])
win.withdraw()
ns['create_fitting_window'](win, os.path.join(OUT, 'nope.csv'), OUT, 5, 15,
                            'Fitted parameter', [], [],
                            ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'], 1.0, True)
check('a path that does not exist raises the error panel',
      len(error_panels(win)) == 1 and 'does not exist' in error_panels(win)[0])

win = tk.Toplevel(ns['root'])
win.withdraw()
ns['create_fitting_window'](win, DATA, OUT, 5, 15, 'Fitted parameter', [], [],
                            ['NOPE_X', 'NOPE_Y', 'FRAME', 'TRACK_ID'], 1.0, True)
check('headers that do not match raise the error panel', len(error_panels(win)) == 1,
      (error_panels(win)[0][:80].replace(os.linesep, ' ') if error_panels(win) else ''))

shutil.rmtree(OUT, ignore_errors=True)
if os.path.isfile(CONFIG):
    os.remove(CONFIG)
os.rmdir(os.path.dirname(CONFIG))

print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), '; '.join(fails[:8])))
    raise SystemExit(1)
print('all checks passed')
