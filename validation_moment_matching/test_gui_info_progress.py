#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The '?' hyperparameter info cells and the transient progress windows of the GUI.

Runs the real GUI module with `mainloop` stubbed and checks

  1. every analysis window carries one '?' cell per hyperparameter, the fusion
     model included (9 on the fitting window, 8 on labeling, 5 on lifetime
     histograms, 6 on refinement),
     each to the right of its value, and clicking one expands the explanation
     next to it and clicking again collapses it;
  2. the progress windows: while an analysis computes, a window reading
     '... on-going...' is up (checked from inside the computation itself, by a
     spy planted on the backend), and at the end the same window reads
     '... finished.' with the save path and an OK button that closes it;
  3. an analysis refused by a guard closes its progress window without claiming
     it finished, and a failing analysis reports '... failed:' instead.
"""

import os
import runpy
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np

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

CSV = os.path.join(REPO, 'Tutorials', 'example_tracks.csv')
HEADERS = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID']


def build(maker):
    win = tk.Toplevel(ns['root'])
    win.withdraw()
    ns[maker](win, CSV, HERE, 5, 15, 'Fitted parameter', [], [], HEADERS, 1.0, True)
    return win


def qmarks(win):
    return [w for w in win.winfo_children()
            if w.winfo_class() == 'TButton' and str(w.cget('text')) == '?']


def extrack_toplevels(win):
    return [w for w in win.winfo_children()
            if isinstance(w, tk.Toplevel) and w.winfo_exists()
            and w.title() == 'ExTrack']


def label_of(top):
    return [w for w in top.winfo_children() if isinstance(w, tk.Label)][0]


print('1) the info cells')
wins = {}
for maker, n_expected in [('create_fitting_window', 9), ('create_prediction_window', 8),
                          ('create_lifetime_window', 5), ('create_refinement_window', 6)]:
    wins[maker] = build(maker)
    qs = qmarks(wins[maker])
    check('%-26s has %d info cells' % (maker.replace('create_', '').replace('_window', ''),
                                       n_expected), len(qs) == n_expected,
          'found %d' % len(qs))
    ok_right = all(int(q.grid_info()['column']) == 2 for q in qs)
    check('   all sit in the column right of the values', ok_right)

win = wins['create_fitting_window']
q0 = sorted(qmarks(win), key=lambda w: int(w.grid_info()['row']))[0]
before = set(w for w in win.winfo_children() if w.winfo_manager() == 'grid')
q0.invoke()
shown = [w for w in win.winfo_children() if w.winfo_manager() == 'grid'
         and w not in before]
check('clicking ? expands exactly one explanation', len(shown) == 1)
if shown:
    lbl = shown[0]
    check('   the explanation is the nb_states text, next to its row',
          str(lbl.cget('text')) == ns['HYPERPARAM_INFO']['nb_states']
          and int(lbl.grid_info()['row']) == int(q0.grid_info()['row'])
          and int(lbl.grid_info()['column']) == 3)
    q0.invoke()
    check('   clicking again collapses it', lbl.winfo_manager() == '')

# ---------------------------------------------------------------------------
print('')
print('2) the progress windows')
rng = np.random.default_rng(4)
d = np.array([0.0, np.sqrt(2 * 1.0 * 0.1)])
st = np.zeros((100, 8), dtype=int)
st[:, 0] = rng.integers(0, 2, 100)
for t in range(1, 8):
    flip = rng.random(100) < 0.1
    st[:, t] = np.where(flip, 1 - st[:, t - 1], st[:, t - 1])
var = (d[st[:, :-1]] ** 2 + d[st[:, 1:]] ** 2) / 2
pos = np.concatenate([np.zeros((100, 1, 2)),
                      np.cumsum(rng.normal(size=(100, 7, 2)) * np.sqrt(var)[:, :, None], 1)], 1)
tracks = {'8': pos + rng.normal(0, 0.03, pos.shape)}

seen = {}
real_predict = extrack.tracking.predict_Bs


def spy_predict(*a, **k):
    tops = extrack_toplevels(win)
    seen['during'] = str(label_of(tops[0]).cget('text')) if tops else '(no window)'
    return real_predict(*a, **k)


extrack.tracking.predict_Bs = spy_predict
save = os.path.join(HERE, 'gui_progress_preds.csv')
try:
    ns['run_predictions'](win, tracks, None, {}, dt=0.1, nb_states=2, frame_len=6,
                          cell_dims=1.0, LocErr_type='Fitted parameter',
                          input_LocErr=None, threshold=0.1, max_nb_states=50,
                          savepath=save, Draw_plot='No',
                          fusion_model='Multi-transition')
finally:
    extrack.tracking.predict_Bs = real_predict

check("window read 'State labeling on-going...' while the backend was computing",
      seen.get('during') == 'State labeling on-going...', repr(seen.get('during')))
tops = extrack_toplevels(win)
check('one progress window left at the end', len(tops) == 1)
if tops:
    txt = str(label_of(tops[0]).cget('text'))
    check("it announces 'State labeling finished.' with the save path",
          txt.startswith('State labeling finished.') and save in txt)
    ok = [w for w in tops[0].winfo_children()
          if w.winfo_class() == 'TButton' and str(w.cget('text')) == 'OK']
    check('it carries an OK button that closes it', len(ok) == 1)
    if ok:
        ok[0].invoke()
        check('   OK destroys the window', not tops[0].winfo_exists())

calls = {'n': 0}
real_fit = extrack.tracking.param_fitting


def spy_fit(*a, **k):
    calls['n'] += 1
    return real_fit(*a, **k)


extrack.tracking.param_fitting = spy_fit
try:
    save_fit = os.path.join(HERE, 'gui_progress_fit.csv')
    ns['run_fitting'](win, tracks, dt=0.1, nb_states=2, nb_iterations=1,
                      nb_substeps=1, frame_len=6, cell_dims=1.0,
                      LocErr_type='Fitted parameter', input_LocErr=None,
                      threshold=0.1, max_nb_states=50, savepath=save_fit,
                      fusion_model='Mono-transition')
    tops = extrack_toplevels(win)
    check("run_fitting ends on 'Fitting finished.'",
          len(tops) == 1 and str(label_of(tops[0]).cget('text')).startswith('Fitting finished.'))
    for t in tops:
        t.destroy()

    print('')
    print('3) refusal and failure')
    calls['n'] = 0
    ns['run_fitting'](win, tracks, dt=0.1, nb_states=2, nb_iterations=1,
                      nb_substeps=2, frame_len=6, cell_dims=1.0,
                      LocErr_type='Fitted parameter', input_LocErr=None,
                      threshold=0.1, max_nb_states=50,
                      savepath=os.path.join(HERE, 'gui_progress_guard.csv'),
                      fusion_model='Mono-transition')
    check('a refused analysis closes its progress window without claiming success',
          calls['n'] == 0 and len(extrack_toplevels(win)) == 0)
finally:
    extrack.tracking.param_fitting = real_fit


def boom(*a, **k):
    raise ValueError('backend exploded on purpose')


extrack.tracking.predict_Bs = boom
raised = False
try:
    ns['run_predictions'](win, tracks, None, {}, dt=0.1, nb_states=2, frame_len=6,
                          cell_dims=1.0, LocErr_type='Fitted parameter',
                          input_LocErr=None, threshold=0.1, max_nb_states=50,
                          savepath=save, Draw_plot='No',
                          fusion_model='Multi-transition')
except ValueError:
    raised = True
finally:
    extrack.tracking.predict_Bs = real_predict
tops = extrack_toplevels(win)
check("a failing analysis reports 'State labeling failed:' and re-raises",
      raised and len(tops) == 1
      and str(label_of(tops[0]).cget('text')).startswith('State labeling failed:')
      and 'exploded on purpose' in str(label_of(tops[0]).cget('text')))

for f in ['gui_progress_preds.csv', 'gui_progress_fit.csv', 'gui_progress_guard.csv']:
    p = os.path.join(HERE, f)
    if os.path.isfile(p):
        os.remove(p)

print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), '; '.join(fails[:6])))
    raise SystemExit(1)
print('all checks passed')
