#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The fusion-model selector in the ExTrack GUI.

Runs the real GUI module with `mainloop` stubbed out, builds the actual fitting
window on the tutorial's example csv, and checks

  1. the module still executes end to end and the selector exists on the fitting
     window: the dropdown offers exactly Multi-transition / Mono-transition,
     defaults to Multi-transition, and the info box states both scalings;
  2. the widget rows: the selector and its box sit between the parameter fields
     and the save path, nothing overlaps;
  3. run_predictions wired through: Multi-transition reproduces a direct
     predict_Bs(sequence_scheme='sequences') call bitwise (the default is
     unchanged), Mono-transition reproduces sequence_scheme='ages';
  4. run_fitting wired through for both models (1 iteration, small data), and
     its guard: Mono-transition with nb_substeps > 1 must refuse to fit and
     raise the error window instead.
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
tk.Misc.mainloop = lambda self, n=0: None   # the GUI calls root.mainloop() at import

fails = []


def check(name, ok, detail=''):
    print(('  OK   ' if ok else '  FAIL ') + name + ('   ' + detail if detail else ''))
    if not ok:
        fails.append(name)


ns = runpy.run_path(os.path.join(REPO, 'ExTrack_GUI.py'))
ns['root'].withdraw()
import extrack                     # noqa: E402
assert os.path.dirname(extrack.__file__) == os.path.join(REPO, 'extrack')
extrack.tracking.set_numba('auto', threads=8)

print('1) module and selector')
check('module executed with the selector defined',
      'add_fusion_model_selector' in ns and 'FUSION_MODELS' in ns)
check('FUSION_MODELS maps to the sequence_scheme values',
      ns['FUSION_MODELS'] == {'Multi-transition': 'sequences',
                              'Mono-transition': 'ages'})
check("default stored in params", ns['params'].get('fusion_model') == 'Multi-transition')
info = ns['fusion_model_info']
check('info box states the multi-transition scaling',
      'nb_states ** window_length' in info and 'accurate' in info.lower())
check('info box states the mono-transition scaling',
      'window_length * nb_states**2' in info)
check("panel registered as HYPERPARAM_INFO['fusion_model']",
      ns['HYPERPARAM_INFO'].get('fusion_model') == info)

# ---------------------------------------------------------------------------
print('')
print('2) the real fitting window, built on Tutorials/example_tracks.csv')
win = tk.Toplevel(ns['root'])
win.withdraw()
ns['create_fitting_window'](win, os.path.join(REPO, 'Tutorials', 'example_tracks.csv'),
                            HERE, 5, 15, 'Fitted parameter', [], [],
                            ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],
                            1.0, True)
rows = {}
menu_widget = None
info_widget = None
for w in win.winfo_children():
    if w.winfo_class() == 'TMenubutton':
        menu_widget = w
    if isinstance(w, tk.Label) and 'Multi-transition:' in str(w.cget('text')):
        info_widget = w
    gi = w.grid_info()
    if not gi:
        continue
    rows.setdefault(int(gi['row']), []).append(w)

check('dropdown present on the fitting window', menu_widget is not None)
check('fusion panel exists and starts collapsed',
      info_widget is not None and info_widget.winfo_manager() == '')
fusion_q = [w for w in win.winfo_children()
            if w.winfo_class() == 'TButton' and str(w.cget('text')) == '?'
            and int(w.grid_info().get('row', -1)) == 9]
check("the fusion row carries its own '?' cell at column 2",
      len(fusion_q) == 1 and int(fusion_q[0].grid_info()['column']) == 2)
if fusion_q and info_widget is not None:
    fusion_q[0].invoke()
    check('clicking it expands the panel on the fusion row',
          info_widget.winfo_manager() == 'grid'
          and int(info_widget.grid_info()['row']) == 9
          and int(info_widget.grid_info()['column']) == 3
          and str(info_widget.cget('text')) == ns['fusion_model_info'])
    fusion_q[0].invoke()
    check('clicking again collapses it', info_widget.winfo_manager() == '')
if menu_widget is not None:
    var_name = str(menu_widget.cget('textvariable'))
    val = win.tk.globalgetvar(var_name)
    check('dropdown defaults to Multi-transition', val == 'Multi-transition')
    entries = [menu_widget['menu'].entrycget(i, 'label')
               for i in range(menu_widget['menu'].index('end') + 1)]
    check('dropdown offers exactly the two models',
          entries == ['Multi-transition', 'Mono-transition'], str(entries))
    check('selector sits on row 9, save path on row 11, run on row 12',
          int(menu_widget.grid_info()['row']) == 9
          and any(isinstance(w, tk.ttk.Entry) and 'saved_fitting' in w.get()
                  for w in rows.get(11, []))
          and any(w.winfo_class() == 'TButton' for w in rows.get(12, [])))
occupied = sorted(rows)
check('no two widgets share a (row, column) cell',
      all(len(set((int(w.grid_info()['row']), int(w.grid_info()['column'])) for w in ws))
          == len(ws) for ws in rows.values()))

# ---------------------------------------------------------------------------
print('')
print('3) run_predictions wired through, against direct predict_Bs calls')
rng = np.random.default_rng(2)
d = np.array([0.0, np.sqrt(2 * 1.0 * 0.1)])
st = np.zeros((120, 8), dtype=int)
st[:, 0] = rng.integers(0, 2, 120)
for t in range(1, 8):
    flip = rng.random(120) < 0.1
    st[:, t] = np.where(flip, 1 - st[:, t - 1], st[:, t - 1])
var = (d[st[:, :-1]] ** 2 + d[st[:, 1:]] ** 2) / 2
pos = np.concatenate([np.zeros((120, 1, 2)),
                      np.cumsum(rng.normal(size=(120, 7, 2)) * np.sqrt(var)[:, :, None], 1)], 1)
tracks = {'8': pos + rng.normal(0, 0.03, pos.shape)}

out = {}
for model in ['Multi-transition', 'Mono-transition']:
    save = os.path.join(HERE, 'gui_preds_%s.csv' % ns['FUSION_MODELS'][model])
    ns['run_predictions'](win, {k: v.copy() for k, v in tracks.items()}, None, {},
                          dt=0.1, nb_states=2, frame_len=6, cell_dims=1.0,
                          LocErr_type='Fitted parameter', input_LocErr=None,
                          threshold=0.1, max_nb_states=50, savepath=save,
                          Draw_plot='No', fusion_model=model)
    check('%-16s run_predictions wrote its csv' % model, os.path.isfile(save))
    lmfit_params = ns['params_to_lmfit_params'](ns['params'], 'Fitted parameter')
    out[model] = extrack.tracking.predict_Bs(
        {k: v.copy() for k, v in tracks.items()}, 0.1, lmfit_params, cell_dims=[1.0],
        nb_states=2, frame_len=6, max_nb_states=50, threshold=0.1, workers=1,
        input_LocErr=None, verbose=0, nb_max=1,
        sequence_scheme=ns['FUSION_MODELS'][model])['8']
    import pandas as pd
    got = pd.read_csv(save)
    cols = [c for c in got.columns if c.startswith('pred_')]
    saved = got[cols].values.reshape(out[model].shape[0], out[model].shape[1], -1)
    check('%-16s csv matches a direct sequence_scheme=%r call' %
          (model, ns['FUSION_MODELS'][model]),
          np.allclose(saved, out[model], atol=5e-7),
          'max diff %.2e (csv is rounded)' % np.abs(saved - out[model]).max())
check('the two models give different posteriors (both really ran)',
      not np.allclose(out['Multi-transition'], out['Mono-transition']),
      'mean |diff| %.2e' % np.abs(out['Multi-transition'] - out['Mono-transition']).mean())
check("selection persisted to params", ns['params']['fusion_model'] == 'Mono-transition')

# ---------------------------------------------------------------------------
print('')
print('4) run_fitting: both models fit, and the substeps guard refuses')
calls = {'n': 0}
real_fit = extrack.tracking.param_fitting


def spy(*a, **k):
    calls['n'] += 1
    calls['scheme'] = k.get('sequence_scheme')
    return real_fit(*a, **k)


extrack.tracking.param_fitting = spy
try:
    for model in ['Multi-transition', 'Mono-transition']:
        calls['n'] = 0
        save = os.path.join(HERE, 'gui_fit_%s.csv' % ns['FUSION_MODELS'][model])
        ns['run_fitting'](win, {k: v.copy() for k, v in tracks.items()}, dt=0.1,
                          nb_states=2, nb_iterations=1, nb_substeps=1, frame_len=6,
                          cell_dims=1.0, LocErr_type='Fitted parameter',
                          input_LocErr=None, threshold=0.1, max_nb_states=50,
                          savepath=save, fusion_model=model)
        check('%-16s run_fitting fitted with sequence_scheme=%r' %
              (model, ns['FUSION_MODELS'][model]),
              calls['n'] == 1 and calls['scheme'] == ns['FUSION_MODELS'][model]
              and os.path.isfile(save))

    calls['n'] = 0
    before = set(win.winfo_children()) | {win}
    ns['run_fitting'](win, tracks, dt=0.1, nb_states=2, nb_iterations=1,
                      nb_substeps=2, frame_len=6, cell_dims=1.0,
                      LocErr_type='Fitted parameter', input_LocErr=None,
                      threshold=0.1, max_nb_states=50,
                      savepath=os.path.join(HERE, 'gui_fit_guard.csv'),
                      fusion_model='Mono-transition')
    err_win = [w for w in win.winfo_children()
               if isinstance(w, tk.Toplevel) and w.title() == 'Error']
    check('Mono-transition + substeps 2 refused without fitting',
          calls['n'] == 0 and len(err_win) == 1
          and not os.path.isfile(os.path.join(HERE, 'gui_fit_guard.csv')))
    ns['run_fitting'](win, tracks, dt=0.1, nb_states=2, nb_iterations=1,
                      nb_substeps=2, frame_len=6, cell_dims=1.0,
                      LocErr_type='Fitted parameter', input_LocErr=None,
                      threshold=0.1, max_nb_states=50,
                      savepath=os.path.join(HERE, 'gui_fit_guard.csv'),
                      fusion_model='Multi-transition')
    check('Multi-transition + substeps 2 still fits (guard is specific)',
          calls['n'] == 1)
finally:
    extrack.tracking.param_fitting = real_fit

for f in ['gui_preds_sequences.csv', 'gui_preds_ages.csv', 'gui_fit_sequences.csv',
          'gui_fit_ages.csv', 'gui_fit_guard.csv']:
    p = os.path.join(HERE, f)
    if os.path.isfile(p):
        os.remove(p)

print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), '; '.join(fails[:6])))
    raise SystemExit(1)
print('all checks passed')
