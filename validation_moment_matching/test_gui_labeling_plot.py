#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The State Labeling window's plot menu alignment, and the prediction plot no
longer showing the same track several times.

The duplication had a precise cause: each of the 8 x 8 = 64 subplot slots drew
`ID = np.random.randint(len(track_list))` independently, i.e. sampled WITH
replacement -- guaranteed repeats when fewer than 64 tracks are loaded
(example_tracks.csv loads 35 at the GUI's default lengths) and near-certain
repeats otherwise (birthday effect, P > 0.98 even at 500 tracks). The checks
here record the actual plt.plot(':k') calls the GUI makes:

  1. the 'Plot labeled tracks' menu is gridded like every other hyperparameter
     value: plain column 1, no sticky east, no extra padding;
  2. with 10 tracks the plot draws exactly 10 track lines (it used to draw 64);
  3. with 100 tracks it draws 64 lines and every plotted track is distinct
     (after removing the grid offset, which mean-centering cancels).
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

# ---------------------------------------------------------------------------
print('1) the plot menu sits in the hyperparameter value column')
win = tk.Toplevel(ns['root'])
win.withdraw()
ns['create_prediction_window'](win, os.path.join(REPO, 'Tutorials', 'example_tracks.csv'),
                               HERE, 5, 15, 'Fitted parameter', [], [],
                               ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],
                               1.0, True)
menus = [w for w in win.winfo_children() if w.winfo_class() == 'TMenubutton']
draw = [m for m in menus if int(m.grid_info()['row']) == 8]
check('the Yes/No menu is on row 8', len(draw) == 1)
if draw:
    gi = draw[0].grid_info()
    check('   plain column 1: no sticky east, no extra padding',
          int(gi['column']) == 1 and str(gi['sticky']) == ''
          and int(gi['padx']) == 0 and int(gi['pady']) == 0, str(dict(gi)))
    entries = [w.grid_info() for w in win.winfo_children()
               if w.winfo_class() == 'TEntry' and w.grid_info()
               and int(w.grid_info()['row']) in (0, 2, 3)]
    check('   exactly like the value entries above it',
          all(int(g['column']) == 1 and str(g['sticky']) == ''
              and int(g['padx']) == 0 for g in entries))

# ---------------------------------------------------------------------------
print('')
print('2) the prediction plot draws each track at most once')


def make_tracks(n, T=8, seed=6):
    rng = np.random.default_rng(seed)
    d = np.array([0.0, np.sqrt(2 * 1.0 * 0.1)])
    st = np.zeros((n, T), dtype=int)
    st[:, 0] = rng.integers(0, 2, n)
    for t in range(1, T):
        flip = rng.random(n) < 0.1
        st[:, t] = np.where(flip, 1 - st[:, t - 1], st[:, t - 1])
    var = (d[st[:, :-1]] ** 2 + d[st[:, 1:]] ** 2) / 2
    pos = np.concatenate([np.zeros((n, 1, 2)),
                          np.cumsum(rng.normal(size=(n, T - 1, 2)) * np.sqrt(var)[:, :, None], 1)], 1)
    return {str(T): pos + rng.normal(0, 0.03, pos.shape)}


def run_with_recorder(n_tracks):
    lines = []
    real_plot = ns['plt'].plot

    def rec(*a, **k):
        if len(a) >= 3 and a[2] == ':k':
            lines.append((np.array(a[0]), np.array(a[1])))
        return real_plot(*a, **k)

    ns['plt'].plot = rec
    save = os.path.join(HERE, 'gui_plot_test.csv')
    try:
        ns['run_predictions'](win, make_tracks(n_tracks), None, {}, dt=0.1,
                              nb_states=2, frame_len=6, cell_dims=1.0,
                              LocErr_type='Fitted parameter', input_LocErr=None,
                              threshold=0.1, max_nb_states=50, savepath=save,
                              Draw_plot='Yes', fusion_model='Multi-transition')
    finally:
        ns['plt'].plot = real_plot
        ns['plt'].close('all')
        for w in win.winfo_children():
            if isinstance(w, tk.Toplevel) and w.winfo_exists() and w.title() == 'ExTrack':
                w.destroy()
        if os.path.isfile(save):
            os.remove(save)
    return lines


lines = run_with_recorder(10)
check('10 loaded tracks give exactly 10 plotted tracks (was 64 with repeats)',
      len(lines) == 10, 'plotted %d' % len(lines))

lines = run_with_recorder(100)
check('100 loaded tracks fill the 64 slots', len(lines) == 64,
      'plotted %d' % len(lines))
centered = [np.stack([x - x.mean(), y - y.mean()], 1) for x, y in lines]
dup = 0
for a in range(len(centered)):
    for b in range(a + 1, len(centered)):
        if centered[a].shape == centered[b].shape and np.allclose(centered[a], centered[b]):
            dup += 1
check('every plotted track is distinct once the grid offset is removed',
      dup == 0, '%d duplicated pairs' % dup)

print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), '; '.join(fails[:6])))
    raise SystemExit(1)
print('all checks passed')
