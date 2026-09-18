#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The initial/equilibrium fraction rows of the Parameter Window, and the error
panels on dataset loading.

  1. the Parameter Window labels the fractions as INITIAL fractions, and carries
     a read-only row with the equilibrium fractions implied by the transition
     entries -- computed with the model's own convention (1 - exp(-rate)),
     checked against an independent matrix-power computation, updated live as
     the entries are edited, and degrading to '?' on a non-numerical entry;
  2. a wrong dataset path, a directory with no csv, a file with wrong headers,
     and length filters that keep no track all raise the GUI's error panel (on
     all four analysis windows) instead of a console traceback, while a good
     dataset still opens without one.
"""

import os
import runpy
import sys
import tempfile

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

CSV = os.path.join(REPO, 'Tutorials', 'example_tracks.csv')
HEADERS = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID']

# ---------------------------------------------------------------------------
print('1) the Parameter Window')
pw = ns['ParameterWindow'](ns['root'], 2)
pw.window.withdraw()
labels = [str(w.cget('text')) for w in pw.window.winfo_children()
          if w.winfo_class() == 'TLabel']
check("fractions are labeled 'Initial fractions:'",
      'Initial fractions:' in labels and 'Fractions:' not in labels)
check("an 'Equilibrium fractions:' row exists", 'Equilibrium fractions:' in labels)
check("both rows carry a '?' info cell",
      'initial_fractions' in ns['HYPERPARAM_INFO']
      and 'equilibrium_fractions' in ns['HYPERPARAM_INFO']
      and sum(1 for w in pw.window.winfo_children()
              if w.winfo_class() == 'TButton' and str(w.cget('text')) == '?') == 2)

eq0 = [float(l.cget('text')) for l in pw.equilibrium_labels]
check('symmetric default rates give a [0.5, 0.5] equilibrium',
      np.allclose(eq0, [0.5, 0.5], atol=1e-4), str(eq0))

# edit the rates: r01 = 0.2, r10 = 0.05, with the model's 1 - exp(-r) convention
pw.transition_entries[1].delete(0, tk.END)
pw.transition_entries[1].insert(0, '0.2')
pw.transition_entries[2].delete(0, tk.END)
pw.transition_entries[2].insert(0, '0.05')
pw.window.deiconify()
pw.window.update()
pw.transition_entries[2].focus_set()
pw.transition_entries[2].event_generate('<KeyRelease>')
pw.window.update()
if [float(l.cget('text')) for l in pw.equilibrium_labels] == eq0:
    # Tk does not deliver synthetic key events to every test window; the binding
    # itself is asserted below, so exercise the bound handler directly instead
    pw.update_equilibrium()
pw.window.withdraw()
p01, p10 = 1 - np.exp(-0.2), 1 - np.exp(-0.05)
expected = np.array([p10, p01]) / (p01 + p10)
got = [float(l.cget('text')) for l in pw.equilibrium_labels]
check('editing a rate updates the row live, with the 1 - exp(-rate) convention',
      np.allclose(got, expected, atol=1e-4),
      'got %s expected %s' % (got, np.round(expected, 4)))
check('the update is bound to the transition entries',
      all(pw.transition_entries[k].bind('<KeyRelease>') for k in range(4)))

pw.transition_entries[1].delete(0, tk.END)
pw.transition_entries[1].insert(0, 'abc')
pw.update_equilibrium()
check("a non-numerical entry degrades the row to '?'",
      all(str(l.cget('text')) == '?' for l in pw.equilibrium_labels))
pw.window.destroy()

# the helper itself, against an independent matrix-power computation, 3 states
rates = np.array([[0.0, 0.30, 0.02], [0.05, 0.0, 0.10], [0.20, 0.01, 0.0]])
eq = ns['equilibrium_fractions'](rates)
T = 1 - np.exp(-rates)
T[np.arange(3), np.arange(3)] = 0
T[np.arange(3), np.arange(3)] = 1 - T.sum(1)
brute = (np.ones(3) / 3) @ np.linalg.matrix_power(T, 200000)
check('equilibrium_fractions matches T**200000 at 3 states and sums to 1',
      np.allclose(eq, brute, atol=1e-10) and abs(eq.sum() - 1) < 1e-12,
      'max diff %.1e' % np.abs(eq - brute).max())

pw3 = ns['ParameterWindow'](ns['root'], 3)
pw3.window.withdraw()
eq3 = [float(l.cget('text')) for l in pw3.equilibrium_labels]
check('3-state window shows 3 equilibrium values summing to 1',
      len(eq3) == 3 and abs(sum(eq3) - 1) < 2e-3, str(eq3))
pw3.window.destroy()
ns['get_new_params'](2)

# ---------------------------------------------------------------------------
print('')
print('2) the error panels on dataset loading')


def error_texts(win):
    out = []
    for w in win.winfo_children():
        if isinstance(w, tk.Toplevel) and w.winfo_exists() and w.title() == 'Error':
            for c in w.winfo_children():
                if isinstance(c, tk.Text):
                    out.append(c.get('1.0', 'end'))
    return out


def attempt(maker, path, headers=HEADERS, min_len=5, max_len=15):
    win = tk.Toplevel(ns['root'])
    win.withdraw()
    ns[maker](win, path, HERE, min_len, max_len, 'Fitted parameter', [], [],
              headers, 1.0, True)
    return win, error_texts(win)


for maker in ['create_fitting_window', 'create_prediction_window',
              'create_lifetime_window', 'create_refinement_window']:
    short = maker.replace('create_', '').replace('_window', '')
    win, errs = attempt(maker, r'C:\does\not\exist_extrack_42.csv')
    check('%-10s : missing path raises the panel' % short,
          len(errs) == 1 and 'does not exist' in errs[0])

empty_dir = tempfile.mkdtemp(prefix='extrack_empty_')
win, errs = attempt('create_prediction_window', empty_dir)
check('a directory with no csv raises the panel',
      len(errs) == 1 and 'No csv file detected' in errs[0])
os.rmdir(empty_dir)

win, errs = attempt('create_prediction_window', CSV,
                    headers=['NOT_A_COLUMN', 'POSITION_Y', 'FRAME', 'TRACK_ID'])
check('wrong headers raise the panel, with the underlying error quoted',
      len(errs) == 1 and 'could not be read correctly' in errs[0]
      and 'Error:' in errs[0])

win, errs = attempt('create_fitting_window', CSV,
                    headers=['NOT_A_COLUMN', 'POSITION_Y', 'FRAME', 'TRACK_ID'])
check('the fitting window shows the panel too (it used to print and crash)',
      len(errs) == 1 and 'could not be read correctly' in errs[0])

win, errs = attempt('create_prediction_window', CSV, min_len=500, max_len=501)
check('length filters that keep no track raise the panel',
      len(errs) == 1 and ('No track was loaded' in errs[0]
                          or 'could not be read correctly' in errs[0]))

win, errs = attempt('create_prediction_window', CSV)
check('a good dataset still opens with no error panel', len(errs) == 0)
has_widgets = any(w.winfo_class() == 'TMenubutton' for w in win.winfo_children())
check('   and the analysis window is fully built', has_widgets)

print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), '; '.join(fails[:6])))
    raise SystemExit(1)
print('all checks passed')
