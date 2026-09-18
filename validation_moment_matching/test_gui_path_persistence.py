#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Browse dialogs anchored to the current path, and the path surviving from one
GUI instance to the next.

The persistence is tested literally: the GUI module is executed twice in a row
(with a temp config file via EXTRACK_GUI_CONFIG), and the second instance must
come up with the path the first one saved. The dialogs are tested by stubbing
tkinter.filedialog and capturing the `initialdir` each Browse helper passes.
"""

import os
import runpy
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.chdir(REPO)

CONFIG = os.path.join(tempfile.mkdtemp(prefix='extrack_cfg_'), 'gui_config.json')
os.environ['EXTRACK_GUI_CONFIG'] = CONFIG

import tkinter as tk               # noqa: E402
from tkinter import filedialog     # noqa: E402
tk.Misc.mainloop = lambda self, n=0: None

fails = []


def check(name, ok, detail=''):
    print(('  OK   ' if ok else '  FAIL ') + name + ('   ' + detail if detail else ''))
    if not ok:
        fails.append(name)


def start_gui():
    ns = runpy.run_path(os.path.join(REPO, 'ExTrack_GUI.py'))
    ns['root'].withdraw()
    return ns


CSV = os.path.join(REPO, 'Tutorials', 'example_tracks.csv')

print('1) first instance: no config yet')
ns = start_gui()
check('the config file location honours EXTRACK_GUI_CONFIG',
      ns['CONFIG_PATH'] == CONFIG)
check('with no config the path field starts at the working directory',
      ns['path_entry'].get() == os.getcwd())

print('')
print('2) initialdir_from')
check('a file path anchors to its folder',
      ns['initialdir_from'](CSV) == os.path.dirname(CSV))
check('a folder anchors to itself',
      ns['initialdir_from'](os.path.dirname(CSV)) == os.path.dirname(CSV))
check('a missing file with an existing folder anchors to that folder',
      ns['initialdir_from'](os.path.join(REPO, 'Tutorials', 'nope.csv'))
      == os.path.join(REPO, 'Tutorials'))
check('garbage falls back to the home folder',
      ns['initialdir_from'](r'Z:\no\such\place\at\all.csv') == os.path.expanduser('~'))

print('')
print('3) the Browse dialogs start in the folder of the current path')
seen = {}
real_open = filedialog.askopenfilename
real_save = filedialog.asksaveasfilename
real_dir = filedialog.askdirectory
try:
    filedialog.askopenfilename = lambda **k: seen.update(open=k.get('initialdir')) or ''
    filedialog.asksaveasfilename = lambda **k: seen.update(save=k.get('initialdir')) or ''
    filedialog.askdirectory = lambda **k: seen.update(dir=k.get('initialdir')) or ''

    ns['path_entry'].delete(0, tk.END)
    ns['path_entry'].insert(tk.END, CSV)
    ns['browser']()
    check('the dataset Browse opens in the folder of the current path',
          seen.get('open') == os.path.dirname(CSV))
    check('   cancelling keeps the current path in the field',
          ns['path_entry'].get() == CSV)

    entry = tk.ttk.Entry(ns['root'])
    entry.insert(0, os.path.join(REPO, 'Tutorials', 'results.csv'))
    ns['browse_savepath'](entry)
    check('the save-path Browse opens in the folder of its entry',
          seen.get('save') == os.path.join(REPO, 'Tutorials'))
    entry.delete(0, tk.END)
    entry.insert(0, os.path.join(REPO, 'Tutorials'))
    ns['browse_savefolder'](entry)
    check('the save-folder Browse opens at its entry',
          seen.get('dir') == os.path.join(REPO, 'Tutorials'))

    # a selection through the dialog is saved for the next instance
    filedialog.askopenfilename = lambda **k: CSV
    ns['browser']()
    check('selecting a file through Browse saves it to the config',
          ns['load_gui_config']().get('last_path') == CSV)
finally:
    filedialog.askopenfilename = real_open
    filedialog.asksaveasfilename = real_save
    filedialog.askdirectory = real_dir

print('')
print('4) the path survives to the next instance of the GUI')
ns['root'].destroy()
ns2 = start_gui()
check('the second instance starts on the saved path', ns2['path_entry'].get() == CSV)

# a typed path (never browsed) is saved when moving on with Next
folder = os.path.join(REPO, 'Tutorials')
ns2['path_entry'].delete(0, tk.END)
ns2['path_entry'].insert(tk.END, folder)
ns2['analysis_type_var'].set('Batch Fitting')
ns2['open_analysis_window']()
check('a typed path is saved when clicking Next',
      ns2['load_gui_config']().get('last_path') == folder)
ns2['root'].destroy()

ns3 = start_gui()
check('and the third instance starts on it', ns3['path_entry'].get() == folder)

# a saved path whose file disappeared falls back to its folder
ns3['save_gui_config'](last_path=os.path.join(folder, 'deleted_since.csv'))
ns3['root'].destroy()
ns4 = start_gui()
check('a saved file that disappeared falls back to its folder',
      ns4['path_entry'].get() == folder)
ns4['root'].destroy()

os.remove(CONFIG)
os.rmdir(os.path.dirname(CONFIG))

print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), '; '.join(fails[:6])))
    raise SystemExit(1)
print('all checks passed')
