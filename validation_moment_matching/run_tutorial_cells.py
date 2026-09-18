#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Tutorial_ExTrack.ipynb cell by cell and report each one separately.

nbconvert gives one verdict for the whole notebook and stops at the first cell
that fails or times out, which says nothing about the cells after it. This runs
every code cell in one shared namespace, exactly like a kernel would, and prints
the time and the outcome of each, so a slow cell and a broken cell can be told
apart. A cell that raises is reported and the run continues.

    python run_tutorial_cells.py                  # all cells
    python run_tutorial_cells.py --skip 50        # skip the heaviest ones
    python run_tutorial_cells.py --only 15 38 40  # just these
    python run_tutorial_cells.py --no-numba       # the numpy path, for comparison
"""

import argparse
import io
import json
import os
import sys
import time
import traceback

import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TUTORIALS = os.path.join(REPO, 'Tutorials')
NOTEBOOK = os.path.join(TUTORIALS, 'Tutorial_ExTrack.ipynb')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip', type=int, nargs='*', default=[])
    ap.add_argument('--only', type=int, nargs='*', default=None)
    ap.add_argument('--no-numba', action='store_true')
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()

    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    os.chdir(TUTORIALS)                       # the notebook uses relative paths

    import extrack
    assert os.path.dirname(extrack.__file__) == os.path.join(REPO, 'extrack'), \
        'the installed extrack is being imported instead of the working tree'
    extrack.tracking.set_numba(not args.no_numba and 'auto' or False,
                               threads=args.threads)
    print('extrack from %s' % os.path.dirname(extrack.__file__))
    print(extrack.tracking.numba_status())
    print('')

    nb = json.load(io.open(NOTEBOOK, encoding='utf-8'))
    cells = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells'])
             if c['cell_type'] == 'code' and ''.join(c['source']).strip()]

    ns = {'__name__': '__main__'}
    results = []
    t_all = time.time()
    for i, src in cells:
        if args.only is not None and i not in args.only:
            continue
        if i in args.skip:
            print('  cell %-3d SKIPPED' % i)
            results.append((i, 'skipped', 0.0, ''))
            continue
        t0 = time.time()
        try:
            exec(compile(src, '<cell %d>' % i, 'exec'), ns)
            dt = time.time() - t0
            status, detail = 'ok', ''
        except Exception as exc:
            dt = time.time() - t0
            status = 'FAILED'
            detail = '%s: %s' % (type(exc).__name__, exc)
            traceback.print_exc()
        first = src.strip().split('\n')[0][:58]
        print('  cell %-3d %-7s %8.2f s   %s' % (i, status, dt, first))
        sys.stdout.flush()
        results.append((i, status, dt, detail))

    print('')
    print('%d cells, %d ok, %d failed, %d skipped, %.0f s total'
          % (len(results), sum(r[1] == 'ok' for r in results),
             sum(r[1] == 'FAILED' for r in results),
             sum(r[1] == 'skipped' for r in results), time.time() - t_all))
    slow = sorted([r for r in results if r[1] == 'ok'], key=lambda r: -r[2])[:5]
    print('slowest cells: ' + ', '.join('%d (%.0f s)' % (r[0], r[2]) for r in slow))
    for i, status, dt, detail in results:
        if status == 'FAILED':
            print('  cell %d: %s' % (i, detail))
    return 1 if any(r[1] == 'FAILED' for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
