#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Where each fusion puts the maximum of the log likelihood.

A fit mixes three things: the fusion error, the sampling noise of the data set and
the behaviour of the optimizer. This script removes the last two. It scans one
parameter at a time, the others held at the truth, and locates the maximum of the
total log likelihood by parabolic interpolation, for

    exact             no fusion at all (2**track_len sequences of states kept)
    legacy            ExTrack's historical fusion
    moment_matching   the fused variance also carries the spread of the means

on exactly the same data. The distance between an approximate maximum and the
exact one is the bias the fusion alone puts on that parameter.

The same run also reports the error on the total log likelihood at the truth.
"""

import argparse
import time

import numpy as np

import common
from common import tracking

MAX_NB_STATES = 10 ** 9


def log_likelihood(Cs, params, mode, frame_len, threshold=0.2):
    common.set_mode(mode)
    LocErr, ds, Fs, TrMat, pBL = tracking.extract_params(
        params, common.DT, 2, 1, None, Matrix_type=1)
    return np.asarray(tracking.Proba_Cs(Cs, LocErr[0], ds, Fs, TrMat, pBL, 0, common.BIG_FOV,
                                        1, frame_len, Cs.shape[1], threshold, MAX_NB_STATES))


def argmax_1d(xs, ys):
    """
    Parabolic interpolation of the maximum. When the sampled maximum sits on an
    edge the parabola is fitted on the 3 points at that edge and its vertex is
    returned even if it falls outside the scan, so that a maximum pinned at a
    boundary (D0 >= 0 here) still gives a signed comparison.
    """
    i = int(np.clip(np.argmax(ys), 1, len(xs) - 2))
    x0, x1, x2 = xs[i - 1], xs[i], xs[i + 1]
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    d = y0 - 2 * y1 + y2
    if d == 0:
        return x1
    return x1 + 0.5 * (x1 - x0) * (y0 - y2) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track-len', type=int, default=7)
    ap.add_argument('--frame-len', type=int, default=6)
    ap.add_argument('--nb-tracks', type=int, default=2000)
    ap.add_argument('--nb-points', type=int, default=21)
    args = ap.parse_args()
    T, L = args.track_len, args.frame_len

    print('%d tracks of %d points, frame_len = %d, LocErr = %.3f um, d0 = 0'
          % (args.nb_tracks, T, L, common.LOC_ERR))
    print('')
    print('A. error on the total log likelihood at the true parameters')
    print('   d1     p    |   legacy      moment m.  |  gain  |  rms per track legacy   moment m.   gain')
    err = []
    for d1 in common.D1_LENGTHS:
        for p in common.TRANSITION_PS:
            tracks, states, m = common.simulate_well_specified(d1, p, args.nb_tracks, T, 99)
            Cs = tracks[str(T)]
            params = common.params_from_truth(m)
            ex = log_likelihood(Cs, params, 'exact', L)
            lg = log_likelihood(Cs, params, 'legacy', L)
            mm = log_likelihood(Cs, params, 'moment_matching', L)
            a, b = (lg - ex), (mm - ex)
            err.append((d1, p, a.sum(), b.sum(), np.sqrt((a ** 2).mean()), np.sqrt((b ** 2).mean())))
            print('   %-5.3f  %-4.2f |  %+10.4f  %+10.4f  | %5.2f x |  %11.3e %11.3e  %5.2f x'
                  % (d1, p, a.sum(), b.sum(), abs(a.sum()) / max(abs(b.sum()), 1e-12),
                     err[-1][4], err[-1][5], err[-1][4] / err[-1][5]))
    err = np.array(err)
    print('   ' + '-' * 88)
    print('   mean       |  %10.4f  %10.4f  | %5.2f x |  %11.3e %11.3e  %5.2f x   (|.| for the totals)'
          % (np.abs(err[:, 2]).mean(), np.abs(err[:, 3]).mean(),
             np.abs(err[:, 2]).mean() / np.abs(err[:, 3]).mean(),
             err[:, 4].mean(), err[:, 5].mean(), err[:, 4].mean() / err[:, 5].mean()))
    print('   moment matching has the smaller rms error at %d/%d models, the smaller total at %d/%d'
          % ((err[:, 5] < err[:, 4]).sum(), len(err),
             (np.abs(err[:, 3]) < np.abs(err[:, 2])).sum(), len(err)))

    print('')
    print('B. bias of the maximum of the log likelihood, one parameter at a time')
    print('   (the other parameters stay at the truth: this is the fusion alone,')
    print('    no optimizer and no sampling noise in the comparison)')
    print('')
    rows = []
    for d1 in common.D1_LENGTHS:
        for p in common.TRANSITION_PS:
            tracks, states, m = common.simulate_well_specified(d1, p, args.nb_tracks, T, 99)
            Cs = tracks[str(T)]
            params = common.params_from_truth(m)
            scans = {
                'D1': m['Ds'][1] * np.linspace(0.8, 1.25, args.nb_points),
                'LocErr': common.LOC_ERR * np.linspace(0.9, 1.1, args.nb_points),
                'p01': -np.log(1 - p) * np.linspace(0.6, 1.6, args.nb_points),
                'D0': np.linspace(0.0, common.D_from_length(0.4 * common.LOC_ERR), args.nb_points),
            }
            for name, values in scans.items():
                got = {}
                for mode in ['exact', 'legacy', 'moment_matching']:
                    ys = []
                    ref = params[name].value
                    for v in values:
                        params[name].value = v
                        ys.append(log_likelihood(Cs, params, mode, L).sum())
                    params[name].value = ref
                    got[mode] = argmax_1d(values, np.array(ys))
                scale = params[name].value if params[name].value > 0 else values[-1]
                rows.append((d1, p, name, params[name].value, got['exact'],
                             got['legacy'], got['moment_matching'], scale))

    print('   parameter | mean bias legacy   mean bias moment m. | mean |bias| legacy  moment m.   gain')
    for name in ['D1', 'LocErr', 'p01', 'D0']:
        sub = [r for r in rows if r[2] == name]
        bl = np.array([(r[5] - r[4]) / r[7] for r in sub]) * 100
        bm = np.array([(r[6] - r[4]) / r[7] for r in sub]) * 100
        print('   %-9s | %+15.3f %% %+18.3f %% | %15.3f %% %10.3f %%  %5.2f x'
              % (name, bl.mean(), bm.mean(), np.abs(bl).mean(), np.abs(bm).mean(),
                 np.abs(bl).mean() / max(np.abs(bm).mean(), 1e-12)))
    print('   (in %% of the true value, D0 in %% of the scan range since its truth is 0)')
    np.save('argmax_bias_T%d_L%d.npy' % (T, L), np.array(rows, dtype=object))
    print('')
    print('   moment matching closer to the exact maximum at %d/%d (parameter, model) pairs'
          % (sum(abs(r[6] - r[4]) < abs(r[5] - r[4]) for r in rows), len(rows)))


if __name__ == '__main__':
    t0 = time.time()
    main()
    print('')
    print('done in %.0f s' % (time.time() - t0))
