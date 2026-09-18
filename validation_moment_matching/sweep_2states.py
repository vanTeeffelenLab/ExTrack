#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the moment matched fusion improve the fitted parameters and the predicted
states of a 2 state model ?

Grid: d0 = 0, d1 in {0.01, 0.015, 0.02, 0.03, 0.04} um per step, transition
probability per step in {0.02, 0.05, 0.1, 0.2}, 3 replicates each, tracks of 7
time points, localization error 0.02 um.

Three fusion modes are compared on identical data, from an identical (deliberately
imperfect) starting point:

  exact             no fusion at all, every one of the 2**7 sequences of states is
                    kept. This is the likelihood the two others approximate, so it
                    is both the gold standard estimator and the reference the
                    approximation error is measured against.
  legacy            ExTrack's historical fusion.
  moment_matching   the fused variance also carries the spread of the means.

Usage:
    python sweep_2states.py                 # runs the whole grid in parallel
    python sweep_2states.py --nb-tracks 2000 --workers 8
"""

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODES = ['exact', 'legacy', 'moment_matching']

TRACK_LEN = 7
FRAME_LEN = 6
THRESHOLD = 0.2
MAX_NB_STATES = 200
NB_REPLICATES = 3


def start_params(m):
    """
    Same imperfect starting point for every mode, so that the comparison measures
    the estimator and not the optimizer.
    """
    import common
    params = common.params_from_truth(m)
    params['D1'].value = m['Ds'][1] * 1.5
    params['D0'].value = m['Ds'][1] * 0.05
    params['LocErr'].value = m['LocErr'] * 1.2
    params['F0'].value = 0.45
    params['p01'].value = -np.log(1 - m['p']) * 1.5
    params['p10'].value = -np.log(1 - m['p']) * 0.7
    return params


def posterior(Cs, params, mode, frame_len=FRAME_LEN, threshold=THRESHOLD):
    """P(state | track) for every time point"""
    import common
    from common import tracking
    common.set_mode(mode)
    LocErr, ds, Fs, TrMat, pBL = tracking.extract_params(
        params, common.DT, 2, 1, None, Matrix_type=1)
    _, _, preds = tracking.P_Cs_inter_bound_stats_th(
        Cs, LocErr[0], ds, Fs, TrMat, pBL, 0, common.BIG_FOV, 1, frame_len,
        do_preds=1, min_len=Cs.shape[1], threshold=threshold,
        max_nb_states=MAX_NB_STATES)
    return np.asarray(preds)


def total_log_likelihood(Cs, params, mode, frame_len=FRAME_LEN, threshold=THRESHOLD):
    import common
    from common import tracking
    common.set_mode(mode)
    LocErr, ds, Fs, TrMat, pBL = tracking.extract_params(
        params, common.DT, 2, 1, None, Matrix_type=1)
    return np.asarray(tracking.Proba_Cs(Cs, LocErr[0], ds, Fs, TrMat, pBL, 0,
                                        common.BIG_FOV, 1, frame_len, Cs.shape[1],
                                        threshold, MAX_NB_STATES))


def state_metrics(pred, truth_states, exact_pred):
    """
    pred, exact_pred: (nb_tracks, track_len, 2) posteriors
    truth_states: (nb_tracks, track_len) simulated hidden states
    """
    eps = 1e-12
    p1 = np.clip(pred[:, :, 1], eps, 1 - eps)
    t = truth_states.astype(float)
    return dict(
        accuracy=float(np.mean((p1 > 0.5) == (t > 0.5))),
        log_loss=float(-np.mean(t * np.log(p1) + (1 - t) * np.log(1 - p1))),
        brier=float(np.mean((p1 - t) ** 2)),
        mean_abs_dev_from_exact=float(np.mean(np.abs(p1 - exact_pred[:, :, 1]))),
        rms_dev_from_exact=float(np.sqrt(np.mean((p1 - exact_pred[:, :, 1]) ** 2))),
        max_dev_from_exact=float(np.max(np.abs(p1 - exact_pred[:, :, 1]))),
    )


def run_point(job):
    d1, p, rep, nb_tracks, simulator = job
    import common
    from common import tracking

    seed = (int(round(d1 * 1000)) * 100003 + int(round(p * 1000)) * 1009 + rep * 7919) % (2 ** 31)
    if simulator == 'well_specified':
        tracks, states, m = common.simulate_well_specified(d1, p, nb_tracks, TRACK_LEN, seed)
    else:
        tracks, states, m = common.simulate(d1, p, nb_tracks, TRACK_LEN, seed)
    Cs = tracks[str(TRACK_LEN)]
    truth_states = states[str(TRACK_LEN)]
    truth = dict(D0=float(m['Ds'][0]), D1=float(m['Ds'][1]), LocErr=common.LOC_ERR,
                 F0=0.5, T01=p, T10=p, d1=d1, p=p)

    out = dict(d1=d1, p=p, rep=rep, nb_tracks=nb_tracks, simulator=simulator,
               truth=truth, modes={})

    # posterior of every mode at the TRUE parameters : isolates the fusion from the fit
    true_params = common.params_from_truth(m)
    post = {}
    for mode in MODES:
        post[mode] = posterior(Cs, true_params, mode)
    lp = {mode: total_log_likelihood(Cs, true_params, mode) for mode in MODES}

    for mode in MODES:
        t0 = time.time()
        common.set_mode(mode)
        params = start_params(m)
        fit = tracking.param_fitting(all_tracks={str(TRACK_LEN): Cs}, dt=common.DT,
                                     params=params, nb_states=2, nb_substeps=1,
                                     frame_len=FRAME_LEN, verbose=0, workers=1,
                                     cell_dims=common.BIG_FOV, threshold=THRESHOLD,
                                     max_nb_states=MAX_NB_STATES, method='BFGS')
        v = fit.params
        fitted = dict(D0=float(v['D0'].value), D1=float(v['D1'].value),
                      LocErr=float(v['LocErr'].value), F0=float(v['F0'].value),
                      T01=float(1 - np.exp(-v['p01'].value)),
                      T10=float(1 - np.exp(-v['p10'].value)))
        fitted['d0'] = float(common.length_from_D(max(fitted['D0'], 0)))
        fitted['d1'] = float(common.length_from_D(max(fitted['D1'], 0)))

        # states predicted with each mode's own fitted parameters
        fit_post = posterior(Cs, v, mode)

        out['modes'][mode] = dict(
            fitted=fitted,
            nfev=int(fit.nfev),
            fit_seconds=time.time() - t0,
            logL_error_at_truth=float(lp[mode].sum() - lp['exact'].sum()),
            logL_rms_error_at_truth=float(np.sqrt(np.mean((lp[mode] - lp['exact']) ** 2))),
            states_at_true_params=state_metrics(post[mode], truth_states, post['exact']),
            states_at_fitted_params=state_metrics(fit_post, truth_states, post['exact']),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nb-tracks', type=int, default=5000)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--simulator', default='well_specified',
                    choices=['well_specified', 'sim_noBias'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(HERE, 'results_2states_%s.json' % args.simulator)

    import common
    jobs = [(d1, p, rep, args.nb_tracks, args.simulator)
            for d1 in common.D1_LENGTHS
            for p in common.TRANSITION_PS
            for rep in range(NB_REPLICATES)]

    print('%d grid points x %d modes, %d tracks of %d points each, simulator=%s'
          % (len(jobs), len(MODES), args.nb_tracks, TRACK_LEN, args.simulator))
    t0 = time.time()
    results = []
    if args.workers <= 1:
        for j in jobs:
            results.append(run_point(j))
            print('  %d/%d  %.0f s' % (len(results), len(jobs), time.time() - t0))
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(run_point, jobs):
                results.append(r)
                print('  %d/%d  d1=%.3f p=%.2f rep=%d   %.0f s'
                      % (len(results), len(jobs), r['d1'], r['p'], r['rep'],
                         time.time() - t0))

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('wrote %s in %.0f s' % (args.out, time.time() - t0))


if __name__ == '__main__':
    main()
