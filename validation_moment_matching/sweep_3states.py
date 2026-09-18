#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What does the coarser (segment age, current state) buffer cost on a 3 state model?

Grid: d0 = 0, d1 = k*LocErr, d2 = 3*k*LocErr with
k in {0.2, 0.4, 0.6, 0.8, 1, 1.3, 1.6, 2, 2.5, 3}, every transition probability
0.05 per step, 3 replicates, 5000 tracks of 12 time points, LocErr = 0.02 um,
sequence_length (frame_len) = 4.

Two schemes are fitted on identical data from an identical starting point:

  sequences   one hypothesis per sequence of states over the last frame_len frames
              (up to nb_states**frame_len = 81), adaptively grouped
  ages        one hypothesis per (segment age, current state): a fixed buffer of
              frame_len*nb_states = 12, frame_len*nb_states**2 = 36 branches per step

Both use the moment matched fusion. What is measured is therefore the price of the
coarser state memory, not of the fusion formula.
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

KS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
TRANSITION_P = 0.05
TRACK_LEN = 12
FRAME_LEN = 4
NB_REPLICATES = 3
SCHEMES = ['sequences', 'ages']


def truth_of(k, loc_err):
    ds = np.array([0.0, k * loc_err, 3 * k * loc_err])
    TrMat = np.full((3, 3), TRANSITION_P)
    TrMat[np.arange(3), np.arange(3)] = 1 - 2 * TRANSITION_P
    return ds, TrMat, np.ones(3) / 3.0


def simulate(k, nb_tracks, track_len, seed, loc_err, nb_dims=2):
    """ExTrack's own model: a frame level Markov chain, the displacement variance
    of a step being the average of the variances of its two states."""
    rng = np.random.default_rng(seed)
    ds, TrMat, Fs = truth_of(k, loc_err)
    cum = np.cumsum(TrMat, 1)
    states = np.empty((nb_tracks, track_len), dtype=int)
    states[:, 0] = (rng.random(nb_tracks)[:, None] > np.cumsum(Fs)[None, :-1]).sum(1)
    for t in range(1, track_len):
        u = rng.random(nb_tracks)
        states[:, t] = (u[:, None] > cum[states[:, t - 1]][:, :-1]).sum(1)
    var = (ds[states[:, :-1]] ** 2 + ds[states[:, 1:]] ** 2) / 2
    steps = rng.normal(size=(nb_tracks, track_len - 1, nb_dims)) * np.sqrt(var)[:, :, None]
    pos = np.concatenate([np.zeros((nb_tracks, 1, nb_dims)), np.cumsum(steps, 1)], 1)
    return pos + rng.normal(0, loc_err, pos.shape), states, ds, TrMat, Fs


def make_params(ds, TrMat, loc_err, perturb=True):
    import common
    from common import tracking
    Ds = ds ** 2 / (2 * common.DT)
    params = tracking.generate_params(nb_states=3, LocErr_type=1, nb_dims=2,
                                      LocErr_bounds=[0.002, 0.2], D_max=10,
                                      Fractions_bounds=[0.001, 0.999],
                                      estimated_LocErr=[loc_err],
                                      estimated_Ds=list(Ds),
                                      estimated_Fs=[1 / 3., 1 / 3.],
                                      estimated_transition_rates=[-np.log(1 - TRANSITION_P)] * 6)
    params['pBL'].value = 1e-8
    params['pBL'].vary = False
    if perturb:   # the same imperfect start for both schemes
        params['D0'].value = Ds[2] * 0.02
        params['D1'].value = Ds[1] * 1.4 + 1e-6
        params['D2'].value = Ds[2] * 1.3 + 1e-6
        params['LocErr'].value = loc_err * 1.15
        params['F0'].value = 0.30
        params['F1'].value = 0.38
        for i in range(3):
            for j in range(3):
                if i != j:
                    params['p%d%d' % (i, j)].value = -np.log(1 - TRANSITION_P) * (1.4 if i < j else 0.75)
    return params


def state_metrics(pred, truth_states):
    eps = 1e-12
    p = np.clip(pred, eps, 1)
    oh = np.eye(3)[truth_states]
    return dict(accuracy=float(np.mean(np.argmax(pred, -1) == truth_states)),
                log_loss=float(-np.mean(np.sum(oh * np.log(p), -1))),
                brier=float(np.mean(np.sum((pred - oh) ** 2, -1))))


def run_point(job):
    k, rep, nb_tracks, loc_err = job
    import common
    from common import tracking
    common.set_mode('moment_matching')

    seed = (int(round(k * 100)) * 7919 + rep * 104729) % (2 ** 31)
    Cs, states, ds, TrMat, Fs = simulate(k, nb_tracks, TRACK_LEN, seed, loc_err)
    LocErr = np.array(loc_err)[None, None, None]
    truth = dict(k=k, ds=list(ds), LocErr=loc_err, Fs=list(Fs),
                 T=[[float(TrMat[i, j]) for j in range(3)] for i in range(3)])
    out = dict(k=k, rep=rep, nb_tracks=nb_tracks, truth=truth, schemes={})

    for scheme in SCHEMES:
        t0 = time.time()
        lp = np.asarray(tracking.Proba_Cs(Cs, LocErr, ds, Fs, TrMat, 1e-8, 0, common.BIG_FOV,
                                          1, FRAME_LEN, TRACK_LEN, 0.2, 200, scheme))
        eval_time = time.time() - t0

        t0 = time.time()
        params = make_params(ds, TrMat, loc_err)
        fit = tracking.param_fitting(all_tracks={str(TRACK_LEN): Cs}, dt=common.DT,
                                     params=params, nb_states=3, nb_substeps=1,
                                     frame_len=FRAME_LEN, verbose=0, workers=1,
                                     cell_dims=common.BIG_FOV, threshold=0.2,
                                     max_nb_states=200, method='BFGS',
                                     sequence_scheme=scheme)
        fit_time = time.time() - t0
        v = fit.params
        fitted = dict(LocErr=float(v['LocErr'].value),
                      ds=[float(np.sqrt(2 * max(v['D%d' % s].value, 0) * common.DT)) for s in range(3)],
                      Fs=[float(v['F%d' % s].value) for s in range(3)],
                      T={'%d%d' % (i, j): float(1 - np.exp(-v['p%d%d' % (i, j)].value))
                         for i in range(3) for j in range(3) if i != j})

        preds = tracking.predict_Bs({str(TRACK_LEN): Cs}, common.DT, v, cell_dims=common.BIG_FOV,
                                    nb_states=3, frame_len=FRAME_LEN, workers=1,
                                    nb_max=1000, sequence_scheme=scheme)[str(TRACK_LEN)]

        out['schemes'][scheme] = dict(fitted=fitted, nfev=int(fit.nfev),
                                      fit_seconds=fit_time, eval_seconds=eval_time,
                                      logL=float(lp.sum()),
                                      states=state_metrics(preds, states))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nb-tracks', type=int, default=5000)
    ap.add_argument('--loc-err', type=float, default=0.02)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--out', default=os.path.join(HERE, 'results_3states.json'))
    args = ap.parse_args()

    jobs = [(k, rep, args.nb_tracks, args.loc_err) for k in KS for rep in range(NB_REPLICATES)]
    print('%d points x %d schemes, %d tracks of %d points, frame_len = %d'
          % (len(jobs), len(SCHEMES), args.nb_tracks, TRACK_LEN, FRAME_LEN))
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
                print('  %d/%d  k=%.1f rep=%d   %.0f s'
                      % (len(results), len(jobs), r['k'], r['rep'], time.time() - t0))
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('wrote %s in %.0f s' % (args.out, time.time() - t0))


if __name__ == '__main__':
    main()
