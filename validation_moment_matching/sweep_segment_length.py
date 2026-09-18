#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Which segment length is the most efficient?

Two things trade off. A short segment packs the batches tightly, because a track
only occupies slots it actually has data for, but it costs one kernel call per
segment and the carry buffers are read and written at every cut. A long segment
makes fewer, larger calls but leaves the short tracks' slots idle -- except that
this implementation gives every track its own step count, so an idle slot costs
nothing beyond the array space. The sweep measures where that lands.

The baseline is ExTrack's own path: one recursion per track length.
"""

import time

import numpy as np

import common
from common import tracking
from extrack import segmentation

LOC = 0.02
PBL = 0.05
CELL = common.BIG_FOV


def simulate(nb_tracks, lengths, weights, nb_states=2, nb_dims=2, seed=5):
    rng = np.random.default_rng(seed)
    ds = np.linspace(0, 0.05, nb_states)
    p = 0.08
    TrMat = np.full((nb_states, nb_states), p)
    TrMat[np.arange(nb_states), np.arange(nb_states)] = 1 - p * (nb_states - 1)
    Fs = np.ones(nb_states) / nb_states
    cum = np.cumsum(TrMat, 1)
    counts = rng.multinomial(nb_tracks, np.asarray(weights, float) / np.sum(weights))
    out = {}
    for L, n in zip(lengths, counts):
        if n == 0:
            continue
        st = np.empty((n, L), dtype=int)
        st[:, 0] = rng.integers(0, nb_states, n)
        for t in range(1, L):
            st[:, t] = (rng.random(n)[:, None] > cum[st[:, t - 1]][:, :-1]).sum(1)
        var = (ds[st[:, :-1]] ** 2 + ds[st[:, 1:]] ** 2) / 2
        steps = rng.normal(size=(n, L - 1, nb_dims)) * np.sqrt(var)[:, :, None]
        pos = np.concatenate([np.zeros((n, 1, nb_dims)), np.cumsum(steps, 1)], 1)
        out[str(L)] = pos + rng.normal(0, 0.02, pos.shape)
    return out, ds, Fs, TrMat


def med(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    tracking.set_numba('auto', threads=8)
    common.set_mode('moment_matching')
    print(tracking.numba_status())

    # a realistic distribution: many short tracks, a pile-up at the maximum length
    lengths = list(range(5, 21))
    weights = [8, 7, 6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 30]
    for nb_tracks, S, fl in [(5000, 2, 6), (5000, 3, 4)]:
        all_tracks, ds, Fs, TrMat = simulate(nb_tracks, lengths, weights, nb_states=S)
        min_len = min(int(k) for k in all_tracks)
        max_len = max(int(k) for k in all_tracks)
        n_groups = len(all_tracks)
        total_steps = sum(len(v) * (int(k) - 1) for k, v in all_tracks.items())
        print('')
        print('%d tracks, %d states, frame_len %d, %d length groups, %d displacements'
              % (nb_tracks, S, fl, n_groups, total_steps))

        LocErr = np.array(LOC)[None, None, None]

        def per_length():
            for key in sorted(all_tracks.keys(), key=int):
                Cs = np.asarray(all_tracks[key])
                tracking.P_Cs_inter_bound_stats_ages(
                    Cs, LocErr, ds, Fs, TrMat, PBL, 0 if int(key) >= max_len else 1,
                    CELL, 1, fl, do_preds=0, min_len=min_len, threshold=0.2,
                    max_nb_states=200)

        base = med(per_length)
        print('  per length (ExTrack today) : %.4f s   %d recursions' % (base, n_groups))
        print('')
        print('  segment  batch |   s / call   speed-up |  batches  slots used')
        for seg in [None, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
            for bs in [1000, 5000]:
                # the packing depends only on the track lengths, so a fit builds
                # it once; it is hoisted out of the timed region for that reason
                packing = tracking.pack_tracks(all_tracks, seg, bs)

                def run(seg=seg, bs=bs, packing=packing):
                    tracking.Proba_Cs_batched(
                        all_tracks, LocErr, ds, Fs, TrMat, PBL, CELL, 1, fl,
                        min_len, 0.2, 200, input_LocErr=None,
                        segment_length=seg, batch_size=bs, max_len=max_len,
                        packing=packing)
                t = med(run)
                batches = packing['batches']
                slots = sum(b['Cs'].shape[0] * (b['Cs'].shape[1] - 1) for b in batches)
                used = sum(int(b['nsteps'].sum()) for b in batches)
                print('  %-7s  %-5d | %9.4f  %8.2fx |  %6d  %8.1f %%'
                      % (seg, bs, t, base / t, len(batches), 100.0 * used / slots))


if __name__ == '__main__':
    main()
