#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Where does one likelihood evaluation actually spend its time?

The question this answers is whether rewriting the recursion in C++ would help.
That depends entirely on the split between

  * work done *inside* numpy kernels on large arrays  -> already compiled, C++
    would only match it (both are memory-bandwidth bound at these sizes),
  * python-level control flow and per-call numpy dispatch -> the part a compiled
    kernel actually removes.

So the profile is reported as: seconds in the python grouping loop, seconds in
numpy primitives, number of numpy calls, and bytes touched per second.
"""

import cProfile
import io
import pstats
import time

import numpy as np

import common
import sweep_3states as S3
from common import tracking

CASES = [
    ('2 states, frame_len 6', 2, 6, 2000, 20),
    ('2 states, frame_len 8', 2, 8, 2000, 20),
    ('3 states, frame_len 4', 3, 4, 2000, 12),
    ('3 states, frame_len 6', 3, 6, 2000, 12),
    ('4 states, frame_len 4', 4, 4, 2000, 12),
    ('2 states, 50 tracks', 2, 6, 50, 20),
]


def make(nb_states, nb_tracks, track_len, seed=7):
    """tracks from the frame level model, states spread over a realistic range"""
    rng = np.random.default_rng(seed)
    ds = np.linspace(0, 0.05, nb_states)
    p = 0.06
    TrMat = np.full((nb_states, nb_states), p)
    TrMat[np.arange(nb_states), np.arange(nb_states)] = 1 - p * (nb_states - 1)
    Fs = np.ones(nb_states) / nb_states
    cum = np.cumsum(TrMat, 1)
    st = np.empty((nb_tracks, track_len), dtype=int)
    st[:, 0] = rng.integers(0, nb_states, nb_tracks)
    for t in range(1, track_len):
        st[:, t] = (rng.random(nb_tracks)[:, None] > cum[st[:, t - 1]][:, :-1]).sum(1)
    var = (ds[st[:, :-1]] ** 2 + ds[st[:, 1:]] ** 2) / 2
    steps = rng.normal(size=(nb_tracks, track_len - 1, 2)) * np.sqrt(var)[:, :, None]
    pos = np.concatenate([np.zeros((nb_tracks, 1, 2)), np.cumsum(steps, 1)], 1)
    pos = pos + rng.normal(0, 0.02, pos.shape)
    return pos, ds, Fs, TrMat


def one_eval(Cs, LocErr, ds, Fs, TrMat, L, scheme):
    return tracking.Proba_Cs(Cs, LocErr, ds, Fs, TrMat, 1e-8, 0, common.BIG_FOV, 1, L,
                             Cs.shape[1], 0.2, 200, scheme)


def timed(fn, n=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def profile(fn):
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    st = pstats.Stats(pr)
    tot = st.total_tt
    grouping = 0.0
    numpy_calls = 0
    numpy_time = 0.0
    for (fname, line, func), (cc, nc, tt, ct, callers) in st.stats.items():
        if func == 'fuse_tracks_th':
            grouping = ct
        if fname == '~' or 'numpy' in fname or fname.startswith('<'):
            numpy_calls += nc
            numpy_time += tt
    return tot, grouping, numpy_calls, numpy_time


def main():
    common.set_mode('moment_matching')
    LocErr = np.array(0.02)[None, None, None]

    print('One likelihood evaluation, %s' % time.strftime('%Y-%m-%d'))
    print('')
    head = ('  case                     tracks x len |  sequences   ages   speed-up |'
            '  grouping loop  numpy calls  us/call')
    print(head)
    print('  ' + '-' * (len(head) - 2))

    for label, S, L, N, T in CASES:
        Cs, ds, Fs, TrMat = make(S, N, T)
        res = {}
        for scheme in ['sequences', 'ages']:
            res[scheme] = timed(lambda sc=scheme: one_eval(Cs, LocErr, ds, Fs, TrMat, L, sc))
        tot, grouping, ncalls, ntime = profile(
            lambda: one_eval(Cs, LocErr, ds, Fs, TrMat, L, 'sequences'))
        print('  %-24s %5d x %-3d |  %8.4f %6.4f %8.2f x |  %10.1f %%  %10d %7.2f'
              % (label, N, T, res['sequences'], res['ages'],
                 res['sequences'] / res['ages'],
                 100 * grouping / max(tot, 1e-12), ncalls,
                 1e6 * res['sequences'] / max(ncalls, 1)))

    print('')
    print('  "grouping loop" = share of the evaluation inside fuse_tracks_th, which is')
    print('  a python loop over pairs of branches. "us/call" is the whole evaluation')
    print('  divided by the number of numpy/builtin calls it makes.')

    # ---------------------------------------------------------------------------
    print('')
    print('How big are the arrays the numpy kernels actually work on?')
    print('  case                     |  largest array (elements)   bytes moved / s')
    for label, S, L, N, T in CASES[:5]:
        Cs, ds, Fs, TrMat = make(S, N, T)
        sizes = []
        real = tracking.log_integrale_dif

        def spy(Ci, l2, cur_d2s, m_arr, s2_arr):
            sizes.append(m_arr.size)
            return real(Ci, l2, cur_d2s, m_arr, s2_arr)

        tracking.log_integrale_dif = spy
        one_eval(Cs, LocErr, ds, Fs, TrMat, L, 'sequences')
        tracking.log_integrale_dif = real
        t = timed(lambda: one_eval(Cs, LocErr, ds, Fs, TrMat, L, 'sequences'), 3)
        touched = 8 * sum(sizes) * 12   # ~12 array traversals per log_integrale_dif call
        print('  %-24s |  %18d   %13.2f GB/s'
              % (label, max(sizes), touched / t / 1e9))
    print('')
    print('  (a single core of this machine sustains roughly 10-20 GB/s from L3/RAM,')
    print('   so a figure well under that means the kernels are not bandwidth bound')
    print('   and the time is going somewhere else)')


if __name__ == '__main__':
    main()
