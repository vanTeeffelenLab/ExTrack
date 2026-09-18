#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The numba kernels against the numpy implementation they replace.

Both fusion schemes, both fusion formulas, several state counts, dimensions,
localization-error layouts and track lengths. The likelihood and the state
posteriors are compared, and so are the grouping decisions themselves, because
those are the one place where a difference of an ulp could change the answer by
more than an ulp.
"""

import numpy as np

import common
import profile_likelihood as PL
from common import tracking

fails = []


def check(name, ok, detail=''):
    print(('  OK   ' if ok else '  FAIL ') + name + ('   ' + detail if detail else ''))
    if not ok:
        fails.append(name)


def both(fn):
    """run fn once with the numba kernels and once without"""
    tracking.set_numba(False)
    a = fn()
    tracking.set_numba('auto')
    b = fn()
    return a, b


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = max(np.abs(a).max(), 1e-300)
    return np.abs(a - b).max() / scale


CASES = [
    # label,                    S, frame_len, tracks, len, dims, isBL, LocErr layout
    ('2 states, fl 6',          2, 6, 400, 15, 2, 0, 'scalar'),
    ('2 states, fl 4, isBL',    2, 4, 400, 15, 2, 1, 'scalar'),
    ('3 states, fl 4',          3, 4, 400, 12, 2, 0, 'scalar'),
    ('3 states, fl 5, 3 dims',  3, 5, 300, 10, 3, 0, 'scalar'),
    ('4 states, fl 3',          4, 3, 300, 10, 2, 1, 'scalar'),
    ('2 states, per-dim LocErr', 2, 5, 300, 12, 2, 0, 'perdim'),
    ('2 states, per-peak LocErr', 2, 5, 300, 12, 2, 0, 'perpeak'),
    ('2 states, short tracks',  2, 6, 200, 4, 2, 1, 'scalar'),
    ('3 states, 5 tracks',      3, 4, 5, 12, 2, 0, 'scalar'),
]


def locerr_for(kind, nb_tracks, track_len, nb_dims, rng):
    if kind == 'scalar':
        return np.array(0.02)[None, None, None]
    if kind == 'perdim':
        return np.array([0.02, 0.035][:nb_dims])[None, None]
    return np.abs(rng.normal(0.025, 0.004, (nb_tracks, track_len, 1)))


def make(S, N, T, D, rng):
    ds = np.linspace(0, 0.05, S)
    p = 0.07
    TrMat = np.full((S, S), p)
    TrMat[np.arange(S), np.arange(S)] = 1 - p * (S - 1)
    Fs = np.ones(S) / S
    cum = np.cumsum(TrMat, 1)
    st = np.empty((N, T), dtype=int)
    st[:, 0] = rng.integers(0, S, N)
    for t in range(1, T):
        st[:, t] = (rng.random(N)[:, None] > cum[st[:, t - 1]][:, :-1]).sum(1)
    var = (ds[st[:, :-1]] ** 2 + ds[st[:, 1:]] ** 2) / 2
    steps = rng.normal(size=(N, T - 1, D)) * np.sqrt(var)[:, :, None]
    pos = np.concatenate([np.zeros((N, 1, D)), np.cumsum(steps, 1)], 1)
    return pos + rng.normal(0, 0.02, pos.shape), ds, Fs, TrMat


print('numba status:', tracking.numba_status())
print('')

for fusion in ['moment_matching', 'legacy']:
    print('fusion = %s' % fusion)
    for label, S, L, N, T, D, isBL, lk in CASES:
        rng = np.random.default_rng(hash((label, fusion)) % 2 ** 31)
        Cs, ds, Fs, TrMat = make(S, N, T, D, rng)
        LocErr = locerr_for(lk, N, T, D, rng)

        def run(scheme, do_preds):
            common.set_mode(fusion)
            return tracking.P_Cs_inter_bound_stats_th(
                Cs, LocErr, ds, Fs, TrMat, 0.05, isBL, common.BIG_FOV, 1, L,
                do_preds=do_preds, min_len=T, threshold=0.2, max_nb_states=200) \
                if scheme == 'sequences' else \
                tracking.P_Cs_inter_bound_stats_ages(
                    Cs, LocErr, ds, Fs, TrMat, 0.05, isBL, common.BIG_FOV, 1, L,
                    do_preds=do_preds, min_len=T, threshold=0.2, max_nb_states=200)

        for scheme in ['sequences', 'ages']:
            (LPa, _, pa), (LPb, _, pb) = both(lambda: run(scheme, 1))
            tot_a = np.log(np.exp(np.asarray(LPa) - np.asarray(LPa).max(1, keepdims=True)).sum(1)) \
                + np.asarray(LPa).max(1)
            tot_b = np.log(np.exp(np.asarray(LPb) - np.asarray(LPb).max(1, keepdims=True)).sum(1)) \
                + np.asarray(LPb).max(1)
            e_lp = rel(tot_a, tot_b)
            e_pr = np.abs(np.asarray(pa) - np.asarray(pb)).max()
            check('%-27s %-10s likelihood' % (label, scheme), e_lp < 1e-11,
                  'rel %.2e' % e_lp)
            check('%-27s %-10s posteriors' % (label, scheme), e_pr < 1e-9,
                  'max %.2e' % e_pr)
    print('')

# ---------------------------------------------------------------------------
print('grouping decisions are identical (sequences scheme)')
seen = {'same': 0, 'total': 0}
real = tracking.fuse_tracks_th
real_nb = tracking.fuse_tracks_th_numba


def spy(m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat, nb_Tracks, nb_states=2, nb_dims=2,
        do_preds=1, threshold=0.2, frame_len=6):
    """run both groupings on the same inputs and compare the partitions"""
    tracking.set_numba(False)
    out_np = real(m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat, nb_Tracks, nb_states,
                  nb_dims, do_preds, threshold, frame_len)
    tracking.set_numba('auto')
    out_nb = real_nb(m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat, nb_Tracks, nb_states,
                     nb_dims, do_preds, threshold, frame_len)
    seen['total'] += 1
    seen['same'] += int(out_np[0].shape[1] == out_nb[0].shape[1]
                        and np.allclose(out_np[2], out_nb[2], rtol=1e-12, atol=1e-12))
    return out_nb


for label, S, L, N, T, D, isBL, lk in CASES[:6]:
    rng = np.random.default_rng(11)
    Cs, ds, Fs, TrMat = make(S, N, T, D, rng)
    LocErr = locerr_for(lk, N, T, D, rng)
    common.set_mode('moment_matching')
    tracking.fuse_tracks_th = spy
    tracking.P_Cs_inter_bound_stats_th(Cs, LocErr, ds, Fs, TrMat, 0.05, isBL,
                                       common.BIG_FOV, 1, L, do_preds=0, min_len=T,
                                       threshold=0.2, max_nb_states=200)
    tracking.fuse_tracks_th = real
check('same partition and same fused weights at every fusion',
      seen['same'] == seen['total'], '%d/%d' % (seen['same'], seen['total']))

tracking.set_numba('auto')
common.set_mode('moment_matching')
print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), ', '.join(fails[:6])))
    raise SystemExit(1)
print('all checks passed')
