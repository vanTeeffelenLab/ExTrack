#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The batched / segmented path against the per length path, bit for bit.

The claim being tested is exact identity, not agreement to a tolerance: cutting a
track into segments and carrying the message across the cut must apply the same
operations to that track's numbers in the same order as running it in one go, and
putting tracks of different lengths in one batch must not change any of them.
`np.array_equal` on the log likelihoods is the check; anything else is a failure.

  1. segmentation is exact: one data set, every segment length, against the
     unsegmented run.
  2. batching is exact: tracks of many lengths in shared batches, against
     ExTrack's own per length path.
  3. the packing is what it claims to be: longest first, and the unused fraction
     of the last batches.
  4. state predictions survive the same treatment.
"""

import numpy as np

import common
from common import tracking
from extrack import segmentation

fails = []


def check(name, ok, detail=''):
    print(('  OK   ' if ok else '  FAIL ') + name + ('   ' + detail if detail else ''))
    if not ok:
        fails.append(name)


def make_tracks(lengths, counts, nb_states=2, nb_dims=2, seed=3):
    """tracks of several lengths, from the frame level model"""
    rng = np.random.default_rng(seed)
    ds = np.linspace(0, 0.05, nb_states)
    p = 0.08
    TrMat = np.full((nb_states, nb_states), p)
    TrMat[np.arange(nb_states), np.arange(nb_states)] = 1 - p * (nb_states - 1)
    Fs = np.ones(nb_states) / nb_states
    cum = np.cumsum(TrMat, 1)
    out = {}
    for L, n in zip(lengths, counts):
        st = np.empty((n, L), dtype=int)
        st[:, 0] = rng.integers(0, nb_states, n)
        for t in range(1, L):
            st[:, t] = (rng.random(n)[:, None] > cum[st[:, t - 1]][:, :-1]).sum(1)
        var = (ds[st[:, :-1]] ** 2 + ds[st[:, 1:]] ** 2) / 2
        steps = rng.normal(size=(n, L - 1, nb_dims)) * np.sqrt(var)[:, :, None]
        pos = np.concatenate([np.zeros((n, 1, nb_dims)), np.cumsum(steps, 1)], 1)
        out[str(L)] = pos + rng.normal(0, 0.02, pos.shape)
    return out, ds, Fs, TrMat


LOC = 0.02
PBL = 0.05
CELL = common.BIG_FOV


def per_length_reference(all_tracks, ds, Fs, TrMat, frame_len, min_len, max_len,
                         do_preds=0):
    """ExTrack's own path: one recursion per track length"""
    LocErr = np.array(LOC)[None, None, None]
    LP, preds = {}, {}
    for key in sorted(all_tracks.keys(), key=int):
        Cs = np.asarray(all_tracks[key])
        if len(Cs) == 0:
            continue
        isBL = 0 if int(key) >= max_len else 1
        out = tracking.P_Cs_inter_bound_stats_ages(
            Cs, LocErr, ds, Fs, TrMat, PBL, isBL, CELL, 1, frame_len,
            do_preds=do_preds, min_len=min_len, threshold=0.2, max_nb_states=200)
        lp = np.asarray(out[0])
        mx = lp.max(1)
        LP[key] = mx + np.log(np.exp(lp - mx[:, None]).sum(1))
        if do_preds:
            preds[key] = np.asarray(out[2])
    return LP, preds


def batched(all_tracks, ds, Fs, TrMat, frame_len, min_len, max_len,
            segment_length, batch_size, do_preds=0):
    return tracking.Proba_Cs_batched(
        all_tracks, np.array(LOC)[None, None, None], ds, Fs, TrMat, PBL, CELL, 1,
        frame_len, min_len, 0.2, 200, input_LocErr=None,
        segment_length=segment_length, batch_size=batch_size, max_len=max_len,
        do_preds=do_preds)


tracking.set_numba('auto', threads=8)
common.set_mode('moment_matching')
print(tracking.numba_status())
print('')

# ---------------------------------------------------------------------------
print('1) segmentation is exact: one length, every segment length')
for L, n, S, fl in [(20, 300, 2, 6), (25, 200, 3, 4), (12, 150, 2, 4), (31, 120, 2, 5)]:
    all_tracks, ds, Fs, TrMat = make_tracks([L], [n], nb_states=S)
    ref, _ = per_length_reference(all_tracks, ds, Fs, TrMat, fl, L, L)
    for seg in [None, 3, 4, 5, 7, 10, 13, L - 1, L, L + 5]:
        got, _ = batched(all_tracks, ds, Fs, TrMat, fl, L, L, seg, 1000)
        same = np.array_equal(ref[str(L)], got[str(L)])
        if not same:
            d = np.abs(ref[str(L)] - got[str(L)]).max()
            check('L=%d S=%d segment_length=%s' % (L, S, seg), False, 'max diff %.3e' % d)
    check('L=%2d S=%d fl=%d : bitwise identical at every segment length' % (L, S, fl),
          all(np.array_equal(ref[str(L)],
                             batched(all_tracks, ds, Fs, TrMat, fl, L, L, seg, 1000)[0][str(L)])
              for seg in [None, 3, 4, 5, 7, 10, 13, L - 1, L, L + 5]))

# ---------------------------------------------------------------------------
print('')
print('2) batching tracks of many lengths is exact')
CASES = [
    ('lengths 5..20, 2 states', list(range(5, 21)), [40] * 16, 2, 6),
    ('lengths 5..20, 3 states', list(range(5, 21)), [30] * 16, 3, 4),
    ('lengths 4..30 sparse', [4, 7, 11, 18, 23, 30], [50, 80, 60, 40, 30, 90], 2, 5),
    ('one length repeated', [9, 9], [100, 100], 2, 6),
    ('very short and very long', [3, 40], [60, 20], 2, 5),
]
for label, lengths, counts, S, fl in CASES:
    all_tracks, ds, Fs, TrMat = make_tracks(lengths, counts, nb_states=S)
    min_len = min(int(k) for k in all_tracks)
    max_len = max(int(k) for k in all_tracks)
    ref, _ = per_length_reference(all_tracks, ds, Fs, TrMat, fl, min_len, max_len)
    for seg, bs in [(None, 10 ** 9), (None, 128), (5, 10 ** 9), (5, 128), (8, 64), (3, 256)]:
        got, _ = batched(all_tracks, ds, Fs, TrMat, fl, min_len, max_len, seg, bs)
        bad = [k for k in ref if not np.array_equal(ref[k], got[k])]
        if bad:
            d = max(np.abs(ref[k] - got[k]).max() for k in bad)
            check('%s seg=%s batch=%s' % (label, seg, bs), False,
                  '%d/%d lengths differ, max %.3e' % (len(bad), len(ref), d))
    ok = True
    for seg, bs in [(None, 10 ** 9), (None, 128), (5, 10 ** 9), (5, 128), (8, 64), (3, 256)]:
        got, _ = batched(all_tracks, ds, Fs, TrMat, fl, min_len, max_len, seg, bs)
        ok &= all(np.array_equal(ref[k], got[k]) for k in ref)
    check('%-26s bitwise identical, 6 packings' % label, ok,
          '%d tracks over %d lengths' % (sum(counts), len(lengths)))

# ---------------------------------------------------------------------------
print('')
print('3) the packing: longest first, and how full the batches are')
all_tracks, ds, Fs, TrMat = make_tracks(list(range(5, 21)), [40] * 16)
tracks, locerrs, origin = segmentation.flatten_tracks(all_tracks)
for seg in [None, 4, 6, 10]:
    batches, order, lengths = segmentation.segment_tracks(tracks, seg, 128)
    lens = np.array([len(tracks[i]) for i in order])
    check('segment_length=%-4s sorted longest first' % seg,
          bool(np.all(lens[:-1] >= lens[1:])))
    print('       ' + segmentation.describe(batches, tracks))

# ---------------------------------------------------------------------------
print('')
print('4) state predictions survive batching')
for label, lengths, counts, S, fl in CASES[:3]:
    all_tracks, ds, Fs, TrMat = make_tracks(lengths, counts, nb_states=S)
    min_len = min(int(k) for k in all_tracks)
    max_len = max(int(k) for k in all_tracks)
    ref, ref_p = per_length_reference(all_tracks, ds, Fs, TrMat, fl, min_len, max_len,
                                      do_preds=1)
    got, got_p = batched(all_tracks, ds, Fs, TrMat, fl, min_len, max_len, 5, 128,
                         do_preds=1)
    ok_lp = all(np.array_equal(ref[k], got[k]) for k in ref)
    ok_p = all(np.array_equal(ref_p[k], got_p[k]) for k in ref_p)
    worst = max(np.abs(ref_p[k] - got_p[k]).max() for k in ref_p)
    check('%-26s posteriors bitwise identical' % label, ok_lp and ok_p,
          'max diff %.3e' % worst)

print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), '; '.join(fails[:6])))
    raise SystemExit(1)
print('all checks passed')
