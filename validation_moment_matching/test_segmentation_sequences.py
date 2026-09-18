#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The `sequences` scheme, batched over tracks of different lengths.

That scheme shares one *dynamic* set of branches across a batch, and the grouping
of `fuse_tracks_th` samples the batch's 30 first tracks: re-batching therefore
changes which branches merge, and the likelihood with it. There is one regime
where that objection disappears entirely -- when no fusion happens at all:

    frame_len >= longest track     the frame_len truncation never fires
    threshold tiny                 only a branch matches itself

and there the branch set is just the full tree, identical for every batching.
This checks bitwise identity there, on tracks short enough for the full tree
(nb_states ** track_len) to be affordable, and then measures how far the two
paths drift once the fusion is switched back on.

The threshold must be small but not zero: at exactly 0 a branch fails its own
`|dm| / sigma < threshold` test, ends up in no group at all, and the reference
implementation raises.
"""

import numpy as np

import common
from common import tracking

NO_FUSION_THRESHOLD = 1e-12
LOC = 0.02
PBL = 0.05
CELL = common.BIG_FOV

fails = []


def check(name, ok, detail=''):
    print(('  OK   ' if ok else '  FAIL ') + name + ('   ' + detail if detail else ''))
    if not ok:
        fails.append(name)


def make_tracks(lengths, counts, nb_states=2, nb_dims=2, seed=17):
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


def per_length(all_tracks, ds, Fs, TrMat, frame_len, min_len, max_len, threshold,
               do_preds=0):
    """ExTrack's own path: one recursion per track length"""
    LocErr = np.array(LOC)[None, None, None]
    LP, preds = {}, {}
    for key in sorted(all_tracks.keys(), key=int):
        Cs = np.asarray(all_tracks[key])
        out = tracking.P_Cs_inter_bound_stats_th(
            Cs, LocErr, ds, Fs, TrMat, PBL, 0 if int(key) >= max_len else 1, CELL,
            1, frame_len, do_preds=do_preds, min_len=min_len, threshold=threshold,
            max_nb_states=10 ** 9)
        lp = np.asarray(out[0])
        mx = lp.max(1)
        LP[key] = mx + np.log(np.exp(lp - mx[:, None]).sum(1))
        if do_preds:
            preds[key] = np.asarray(out[2])
    return LP, preds


def batched(all_tracks, ds, Fs, TrMat, frame_len, min_len, max_len, threshold,
            do_preds=0):
    """one recursion for the whole set, tracks retiring as they end"""
    from extrack import segmentation
    tracks, _, origin = segmentation.flatten_tracks(all_tracks)
    lengths = np.array([len(t) for t in tracks])
    order = np.argsort(-lengths, kind='stable')
    longest = int(lengths.max())
    nb_dims = tracks[0].shape[1]
    Cs = np.zeros((len(order), longest, nb_dims))
    for k, i in enumerate(order):
        Cs[k, :lengths[i]] = tracks[i]
        Cs[k, lengths[i]:] = tracks[i][-1]
    LP_flat, preds_flat = tracking.sequences_batch(
        Cs, lengths[order], np.array(LOC)[None, None, None], ds, Fs, TrMat, PBL,
        max_len, CELL, frame_len, do_preds, min_len, threshold, 10 ** 9)

    LP, preds = {}, {}
    for key in sorted(all_tracks.keys(), key=int):
        LP[key] = np.zeros(len(all_tracks[key]))
        if do_preds:
            preds[key] = np.zeros((len(all_tracks[key]), int(key), TrMat.shape[0]))
    for k, i in enumerate(order):
        key, row = origin[i]
        LP[key][row] = LP_flat[k]
        if do_preds:
            preds[key][row] = preds_flat[k, :int(key)]
    return LP, preds


tracking.set_numba('auto', threads=8)
common.set_mode('moment_matching')
print(tracking.numba_status())
print('')
print('1) no fusion (frame_len >= longest, threshold %g): bitwise identical'
      % NO_FUSION_THRESHOLD)
CASES = [
    ('lengths 5..10, 2 states', list(range(5, 11)), [40] * 6, 2),
    ('lengths 4..9, 2 states', list(range(4, 10)), [50] * 6, 2),
    ('lengths 3..8, 3 states', list(range(3, 9)), [40] * 6, 3),
    ('lengths 2..7, 2 states', list(range(2, 8)), [30] * 6, 2),
    ('one length, 10', [10], [120], 2),
    ('two lengths far apart', [5, 10], [80, 80], 2),
]
for label, lengths, counts, S in CASES:
    all_tracks, ds, Fs, TrMat = make_tracks(lengths, counts, nb_states=S)
    min_len = min(int(k) for k in all_tracks)
    max_len = max(int(k) for k in all_tracks)
    fl = max_len + 2                       # never truncates
    ref, _ = per_length(all_tracks, ds, Fs, TrMat, fl, min_len, max_len,
                        NO_FUSION_THRESHOLD)
    got, _ = batched(all_tracks, ds, Fs, TrMat, fl, min_len, max_len,
                     NO_FUSION_THRESHOLD)
    bad = [k for k in ref if not np.array_equal(ref[k], got[k])]
    detail = '%d tracks over %d lengths' % (sum(counts), len(lengths))
    if bad:
        detail += ', max diff %.3e' % max(np.abs(ref[k] - got[k]).max() for k in bad)
    check('%-26s likelihood' % label, not bad, detail)

print('')
print('2) the same, with the state predictions')
for label, lengths, counts, S in CASES[:4]:
    all_tracks, ds, Fs, TrMat = make_tracks(lengths, counts, nb_states=S)
    min_len = min(int(k) for k in all_tracks)
    max_len = max(int(k) for k in all_tracks)
    fl = max_len + 2
    ref, ref_p = per_length(all_tracks, ds, Fs, TrMat, fl, min_len, max_len,
                            NO_FUSION_THRESHOLD, do_preds=1)
    got, got_p = batched(all_tracks, ds, Fs, TrMat, fl, min_len, max_len,
                         NO_FUSION_THRESHOLD, do_preds=1)
    ok = all(np.array_equal(ref[k], got[k]) for k in ref) and \
        all(np.array_equal(ref_p[k], got_p[k]) for k in ref_p)
    worst = max(np.abs(ref_p[k] - got_p[k]).max() for k in ref_p)
    check('%-26s posteriors' % label, ok, 'max diff %.3e' % worst)

print('')
print('3) with the fusion switched back on, how far do the two paths drift?')
print('   frame_len  threshold |  max |dlogL|   mean |dlogL|   (per track, nats)')
all_tracks, ds, Fs, TrMat = make_tracks(list(range(5, 11)), [60] * 6)
min_len, max_len = 5, 10
for fl, th in [(12, 1e-12), (6, 1e-12), (4, 1e-12), (12, 0.2), (6, 0.2), (4, 0.2)]:
    ref, _ = per_length(all_tracks, ds, Fs, TrMat, fl, min_len, max_len, th)
    got, _ = batched(all_tracks, ds, Fs, TrMat, fl, min_len, max_len, th)
    d = np.concatenate([np.abs(ref[k] - got[k]) for k in ref])
    tag = '  exact' if np.all(d == 0) else ''
    print('   %-9d  %-9g |  %11.3e   %13.3e%s' % (fl, th, d.max(), d.mean(), tag))

print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), '; '.join(fails[:5])))
    raise SystemExit(1)
print('all checks passed')
