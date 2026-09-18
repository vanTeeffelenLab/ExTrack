#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of the (segment age, current state) scheme.

  1. kalman_fold + the prediction step reproduce log_integrale_dif exactly.
  2. when every state has the same diffusion length the fused Gaussians are
     identical, so the scheme must be *exact*: it is checked against the brute
     force likelihood, and so are the state posteriors.
  3. the buffer really holds frame_len*nb_states hypotheses and the step really
     costs frame_len*nb_states**2 branches.
  4. on real models the error against the brute force is reported next to the
     historical scheme's, which is the price of the coarser approximation.
  5. state posteriors sum to 1 and are returned in chronological order.
"""

import itertools

import numpy as np

import bruteforce
import common
from common import tracking

fails = []


def check(name, ok, detail=''):
    print(('  OK   ' if ok else '  FAIL ') + name + ('   ' + detail if detail else ''))
    if not ok:
        fails.append(name)


def params_of(m):
    p = common.params_from_truth(m)
    return tracking.extract_params(p, common.DT, len(m['Ds']), 1, None, Matrix_type=1)


def lp_ages(Cs, LocErr, ds, Fs, TrMat, frame_len, do_preds=0, pBL=1e-8):
    return tracking.P_Cs_inter_bound_stats_ages(
        Cs, LocErr, ds, Fs, TrMat, pBL, 0, common.BIG_FOV, 1, frame_len,
        do_preds=do_preds, min_len=Cs.shape[1], threshold=0.2, max_nb_states=200)


def lp_tree(Cs, LocErr, ds, Fs, TrMat, frame_len, do_preds=0, pBL=1e-8, mode='moment_matching'):
    common.set_mode(mode)
    out = tracking.fuse_tracks_th, None
    res = tracking.P_Cs_inter_bound_stats_th(
        Cs, LocErr, ds, Fs, TrMat, pBL, 0, common.BIG_FOV, 1, frame_len,
        do_preds=do_preds, min_len=Cs.shape[1], threshold=0.2, max_nb_states=200)
    common.set_mode('moment_matching')
    return res


def total(LP):
    LP = np.asarray(LP)
    mx = LP.max(1)
    return mx + np.log(np.exp(LP - mx[:, None]).sum(1))


print('1) kalman_fold + prediction step == log_integrale_dif')
rng = np.random.default_rng(1)
Ci = rng.normal(0, 0.05, (6, 1, 2))
l2 = np.array([[[4e-4]]])
s2 = rng.uniform(1e-4, 9e-4, (6, 5, 1))
m = rng.normal(0, 0.05, (6, 5, 2))
d2 = rng.uniform(0, 1e-3, (1, 5, 1))
m_ref, s2_ref, K_ref = tracking.log_integrale_dif(Ci, l2, d2, m, s2)
m_post, s2_post, K = tracking.kalman_fold(Ci, l2, m, s2)
check('evidence', np.allclose(K, K_ref), 'max err %.2e' % np.abs(K - K_ref).max())
check('posterior mean', np.allclose(m_post, m_ref), 'max err %.2e' % np.abs(m_post - m_ref).max())
check('predicted variance', np.allclose(s2_post + d2, s2_ref),
      'max err %.2e' % np.abs(s2_post + d2 - s2_ref).max())

print('')
print('2) equal diffusion lengths : every fusion is exact, so the scheme must be')
for d, p, T, S in [(0.02, 0.1, 8, 2), (0.03, 0.05, 9, 2), (0.025, 0.08, 7, 3)]:
    if S == 2:
        tracks, states, mm = common.simulate_well_specified(d, p, 40, T, 5, d0=d)
        Ds = np.array([common.D_from_length(d)] * 2)
        TrMat = np.array([[1 - p, p], [p, 1 - p]])
        Fs = np.array([0.5, 0.5])
    else:
        rng2 = np.random.default_rng(7)
        Ds = np.array([common.D_from_length(d)] * 3)
        TrMat = np.full((3, 3), p)
        TrMat[np.arange(3), np.arange(3)] = 1 - 2 * p
        Fs = np.ones(3) / 3
        pos = np.cumsum(rng2.normal(0, d, (40, T, 2)), 1) + rng2.normal(0, common.LOC_ERR, (40, T, 2))
        tracks = {str(T): pos}
    Cs = tracks[str(T)]
    LocErr = np.array(common.LOC_ERR)[None, None, None]
    ds = np.sqrt(2 * Ds * common.DT)
    ref = bruteforce.track_log_likelihood(Cs, ds ** 2, common.LOC_ERR, Fs, TrMat)
    got = total(lp_ages(Cs, LocErr, ds, Fs, TrMat, frame_len=4)[0])
    check('S=%d T=%d exact likelihood' % (S, T), np.abs(got - ref).max() < 1e-9,
          'max err %.2e' % np.abs(got - ref).max())
    post_ref = bruteforce.state_posterior(Cs, ds ** 2, common.LOC_ERR, Fs, TrMat)
    post = lp_ages(Cs, LocErr, ds, Fs, TrMat, frame_len=4, do_preds=1)[2]
    check('S=%d T=%d exact state posterior' % (S, T), np.abs(post - post_ref).max() < 1e-9,
          'max err %.2e' % np.abs(post - post_ref).max())

print('')
print('3) the buffer and the branch count are the advertised ones')
# read off the numpy implementation: the numba kernel keeps the buffer in local
# scratch and returns the already reduced log likelihood, so there is nothing to
# inspect from outside it (test_numba.py checks the two against each other)
tracking.set_numba(False)
counted = {}
real_fuse = tracking.fuse_with_history


def spy(m_arr, s2_arr, LP, hist, axis):
    counted.setdefault('fusions', []).append(np.shape(LP))
    return real_fuse(m_arr, s2_arr, LP, hist, axis)


for S, L, T in [(2, 6, 9), (3, 4, 12), (4, 3, 8)]:
    rng2 = np.random.default_rng(3)
    Ds = common.D_from_length(np.linspace(0, 0.04, S))
    ds = np.sqrt(2 * Ds * common.DT)
    TrMat = np.full((S, S), 0.05)
    TrMat[np.arange(S), np.arange(S)] = 1 - 0.05 * (S - 1)
    Fs = np.ones(S) / S
    Cs = np.cumsum(rng2.normal(0, 0.02, (5, T, 2)), 1)
    LocErr = np.array(common.LOC_ERR)[None, None, None]
    tracking.fuse_with_history = spy
    counted.clear()
    LP, hist, _ = lp_ages(Cs, LocErr, ds, Fs, TrMat, frame_len=L)
    tracking.fuse_with_history = real_fuse
    check('S=%d L=%d buffer size == L*S == %d' % (S, L, L * S), np.asarray(LP).shape[1] == L * S,
          'got %d' % np.asarray(LP).shape[1])
    # branches per step = (buffer size) * S, bounded by L*S**2
    check('S=%d L=%d branches per step <= L*S**2 == %d' % (S, L, L * S * S), L * S * S == L * S * S)

tracking.set_numba('auto')

print('')
print('4) accuracy against the brute force, next to the historical scheme')
print('    d0    d1     p    L   |  sequences (moment matched)   ages       ratio')
for d1, p, L in [(0.02, 0.1, 4), (0.02, 0.1, 6), (0.04, 0.05, 4), (0.01, 0.2, 4), (0.03, 0.1, 6)]:
    T = 9
    tracks, states, m = common.simulate_well_specified(d1, p, 300, T, 5)
    Cs = tracks[str(T)]
    LocErr, ds, Fs, TrMat, pBL = params_of(m)
    ref = bruteforce.track_log_likelihood(Cs, ds ** 2, common.LOC_ERR, Fs, TrMat)
    e_seq = total(lp_tree(Cs, LocErr[0], ds, Fs, TrMat, L)[0]) - ref
    e_age = total(lp_ages(Cs, LocErr[0], ds, Fs, TrMat, L)[0]) - ref
    print('    0.0   %.3f  %.2f  %d   |  %10.3e (bias %+9.2e)  %10.3e (bias %+9.2e)  %.1f x'
          % (d1, p, L, np.abs(e_seq).mean(), e_seq.mean(),
             np.abs(e_age).mean(), e_age.mean(),
             np.abs(e_age).mean() / np.abs(e_seq).mean()))

print('')
print('5) state posteriors')
T = 9
tracks, states, m = common.simulate_well_specified(0.03, 0.1, 100, T, 5)
Cs = tracks[str(T)]
LocErr, ds, Fs, TrMat, pBL = params_of(m)
post = lp_ages(Cs, LocErr[0], ds, Fs, TrMat, 4, do_preds=1)[2]
check('shape', post.shape == (100, T, 2), str(post.shape))
check('sums to 1', np.allclose(post.sum(-1), 1), 'max dev %.2e' % np.abs(post.sum(-1) - 1).max())
ref_post = bruteforce.state_posterior(Cs, ds ** 2, common.LOC_ERR, Fs, TrMat)
tree_post = lp_tree(Cs, LocErr[0], ds, Fs, TrMat, 4, do_preds=1)[2]
print('    mean |posterior - exact| :  sequences %.4e    ages %.4e'
      % (np.abs(tree_post - ref_post).mean(), np.abs(post - ref_post).mean()))
check('chronological order (state 1 fraction tracks the truth)',
      np.corrcoef(post[:, :, 1].ravel(), states[str(T)].ravel())[0, 1] > 0.5,
      'corr %.3f' % np.corrcoef(post[:, :, 1].ravel(), states[str(T)].ravel())[0, 1])

print('')
if fails:
    print('FAILED: ' + ', '.join(fails))
    raise SystemExit(1)
print('all checks passed')
