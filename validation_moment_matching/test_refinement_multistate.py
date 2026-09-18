#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extrack.refined_localization.position_refinement against an exact reference,
for 1, 2, 3 and 4 states.

The reference enumerates every one of the nb_states**track_len state sequences;
given a sequence the positions are a linear Gaussian chain, so a Kalman filter
plus an RTS smoother give its posterior and its evidence exactly. The covariance
form rather than the precision matrix, because an immobile state has ds = 0 and
a step of variance exactly zero.

With the fusion of branches disabled the algorithm is exact, and this asserts it
to 1e-13. That was not the case before the three fixes this file was written to
find:

  A. the two END positions counted the end observation twice. get_LC_Km_Ks ends
     with `LP += log_integrated_term + LF`, and numpy `+=` is in place, so it
     mutated the array already appended as all_LP[-1]; get_pos_PDF then added
     prod_2GaussPDF's LC, which is that same evidence term. The end positions
     were off by 0.09-0.13 x LocErr.
  B. every position EXCEPT the first ignored the state fractions Fs. Of the two
     passes, the one that meets position 0 first (the one run on the reversed
     track) is the one that should carry Fs[b_0], and it was the one given
     ones(nb_states)/nb_states. Cost 0.09-0.11 x LocErr; exactly zero when the
     fractions happen to be uniform, which is why it was easy to miss.
  C. threshold = 0 -- the obvious way to ask for no fusion -- raised
     ValueError('problem with grouping: some branches were left out'): the tests
     in the grouping loop are strict (`dm / s < threshold`), so at exactly 0 a
     branch failed its own test and no branch was ever assigned a group.
  D. the transition matrix was read TRANSPOSED relative to the rest of the
     package. tracking.py anchors Fs on the first state its walk meets, these
     two recursions on the last, which reverses the direction the chain is read
     in while the transition indexing stays the same. So a TrMat fitted by
     param_fitting -- what the GUI and the tutorial pipelines hand over -- was
     interpreted backwards. Invisible whenever TrMat is symmetric, which every
     example in the tutorial happened to be; on an asymmetric 2-state chain it
     moved the refined positions by ~1 x LocErr.

None of the four was specific to 3 or 4 states; all were present at 2 states,
and A, B and D are invisible at 1 state, where a single sequence makes the
weights cancel.

The conventions below are the ones the code implements, each verified rather
than assumed -- section 1 checks the evidence of the forward pass against the
reference, which pins all three at once:

  * the variance of the step t -> t+1 is (ds[b_t]**2 + ds[b_t+1]**2) / 2, the
    average of the two states ("assuming a transition at the middle of the
    substeps", get_LC_Km_Ks);
  * the transition weight of that step is TrMat[b_t, b_t+1]: the ROW is the
    state departed from -- the convention extract_params builds and
    param_fitting reports. get_pos_PDF transposes on the way in, because the
    two recursions it drives anchor Fs at the opposite end of the track from
    tracking.py and therefore read the chain backwards (defect D below);
  * the first position carries no prior beyond its own observation.
"""

import itertools
import os
import sys
import time

import numpy as np
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import extrack                                                      # noqa: E402
from extrack.refined_localization import position_refinement, get_LC_Km_Ks  # noqa: E402

fails = []


def check(name, ok, detail=''):
    print(('  OK   ' if ok else '  FAIL ') + name + ('   ' + detail if detail else ''))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------------------
def brute_force_posterior(C, LocErr, ds, Fs, TrMat, use_Fs=True):
    """
    C : (T, nb_dims) one track. Returns (mu, sig_cond, sig_tot, logZ):
      mu        (T, nb_dims) exact posterior mean of the true positions
      sig_cond  (T,)  sqrt(E_B[Var(z_t | C, B)])  -- what the code reports
      sig_tot   (T,)  sqrt(Var(z_t | C))          -- including the spread of the
                      per-sequence means, averaged over the dimensions
      logZ      log P(C), up to a constant shared by every model and every track
    `use_Fs = False` drops the initial-state prior, which is what defect B used
    to leave the algorithm computing; it is kept so the tests can show that the
    fractions now reach the answer.
    """
    C = np.asarray(C, dtype=float)
    T, D = C.shape
    S = len(ds)
    ds2 = np.asarray(ds, dtype=float) ** 2
    R = LocErr ** 2
    M = np.asarray(TrMat, dtype=float)            # row is the state departed from

    seqs = np.array(list(itertools.product(range(S), repeat=T)))
    N = len(seqs)
    q = 0.5 * (ds2[seqs[:, :-1]] + ds2[seqs[:, 1:]])

    # forward filter; the first position has a flat prior, so its posterior is
    # its own observation and its evidence is a constant shared by every sequence
    m = np.zeros((N, T, D))
    P = np.zeros((N, T))
    m_pred = np.zeros((N, T, D))
    P_pred = np.zeros((N, T))
    loglik = np.zeros(N)
    m[:, 0] = C[0]
    P[:, 0] = R
    for t in range(1, T):
        mp = m[:, t - 1]
        Pp = P[:, t - 1] + q[:, t - 1]
        m_pred[:, t] = mp
        P_pred[:, t] = Pp
        innov = C[t][None, :] - mp
        Sv = Pp + R
        loglik += -0.5 * D * np.log(2 * np.pi * Sv) - 0.5 * np.sum(innov ** 2, axis=1) / Sv
        K = Pp / Sv
        m[:, t] = mp + K[:, None] * innov
        P[:, t] = Pp * R / Sv

    # RTS smoother
    mus = m.copy()
    var = P.copy()
    for t in range(T - 2, -1, -1):
        G = P[:, t] / P_pred[:, t + 1]
        mus[:, t] = m[:, t] + G[:, None] * (mus[:, t + 1] - m_pred[:, t + 1])
        var[:, t] = P[:, t] + G ** 2 * (var[:, t + 1] - P_pred[:, t + 1])

    logPB = np.log(Fs[seqs[:, 0]]) if use_Fs else np.zeros(N)
    for t in range(T - 1):
        logPB = logPB + np.log(M[seqs[:, t], seqs[:, t + 1]])

    logw = loglik + logPB
    top = logw.max()
    p = np.exp(logw - top)
    Z = p.sum()
    p = p / Z
    mu = np.einsum('n,ntd->td', p, mus)
    sig2_cond = p @ var
    spread = np.einsum('n,ntd->td', p, (mus - mu) ** 2).mean(axis=1)
    return mu, sig2_cond ** 0.5, (sig2_cond + spread) ** 0.5, np.log(Z) + top


def models(S, uniform_fractions=False):
    """
    S states with distinct diffusion coefficients (state 0 immobile) and an
    asymmetric transition matrix, so nothing is symmetric enough to hide an
    index mix-up.

    The ROWS are normalised: TrMat[i, j] = P(i -> j), the convention
    extract_params builds and position_refinement now takes (section 1 verifies
    it against the evidence).

    The fractions are free parameters, as they are in a fit -- deliberately NOT
    the steady state of TrMat. That matters: a matrix normalised the other way
    round has the uniform vector for its steady state, and a uniform Fs hides
    defect B completely.
    """
    D = np.array([0.0, 0.05, 0.15, 0.4, 0.9])[:S]
    ds = np.sqrt(2 * D * 0.02)
    rates = np.linspace(0.06, 0.22, S * S).reshape(S, S)
    TrMat = 1 - np.exp(-rates)
    TrMat[np.arange(S), np.arange(S)] = 0
    TrMat[np.arange(S), np.arange(S)] = 1 - TrMat.sum(1)
    if uniform_fractions:
        return ds, np.ones(S) / S, TrMat
    Fs = np.array([0.50, 0.25, 0.15, 0.07, 0.03])[:S]
    return ds, Fs / Fs.sum(), TrMat


def simulate(nb_tracks, T, ds, Fs, TrMat, LocErr, nb_dims=2, seed=0):
    """Tracks from the model the code assumes: one transition per frame, the
    mid-step variance, and TrMat read the way the recursion reads it.
    extrack.simulate_tracks runs 30 sub-steps per frame, which is a different
    process, so section 5 would otherwise measure a model mismatch rather than
    the estimator."""
    rng = np.random.default_rng(seed)
    S = len(ds)
    P = np.asarray(TrMat, dtype=float)
    states = np.zeros((nb_tracks, T), dtype=int)
    states[:, 0] = rng.choice(S, size=nb_tracks, p=Fs)
    for t in range(1, T):
        for s in range(S):
            msk = states[:, t - 1] == s
            if msk.any():
                states[msk, t] = rng.choice(S, size=int(msk.sum()), p=P[s])
    step_sig = np.sqrt(0.5 * (np.asarray(ds)[states[:, :-1]] ** 2
                              + np.asarray(ds)[states[:, 1:]] ** 2))
    steps = rng.normal(0, 1, (nb_tracks, T - 1, nb_dims)) * step_sig[:, :, None]
    truth = np.concatenate([np.zeros((nb_tracks, 1, nb_dims)),
                            np.cumsum(steps, axis=1)], axis=1)
    return truth + rng.normal(0, LocErr, truth.shape), truth, states


def reference(obs, LocErr, ds, Fs, TrMat, use_Fs=True):
    return np.array([brute_force_posterior(obs[i], LocErr, ds, Fs, TrMat, use_Fs)[0]
                     for i in range(len(obs))])


LOC_ERR = 0.03
T = 6
NB = 5

# ===========================================================================
print('')
print('0) threshold = 0 disables the fusion of branches (defect C)')
for S in [2, 3, 4]:
    ds, Fs, TrMat = models(S)
    obs, truth, states = simulate(NB, T, ds, Fs, TrMat, LOC_ERR, seed=100 + S)
    try:
        zero, _ = position_refinement({str(T): obs}, LOC_ERR, ds, Fs, TrMat,
                                      frame_len=T + 2, threshold=0.0,
                                      max_nb_states=10 ** 7)
        tiny, _ = position_refinement({str(T): obs}, LOC_ERR, ds, Fs, TrMat,
                                      frame_len=T + 2, threshold=1e-12,
                                      max_nb_states=10 ** 7)
        nb = np.asarray(get_LC_Km_Ks(obs, np.array([[[LOC_ERR]]]), ds, Fs, TrMat.T,
                                     1, T + 2, 0.0, 10 ** 7)[0]).shape[1]
        same = float(np.abs(zero[str(T)] - tiny[str(T)]).max())
        check('%d states: it runs, keeps every branch, and agrees with 1e-12' % S,
              nb == S ** T and same < 1e-12,
              '%d branches kept (nb_states**T = %d), max difference %.1e'
              % (nb, S ** T, same))
    except ValueError as error:
        check('%d states: threshold = 0 runs' % S, False, str(error))

# ===========================================================================
print('')
print('1) the forward recursion and the model conventions')
print('   log P(C) from the forward pass vs the brute force: a constant offset')
print('   is expected, the same for every track')
for S in [1, 2, 3, 4]:
    ds, Fs, TrMat = models(S)
    obs, truth, states = simulate(NB, T, ds, Fs, TrMat, LOC_ERR, seed=100 + S)
    # get_LC_Km_Ks is the internal recursion and keeps the internal orientation:
    # get_pos_PDF is what transposes a package-convention TrMat on the way in,
    # and this call bypasses it
    LP = get_LC_Km_Ks(obs, np.array([[[LOC_ERR]]]), ds, Fs, TrMat.T,
                      1, T + 2, 0.0, 10 ** 7)[0]
    got = logsumexp(np.asarray(LP), axis=1)
    ref = np.array([brute_force_posterior(obs[i], LOC_ERR, ds, Fs, TrMat)[3]
                    for i in range(NB)])
    spread = float(np.ptp(got - ref))
    check('%d state%s: the forward pass is the exact evidence'
          % (S, ' ' if S == 1 else 's'), spread < 1e-10,
          'offset spread %.1e over %d tracks' % (spread, NB))

# ===========================================================================
print('')
print('2) the smoothed positions with no fusion are the exact posterior')
print('   |refined - exact| / LocErr, every position including the two ends')
for S in [1, 2, 3, 4]:
    ds, Fs, TrMat = models(S)
    obs, truth, states = simulate(NB, T, ds, Fs, TrMat, LOC_ERR, seed=100 + S)
    t0 = time.time()
    mus, sigs = position_refinement({str(T): obs}, LOC_ERR, ds, Fs, TrMat,
                                    frame_len=T + 2, threshold=0.0,
                                    max_nb_states=10 ** 7)
    took = time.time() - t0
    ex = reference(obs, LOC_ERR, ds, Fs, TrMat)
    err = np.abs(mus[str(T)] - ex)
    check('%d state%s: exact everywhere' % (S, ' ' if S == 1 else 's'),
          err.max() / LOC_ERR < 1e-13,
          'max %.1e x LocErr (ends %.1e, interior %.1e)  (%.1f s)'
          % (err.max() / LOC_ERR, err[:, [0, -1]].max() / LOC_ERR,
             err[:, 1:-1].max() / LOC_ERR, took))

# the same with uniform fractions, which used to be the only case defect B
# left alone: it must still be exact
for S in [2, 3, 4]:
    ds, Fs, TrMat = models(S, uniform_fractions=True)
    obs, truth, states = simulate(NB, T, ds, Fs, TrMat, LOC_ERR, seed=150 + S)
    mus, _ = position_refinement({str(T): obs}, LOC_ERR, ds, Fs, TrMat,
                                 frame_len=T + 2, threshold=0.0, max_nb_states=10 ** 7)
    err = np.abs(mus[str(T)] - reference(obs, LOC_ERR, ds, Fs, TrMat)).max() / LOC_ERR
    check('%d states, uniform fractions: exact too' % S, err < 1e-13,
          'max %.1e x LocErr' % err)

# ===========================================================================
print('')
print('3) the fractions reach the answer (a regression guard on defect B)')
print('   the same tracks scored against the posterior computed WITHOUT Fs must')
print('   now DISAGREE, by about what defect B used to cost')
for S in [2, 3, 4]:
    ds, Fs, TrMat = models(S)
    obs, truth, states = simulate(NB, T, ds, Fs, TrMat, LOC_ERR, seed=100 + S)
    mus, _ = position_refinement({str(T): obs}, LOC_ERR, ds, Fs, TrMat,
                                 frame_len=T + 2, threshold=0.0, max_nb_states=10 ** 7)
    gap = np.abs(mus[str(T)] - reference(obs, LOC_ERR, ds, Fs, TrMat, use_Fs=False))
    check('%d states: dropping Fs would change the answer' % S,
          gap.max() / LOC_ERR > 0.01,
          'interior %.3f, position T-1 %.3f (x LocErr)'
          % (gap[:, 1:-1].max() / LOC_ERR, gap[:, -1].max() / LOC_ERR))

# ===========================================================================
print('')
print('4) what the fusion costs at working settings (frame_len 4, threshold 0.1)')
for S in [2, 3, 4]:
    ds, Fs, TrMat = models(S)
    obs, truth, states = simulate(NB, T, ds, Fs, TrMat, LOC_ERR, seed=200 + S)
    mus, _ = position_refinement({str(T): obs}, LOC_ERR, ds, Fs, TrMat,
                                 frame_len=4, threshold=0.1, max_nb_states=1000)
    err = np.abs(mus[str(T)] - reference(obs, LOC_ERR, ds, Fs, TrMat)) / LOC_ERR
    check('%d states: the fusion error stays well under the localization error' % S,
          err.max() < 0.25,
          'max %.3f x LocErr, rms %.4f x LocErr' % (err.max(), np.sqrt((err ** 2).mean())))

# ===========================================================================
print('')
print('5) does it help in practice? RMSE against the true positions')
print('   200 tracks of 20 points, refined with the true parameters,')
print('   at the settings the GUI uses (frame_len 7, threshold 0.1)')
quality = {}
for S in [2, 3, 4, 5]:
    ds, Fs, TrMat = models(S)
    obs, truth, states = simulate(200, 20, ds, Fs, TrMat, LOC_ERR, seed=300 + S)
    t0 = time.time()
    mus, sigs = position_refinement({'20': obs}, LOC_ERR, ds, Fs, TrMat,
                                    frame_len=7, threshold=0.1, max_nb_states=200)
    took = time.time() - t0
    raw = float(np.sqrt(np.mean((obs - truth) ** 2)))
    ref = float(np.sqrt(np.mean((mus['20'] - truth) ** 2)))
    quality[S] = (raw, ref, took)
    check('%d states: refinement beats the raw positions' % S, ref < raw,
          'raw %.4f -> %.4f um (%.0f%% of the error removed, %.1f s)'
          % (raw, ref, 100 * (1 - ref / raw), took))
    per_state = []
    for s in range(S):
        msk = states == s
        if msk.sum() < 50:
            continue
        per_state.append((s, float(np.sqrt(np.mean((obs - truth)[msk] ** 2))),
                          float(np.sqrt(np.mean((mus['20'] - truth)[msk] ** 2)))))
    check('   and on every state taken separately',
          all(f < r for _, r, f in per_state),
          ', '.join('D%d %.4f->%.4f' % (s, r, f) for s, r, f in per_state))

# short tracks, where the two end positions are a third of the data and defect A
# used to bite hardest
print('')
print('   short tracks (length 5), where the ends are most of the track')
for S in [2, 3, 4]:
    ds, Fs, TrMat = models(S)
    obs, truth, states = simulate(400, 5, ds, Fs, TrMat, LOC_ERR, seed=350 + S)
    mus, _ = position_refinement({'5': obs}, LOC_ERR, ds, Fs, TrMat,
                                 frame_len=7, threshold=0.1, max_nb_states=200)
    exact, _ = position_refinement({'5': obs}, LOC_ERR, ds, Fs, TrMat,
                                   frame_len=7, threshold=0.0, max_nb_states=10 ** 7)
    raw = float(np.sqrt(np.mean((obs - truth) ** 2)))
    ref = float(np.sqrt(np.mean((mus['5'] - truth) ** 2)))
    ex = reference(obs[:20], LOC_ERR, ds, Fs, TrMat)
    # threshold 0.1 still merges branches on a track this short (frame_len is
    # never reached, but the mean/sigma test is), so exactness is asserted with
    # the fusion off and the residual of the working settings only reported
    check('%d states: still helps, and is exact with the fusion off' % S,
          ref < raw and np.abs(exact['5'][:20] - ex).max() / LOC_ERR < 1e-13,
          'raw %.4f -> %.4f um (%.0f%% removed); exact to %.1e, fusion costs %.1e (x LocErr)'
          % (raw, ref, 100 * (1 - ref / raw),
             np.abs(exact['5'][:20] - ex).max() / LOC_ERR,
             np.abs(mus['5'][:20] - ex).max() / LOC_ERR))

# ===========================================================================
print('')
print('6) is the reported sigma the actual spread of the error?')
for S in [2, 3, 4]:
    ds, Fs, TrMat = models(S)
    obs, truth, states = simulate(200, 20, ds, Fs, TrMat, LOC_ERR, seed=400 + S)
    mus, sigs = position_refinement({'20': obs}, LOC_ERR, ds, Fs, TrMat,
                                    frame_len=7, threshold=0.1, max_nb_states=200)
    resid = mus['20'] - truth
    actual = float(np.sqrt(np.mean(resid ** 2)))
    claimed = float(np.sqrt(np.mean(sigs['20'] ** 2)))
    check('%d states: the reported sigma is within 25%% of the actual spread' % S,
          0.75 < claimed / actual < 1.25,
          'reported %.4f, actual %.4f, ratio %.3f, std(z) %.3f'
          % (claimed, actual, claimed / actual,
             float(np.std(resid / sigs['20'][:, :, None]))))

print('')
print('   the reported sigma averages the per-sequence variances and ignores the')
print('   spread between their means -- by construction, not a defect:')
for S in [2, 3, 4]:
    ds, Fs, TrMat = models(S)
    obs, truth, states = simulate(NB, T, ds, Fs, TrMat, LOC_ERR, seed=500 + S)
    both = [brute_force_posterior(obs[i], LOC_ERR, ds, Fs, TrMat)[1:3] for i in range(NB)]
    c = float(np.sqrt(np.mean([x[0] ** 2 for x in both])))
    t = float(np.sqrt(np.mean([x[1] ** 2 for x in both])))
    print('     %d states: E_B[Var] %.4f against the full Var %.4f  (%.1f%% low)'
          % (S, c, t, 100 * (1 - c / t)))

print('')
print('summary: error removed on 20-point tracks  '
      + ', '.join('%d st %.0f%%' % (S, 100 * (1 - r / w))
                  for S, (w, r, _) in sorted(quality.items())))

print('')
if fails:
    print('FAILED (%d): %s' % (len(fails), '; '.join(fails[:8])))
    raise SystemExit(1)
print('all checks passed')
