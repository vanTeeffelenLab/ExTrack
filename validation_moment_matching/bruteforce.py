#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent brute force likelihood of ExTrack's 2 state model.

Written with matrices on purpose, so that it shares nothing with the scalar
recursion of extrack.tracking : for one sequence of states b the increments
Delta_t = c_(t+1) - c_t are jointly Gaussian with a tridiagonal covariance

    Cov(Delta_t, Delta_t)     = (d2[b_t] + d2[b_(t+1)])/2 + 2*LocErr**2
    Cov(Delta_t, Delta_(t+1)) = -LocErr**2

(the position r_0 has a flat prior, which is what first_log_integrale_dif
implements, and the dimensions are independent). The likelihood of a track is the
log-sum-exp over the nb_states**track_len sequences of

    log F[b_0] + sum_t log TrMat[b_t, b_(t+1)] + log N(Delta ; 0, Sigma_b)
"""

import itertools

import numpy as np


def track_log_likelihood(Cs, d2s, LocErr, Fs, TrMat):
    """
    Cs: (nb_tracks, track_len, nb_dims) localizations, in chronological order.
    d2s: (nb_states,) squared diffusion lengths per step, i.e. 2*D*dt.
    returns the (nb_tracks,) log likelihood of c_1..c_(T-1) given c_0.
    """
    Cs = np.asarray(Cs, dtype=float)
    nb_tracks, T, nb_dims = Cs.shape
    nb_states = len(d2s)
    D = np.diff(Cs, axis=1)                      # (nb_tracks, T-1, nb_dims)
    n = T - 1

    per_seq = np.full((nb_tracks, nb_states ** T), -np.inf)
    for k, b in enumerate(itertools.product(range(nb_states), repeat=T)):
        b = np.array(b)
        v = (d2s[b[:-1]] + d2s[b[1:]]) / 2.0     # displacement variance of each step
        S = np.diag(v + 2 * LocErr ** 2)
        i = np.arange(n - 1)
        S[i, i + 1] = -LocErr ** 2
        S[i + 1, i] = -LocErr ** 2
        sign, logdet = np.linalg.slogdet(S)
        assert sign > 0
        sol = np.linalg.solve(S, D.transpose(1, 0, 2).reshape(n, -1))   # (n, tracks*dims)
        quad = np.sum(D.transpose(1, 0, 2).reshape(n, -1) * sol, 0).reshape(nb_tracks, nb_dims)
        lg = -0.5 * quad.sum(1) - nb_dims * (0.5 * logdet + 0.5 * n * np.log(2 * np.pi))
        lprior = np.log(Fs[b[0]]) + np.sum(np.log(TrMat[b[:-1], b[1:]]))
        per_seq[:, k] = lg + lprior

    mx = per_seq.max(1)
    return mx + np.log(np.exp(per_seq - mx[:, None]).sum(1))


def state_posterior(Cs, d2s, LocErr, Fs, TrMat):
    """exact P(b_t = s | track), same enumeration"""
    Cs = np.asarray(Cs, dtype=float)
    nb_tracks, T, nb_dims = Cs.shape
    nb_states = len(d2s)
    D = np.diff(Cs, axis=1)
    n = T - 1

    seqs = np.array(list(itertools.product(range(nb_states), repeat=T)))
    per_seq = np.empty((nb_tracks, len(seqs)))
    for k, b in enumerate(seqs):
        v = (d2s[b[:-1]] + d2s[b[1:]]) / 2.0
        S = np.diag(v + 2 * LocErr ** 2)
        i = np.arange(n - 1)
        S[i, i + 1] = -LocErr ** 2
        S[i + 1, i] = -LocErr ** 2
        sign, logdet = np.linalg.slogdet(S)
        sol = np.linalg.solve(S, D.transpose(1, 0, 2).reshape(n, -1))
        quad = np.sum(D.transpose(1, 0, 2).reshape(n, -1) * sol, 0).reshape(nb_tracks, nb_dims)
        lg = -0.5 * quad.sum(1) - nb_dims * (0.5 * logdet + 0.5 * n * np.log(2 * np.pi))
        per_seq[:, k] = lg + np.log(Fs[b[0]]) + np.sum(np.log(TrMat[b[:-1], b[1:]]))

    w = np.exp(per_seq - per_seq.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True)
    post = np.zeros((nb_tracks, T, nb_states))
    for s in range(nb_states):
        post[:, :, s] = w @ (seqs == s).astype(float)
    return post
