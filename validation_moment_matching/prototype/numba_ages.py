#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The same (age, state) recursion again, this time as a numba kernel.

Same purpose as the C++ prototype: measure, not guess. numba matters here because
ExTrack is a pip-installable package for people on Windows, and `pip install
numba` is a far smaller distribution burden than shipping a compiled DLL. The
question is how much of the C++ win it gives back.
"""

import numpy as np
from numba import njit, prange

LOG_2PI = 1.8378770664093454835606594728112


@njit(cache=True, fastmath=False, inline='always')
def _fuse(bLP, bm, bs2, idx, nsrc, D, out_m, out_s2, off):
    """moment matched fusion of `nsrc` branches listed in idx[:nsrc]"""
    mx = -1e308
    for k in range(nsrc):
        if bLP[idx[k]] > mx:
            mx = bLP[idx[k]]
    W = 0.0
    for k in range(nsrc):
        W += np.exp(bLP[idx[k]] - mx)
    inv = 1.0 / W
    for d in range(D):
        out_m[off + d] = 0.0
    for k in range(nsrc):
        wk = np.exp(bLP[idx[k]] - mx) * inv
        b = idx[k]
        for d in range(D):
            out_m[off + d] += wk * bm[b * D + d]
    for d in range(D):
        out_s2[off + d] = 0.0
    for k in range(nsrc):
        wk = np.exp(bLP[idx[k]] - mx) * inv
        b = idx[k]
        for d in range(D):
            dm = bm[b * D + d] - out_m[off + d]
            out_s2[off + d] += wk * (bs2[b * D + d] + dm * dm)
    return mx + np.log(W)


@njit(cache=True, parallel=True, fastmath=False)
def ages_loglik(Cs, log_tr, log_fs, pair_d2, l2, frame_len):
    nb_tracks, track_len, D = Cs.shape
    S = log_fs.shape[0]
    L = max(frame_len, 2)
    cap = L * S
    bcap = cap * S
    out = np.empty(nb_tracks)

    for t in prange(nb_tracks):
        LP = np.empty(cap)
        m = np.empty(cap * D)
        s2 = np.empty(cap * D)
        nLP = np.empty(cap)
        nm = np.empty(cap * D)
        ns2 = np.empty(cap * D)
        bLP = np.empty(bcap)
        bm = np.empty(bcap * D)
        bs2 = np.empty(bcap * D)
        idx = np.empty(bcap, dtype=np.int64)

        n_act = 1
        for s in range(S):
            LP[s] = log_fs[s]
            for d in range(D):
                m[s * D + d] = Cs[t, 0, d]
                s2[s * D + d] = l2

        for step in range(track_len - 1):
            P_now = n_act * S
            if step > 0:
                for c in range(P_now):
                    K = 0.0
                    for d in range(D):
                        sv = s2[c * D + d]
                        mv = m[c * D + d]
                        tot = sv + l2
                        dv = Cs[t, step, d] - mv
                        K += -0.5 * (LOG_2PI + np.log(tot)) - dv * dv / (2.0 * tot)
                        m[c * D + d] = (mv * l2 + Cs[t, step, d] * sv) / tot
                        s2[c * D + d] = sv * l2 / tot
                    LP[c] += K

            for c in range(P_now):
                s = c % S
                for j in range(S):
                    b = c * S + j
                    bLP[b] = LP[c] + log_tr[s, j]
                    add = pair_d2[s, j]
                    for d in range(D):
                        bm[b * D + d] = m[c * D + d]
                        bs2[b * D + d] = s2[c * D + d] + add

            n_new = min(n_act + 1, L)

            for j in range(S):
                nsrc = 0
                for a in range(n_act):
                    for s in range(S):
                        if s != j:
                            idx[nsrc] = (a * S + s) * S + j
                            nsrc += 1
                nLP[j] = _fuse(bLP, bm, bs2, idx, nsrc, D, nm, ns2, j * D)

            n_copy = n_act if n_act < L else L - 2
            for a in range(n_copy):
                for s in range(S):
                    src = (a * S + s) * S + s
                    dst = (a + 1) * S + s
                    nLP[dst] = bLP[src]
                    for d in range(D):
                        nm[dst * D + d] = bm[src * D + d]
                        ns2[dst * D + d] = bs2[src * D + d]
            if n_act >= L:
                for s in range(S):
                    idx[0] = ((L - 2) * S + s) * S + s
                    idx[1] = ((L - 1) * S + s) * S + s
                    dst = (L - 1) * S + s
                    nLP[dst] = _fuse(bLP, bm, bs2, idx, 2, D, nm, ns2, dst * D)

            for c in range(cap):
                LP[c] = nLP[c]
            for c in range(cap * D):
                m[c] = nm[c]
                s2[c] = ns2[c]
            n_act = n_new

        P_now = n_act * S
        mx = -1e308
        for c in range(P_now):
            K = 0.0
            for d in range(D):
                tot = s2[c * D + d] + l2
                dv = Cs[t, track_len - 1, d] - m[c * D + d]
                K += -0.5 * (LOG_2PI + np.log(tot)) - dv * dv / (2.0 * tot)
            LP[c] += K
            if LP[c] > mx:
                mx = LP[c]
        W = 0.0
        for c in range(P_now):
            W += np.exp(LP[c] - mx)
        out[t] = mx + np.log(W)
    return out


def loglik(Cs, ds, Fs, TrMat, frame_len, loc_err):
    d2 = np.asarray(ds, dtype=np.float64) ** 2
    return ages_loglik(np.ascontiguousarray(Cs, dtype=np.float64),
                       np.ascontiguousarray(np.log(TrMat), dtype=np.float64),
                       np.ascontiguousarray(np.log(Fs), dtype=np.float64),
                       np.ascontiguousarray((d2[:, None] + d2[None, :]) / 2.0),
                       float(loc_err) ** 2, int(frame_len))
