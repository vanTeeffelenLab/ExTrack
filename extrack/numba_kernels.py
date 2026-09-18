#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba kernels for the ExTrack likelihood.

The numpy implementation in `tracking.py` stays the reference and the fallback:
these kernels reproduce it operation for operation, and every one of them is
checked against it. They exist because the numpy version spends most of its time
on things that are not arithmetic. Measured on one likelihood evaluation, 2000
tracks of 12 points, 3 states, frame_len 4:

    ~13-17 %  the grouping loop of fuse_tracks_th, a python loop over pairs of
              branches, O(nb_branches**2) numpy calls on 30-track slices
    ~20-33 %  the fusion itself
    ~55-63 %  the recursion body: repeat, log_integrale_dif, the accumulations

and the numpy kernels move only ~0.3 GB/s against a 10-20 GB/s single core, i.e.
they are dominated by per-call dispatch and temporaries rather than by the work.
Rewriting the same arithmetic as explicit loops removes both.

Nothing here is required: `tracking.py` imports this module defensively and falls
back to numpy when numba is missing, when the inputs fall outside what the
kernels cover, or when the user turns it off with `tracking.set_numba(False)`.
"""

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:                                     # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):                          # no-op decorator
        def wrap(f):
            return f
        return wrap if (args and callable(args[0])) is False else args[0]

    prange = range

_JIT = dict(cache=True, fastmath=False)

# ---------------------------------------------------------------------------
# helpers shared by the kernels
# ---------------------------------------------------------------------------


# Axes that were left at length 1 for broadcasting are indexed inline as
# `(t if arr.shape[k] > 1 else 0)`: a helper function for that could not be
# typed inside a parallel region.


# ---------------------------------------------------------------------------
# the sequences scheme: one recurrence step
# ---------------------------------------------------------------------------


@njit(parallel=True, **_JIT)
def step_kernel(Ci, l2, cur_d2s, m_in, s2_in, LP_in, LT, LL, rep,
                m_out, s2_out, LP_out):
    """
    np.repeat of the carried arrays, then log_integrale_dif, then
    `LP += LT + LC + LL`, fused into one pass.

    Branch `b` of the output comes from branch `b // rep` of the input, which is
    what np.repeat(..., rep, axis=1) produces, and its new state is `b % rep`.

    Ci       : (nb_tracks, nb_dims)          the observation folded at this step
    l2       : (nb_tracks|1, nb_dims|1)      squared localization error
    cur_d2s  : (nb_tracks|1, nb_branches_out, 1)
    m_in     : (nb_tracks, nb_branches_in, nb_dims)
    s2_in    : (nb_tracks|1, nb_branches_in, nb_dims|1)
    LT, LL   : (nb_branches_out,)
    """
    nT, nB, nD = m_out.shape
    # the reference takes a shortcut when the total variance is shared by every
    # dimension: one logarithm, multiplied by nb_dims. Reproduced so the two
    # implementations round the same way.
    scalar_var = (l2.shape[1] == 1) and (s2_in.shape[2] == 1)
    l_st = 0 if l2.shape[0] == 1 else 1
    ld_st = 0 if l2.shape[1] == 1 else 1
    d_st = 0 if cur_d2s.shape[0] == 1 else 1
    s_st = 0 if s2_in.shape[0] == 1 else 1
    sd_st = 0 if s2_in.shape[2] == 1 else 1
    for t in prange(nT):
        li = t * l_st
        di = t * d_st
        si = t * s_st
        for b in range(nB):
            src = b // rep
            d2 = cur_d2s[di, b, 0]
            if scalar_var:
                # the variance is shared by all dimensions: one log, repeated
                sv = s2_in[si, src, 0]
                lv = l2[li, 0]
                tot = lv + sv
                quad = 0.0
                for d in range(nD):
                    dv = Ci[t, d] - m_in[t, src, d]
                    quad += dv * dv / (2.0 * tot)
                    m_out[t, b, d] = (m_in[t, src, d] * lv + Ci[t, d] * sv) / tot
                    s2_out[t, b, d] = (d2 * lv + d2 * sv + lv * sv) / tot
                K = nD * (-0.5 * np.log(2.0 * np.pi * tot)) - quad
            else:
                K = 0.0
                for d in range(nD):
                    lv = l2[li, d * ld_st]
                    sv = s2_in[si, src, d * sd_st]
                    tot = lv + sv
                    dv = Ci[t, d] - m_in[t, src, d]
                    K += -0.5 * np.log(2.0 * np.pi * tot) - dv * dv / (2.0 * tot)
                    m_out[t, b, d] = (m_in[t, src, d] * lv + Ci[t, d] * sv) / tot
                    s2_out[t, b, d] = (d2 * lv + d2 * sv + lv * sv) / tot
            LP_out[t, b] = LP_in[t, src] + K + LT[b] + LL[b]


@njit(parallel=True, **_JIT)
def final_kernel(C0, l2, m_arr, s2_arr, LP, LL):
    """the last observation: LP += sum_dims log N(C0 ; m, s2 + l2) + LL"""
    nT, nB, nD = m_arr.shape
    l_st = 0 if l2.shape[0] == 1 else 1
    ld_st = 0 if l2.shape[1] == 1 else 1
    s_st = 0 if s2_arr.shape[0] == 1 else 1
    sd_st = 0 if s2_arr.shape[2] == 1 else 1
    ll_st = 0 if LL.shape[0] == 1 else 1
    for t in prange(nT):
        li = t * l_st
        si = t * s_st
        for b in range(nB):
            K = 0.0
            for d in range(nD):
                lv = l2[li, d * ld_st]
                sv = s2_arr[si, b, d * sd_st]
                tot = sv + lv
                dv = C0[t, d] - m_arr[t, b, d]
                K += -0.5 * np.log(2.0 * np.pi * tot) - dv * dv / (2.0 * tot)
            LP[t, b] += K + LL[t * ll_st, b]


# ---------------------------------------------------------------------------
# the sequences scheme: grouping and fusion
# ---------------------------------------------------------------------------


@njit(**_JIT)
def group_kernel(m_s, s_s, cat_arg, state0, hist_len, frame_len, threshold):
    """
    The grouping loop of fuse_tracks_th, as explicit loops.

    A branch joins the group of `Bs_ID` when, over the first `test_chunks`
    tracks, its mean and its standard deviation are within `threshold` of the
    reference relative to sigma for more than 80 % of the (track, variance
    component) pairs *and* it has the same current state, OR when its last
    `frame_len` states agree with the reference on every tested track.

    m_s     : (tc_m, nb_branches, nb_dims)          means
    s_s     : (tc_s, nb_branches, nb_var_dims)      standard deviations
    cat_arg : (tc_c, nb_branches, min(frame_len, hist_len))  argmax states

    The three track counts differ on the first step, where the variances and the
    histories are still shared by every track and their arrays have a length-1
    track axis that numpy broadcasts; the reference then averages the mean test
    over tc_m tracks and the sigma test over one. That asymmetry is reproduced
    here rather than smoothed over, because it decides which branches merge.

    state0  : (nb_branches,)  current state of each branch, from track 0

    returns (group_of, nb_groups) with group_of[b] the group index of branch b,
    numbered in the order the reference implementation creates them.
    """
    tc_m, nB, nD = m_s.shape
    tc_s = s_s.shape[0]
    tc_c = cat_arg.shape[0]
    nV = s_s.shape[2]
    nF = cat_arg.shape[2]
    s_st = 0 if tc_s == 1 else 1
    group_of = np.full(nB, -1, dtype=np.int64)
    ng = 0
    for ref in range(nB):
        if group_of[ref] != -1:
            continue
        for b in range(nB):
            if group_of[b] != -1:
                continue
            if b == ref:
                # a branch always belongs to its own group. The tests below
                # are strict (`< threshold`), so at threshold = 0 -- the way
                # to ask for no fusion at all -- a branch would fail its own
                # test and be left without a group.
                group_of[b] = ng
                continue
            # (a) same last frame_len states on every tested track
            same_states = hist_len > frame_len
            if same_states:
                for t in range(tc_c):
                    ok = True
                    for f in range(nF):
                        if cat_arg[t, ref, f] != cat_arg[t, b, f]:
                            ok = False
                            break
                    if not ok:
                        same_states = False
                        break
            if same_states:
                group_of[b] = ng
                continue
            # (b) same current state, and close in mean and in sigma
            if state0[b] != state0[ref]:
                continue
            hits_m = 0
            for t in range(tc_m):
                ts = t * s_st
                dm = 0.0
                for d in range(nD):
                    dm += abs(m_s[t, b, d] - m_s[t, ref, d])
                dm /= nD
                for v in range(nV):
                    if dm / s_s[ts, b, v] < threshold:
                        hits_m += 1
            if not hits_m > 0.8 * (tc_m * nV):
                continue
            hits_s = 0
            for t in range(tc_s):
                ds_ = 0.0
                for v in range(nV):
                    ds_ += abs(s_s[t, b, v] - s_s[t, ref, v])
                ds_ /= nV
                for v in range(nV):
                    if ds_ / s_s[t, b, v] < threshold:
                        hits_s += 1
            if hits_s > 0.8 * (tc_s * nV):
                group_of[b] = ng
        ng += 1
    return group_of, ng


@njit(parallel=True, **_JIT)
def fuse_kernel(m_arr, s2_arr, LP, member, gptr, moment_matching, isotropic,
                new_m, new_s2, new_LP):
    """
    Moment matched fusion of every group, over all tracks.

    member/gptr are the CSR-style group membership: group g owns the branches
    member[gptr[g]:gptr[g+1]].
    """
    nT = m_arr.shape[0]
    nD = m_arr.shape[2]
    nV = new_s2.shape[2]
    ng = gptr.shape[0] - 1
    s_st = 0 if s2_arr.shape[0] == 1 else 1
    sv_st = 0 if s2_arr.shape[2] == 1 else 1
    for t in prange(nT):
        si = t * s_st
        for g in range(ng):
            lo = gptr[g]
            hi = gptr[g + 1]
            mx = -1.0e308
            for k in range(lo, hi):
                v = LP[t, member[k]]
                if v > mx:
                    mx = v
            W = 0.0
            for k in range(lo, hi):
                W += np.exp(LP[t, member[k]] - mx)
            inv = 1.0 / W
            # first moment
            for d in range(nD):
                acc = 0.0
                for k in range(lo, hi):
                    b = member[k]
                    acc += np.exp(LP[t, b] - mx) * inv * m_arr[t, b, d]
                new_m[t, g, d] = acc
            # second moment: the average width plus the spread of the means
            for v in range(nV):
                acc = 0.0
                for k in range(lo, hi):
                    b = member[k]
                    w = np.exp(LP[t, b] - mx) * inv
                    acc += w * s2_arr[si, b, v * sv_st]
                new_s2[t, g, v] = acc
            if moment_matching:
                if isotropic:
                    sp = 0.0
                    for d in range(nD):
                        a = 0.0
                        for k in range(lo, hi):
                            b = member[k]
                            dm = m_arr[t, b, d] - new_m[t, g, d]
                            a += np.exp(LP[t, b] - mx) * inv * dm * dm
                        sp += a
                    sp /= nD
                    for v in range(nV):
                        new_s2[t, g, v] += sp
                else:
                    for v in range(nV):
                        a = 0.0
                        for k in range(lo, hi):
                            b = member[k]
                            dm = m_arr[t, b, v] - new_m[t, g, v]
                            a += np.exp(LP[t, b] - mx) * inv * dm * dm
                        new_s2[t, g, v] += a
            new_LP[t, g] = mx + np.log(W)


@njit(parallel=True, **_JIT)
def fuse_cat_kernel(cat, LP, member, gptr, new_cat):
    """
    weighted average of the state histories, with the fusion weights (do_preds).
    A group of one is copied rather than averaged, as in the reference.
    """
    nT, _, H, S = new_cat.shape
    nC = cat.shape[0]
    ng = gptr.shape[0] - 1
    c_st = 0 if nC == 1 else 1
    for t in prange(nT):
        ct = t * c_st
        for g in range(ng):
            lo = gptr[g]
            hi = gptr[g + 1]
            if hi - lo == 1:
                b = member[lo]
                for h in range(H):
                    for s in range(S):
                        new_cat[t, g, h, s] = cat[ct, b, h, s]
                continue
            mx = -1.0e308
            for k in range(lo, hi):
                v = LP[t, member[k]]
                if v > mx:
                    mx = v
            W = 0.0
            for k in range(lo, hi):
                W += np.exp(LP[t, member[k]] - mx)
            for h in range(H):
                for s in range(S):
                    acc = 0.0
                    for k in range(lo, hi):
                        acc += np.exp(LP[t, member[k]] - mx) * cat[ct, member[k], h, s]
                    new_cat[t, g, h, s] = acc / W


@njit(**_JIT)
def mean_cat_kernel(cat, member, gptr, tc, new_cat):
    """
    the do_preds = 0 branch: the histories then only serve to decide which
    branches share a state sequence, so the reference averages them over the
    tested tracks and over the group and gives every track the same value --
    except for a group of one, which is copied.
    """
    nT, _, H, S = new_cat.shape
    nC = cat.shape[0]
    ng = gptr.shape[0] - 1
    for g in range(ng):
        lo = gptr[g]
        hi = gptr[g + 1]
        if hi - lo == 1:
            b = member[lo]
            for t in range(nT):
                ct = t * (0 if nC == 1 else 1)
                for h in range(H):
                    for s in range(S):
                        new_cat[t, g, h, s] = cat[ct, b, h, s]
            continue
        n = tc * (hi - lo)
        for h in range(H):
            for s in range(S):
                acc = 0.0
                for t in range(tc):
                    for k in range(lo, hi):
                        acc += cat[t, member[k], h, s]
                v = acc / n
                for t in range(nT):
                    new_cat[t, g, h, s] = v


# ---------------------------------------------------------------------------
# the (age, state) scheme: the whole recursion in one kernel
# ---------------------------------------------------------------------------


@njit(inline='always', **_JIT)
def _fuse_into(bLP, bm, bs2, bcat, idx, nsrc, D, S, H, want_cat,
               out_m, out_s2, out_cat, off_c, moment_matching, isotropic):
    """moment matched fusion of nsrc branches, returning the fused log weight"""
    mx = -1.0e308
    for k in range(nsrc):
        if bLP[idx[k]] > mx:
            mx = bLP[idx[k]]
    W = 0.0
    for k in range(nsrc):
        W += np.exp(bLP[idx[k]] - mx)
    inv = 1.0 / W
    for d in range(D):
        out_m[off_c * D + d] = 0.0
    for k in range(nsrc):
        w = np.exp(bLP[idx[k]] - mx) * inv
        b = idx[k]
        for d in range(D):
            out_m[off_c * D + d] += w * bm[b * D + d]
    for d in range(D):
        out_s2[off_c * D + d] = 0.0
    for k in range(nsrc):
        w = np.exp(bLP[idx[k]] - mx) * inv
        b = idx[k]
        for d in range(D):
            out_s2[off_c * D + d] += w * bs2[b * D + d]
    if moment_matching:
        if isotropic:
            sp = 0.0
            for d in range(D):
                a = 0.0
                for k in range(nsrc):
                    w = np.exp(bLP[idx[k]] - mx) * inv
                    dm = bm[idx[k] * D + d] - out_m[off_c * D + d]
                    a += w * dm * dm
                sp += a
            sp /= D
            for d in range(D):
                out_s2[off_c * D + d] += sp
        else:
            for d in range(D):
                a = 0.0
                for k in range(nsrc):
                    w = np.exp(bLP[idx[k]] - mx) * inv
                    dm = bm[idx[k] * D + d] - out_m[off_c * D + d]
                    a += w * dm * dm
                out_s2[off_c * D + d] += a
    if want_cat:
        for h in range(H):
            for s in range(S):
                a = 0.0
                for k in range(nsrc):
                    w = np.exp(bLP[idx[k]] - mx) * inv
                    a += w * bcat[(idx[k] * H + h) * S + s]
                out_cat[(off_c * H + h) * S + s] = a
    return mx + np.log(W)


@njit(parallel=True, **_JIT)
def ages_kernel(Cs, l2, log_tr, log_fs, pair_d2, Lp_stay, end_LL, frame_len,
                min_len, moment_matching, isotropic, want_cat,
                nsteps, isfirst, islast, abs_start,
                carry_LP, carry_m, carry_s2, carry_cat, carry_nact,
                out_LP, out_cat):
    """
    The (segment age, current state) recursion of P_Cs_inter_bound_stats_ages,
    over one segment of each track.

    A whole data set is the special case `isfirst = islast = 1`,
    `abs_start = 0`, `nsteps = track_len - 1`, which is how
    P_Cs_inter_bound_stats_ages calls it; the segmented driver walks the same
    code with the carry buffers instead. Every track advances on its own step
    count, so a batch holding tracks of different lengths computes nothing for
    the steps a short track does not have.

    Cs       : (nb_tracks, seg_points, nb_dims)
    l2       : (nb_tracks|1, seg_points|1, nb_dims|1)  squared localization error
    nsteps   : (nb_tracks,)  displacements this segment folds for each track
    isfirst  : (nb_tracks,)  1 to initialise, 0 to resume from the carry buffers
    islast   : (nb_tracks,)  1 when the track ends inside this segment
    abs_start: (nb_tracks,)  index of the segment's first point inside the track
    carry_*  : (nb_tracks, ...) the message, read when isfirst = 0 and always
               written back, so the next segment resumes from it
    out_LP   : (nb_tracks,)   written for the tracks that end here
    out_cat  : (nb_tracks, hist_len, nb_states)  state posteriors, if want_cat
    """
    nT = Cs.shape[0]
    D = Cs.shape[2]
    S = log_fs.shape[0]
    L = frame_len if frame_len > 2 else 2
    cap = L * S
    bcap = cap * S
    H = out_cat.shape[1] if want_cat else 1
    l_st = 0 if l2.shape[0] == 1 else 1
    lt_st = 0 if l2.shape[1] == 1 else 1
    ld_st = 0 if l2.shape[2] == 1 else 1

    for t in prange(nT):
        LP = np.empty(cap)
        m = np.empty(cap * D)
        s2 = np.empty(cap * D)
        nLP = np.empty(cap)
        nm = np.empty(cap * D)
        ns2 = np.empty(cap * D)
        bLP = np.empty(bcap)
        bm = np.empty(bcap * D)
        bs2 = np.empty(bcap * D)
        cat = np.zeros(cap * H * S)
        bcat = np.zeros(bcap * H * S)
        ncat = np.zeros(cap * H * S)
        idx = np.empty(bcap, dtype=np.int64)

        li_t = t * l_st
        a0 = abs_start[t]

        if isfirst[t] == 1:
            n_act = 1
            for s in range(S):
                LP[s] = log_fs[s]
                for d in range(D):
                    m[s * D + d] = Cs[t, 0, d]
                    s2[s * D + d] = l2[li_t, 0, d * ld_st]
                if want_cat:
                    cat[(s * H + 0) * S + s] = 1.0
        else:
            n_act = carry_nact[t]
            for c in range(cap):
                LP[c] = carry_LP[t, c]
            for c in range(cap * D):
                m[c] = carry_m[t, c]
                s2[c] = carry_s2[t, c]
            if want_cat:
                for c in range(cap * H * S):
                    cat[c] = carry_cat[t, c]

        for step in range(nsteps[t]):
            abs_step = a0 + step               # index of the point folded here
            P_now = n_act * S
            # ---- fold the observation, except at the very first transition ----
            if abs_step > 0:
                lt = step * lt_st
                for c in range(P_now):
                    K = 0.0
                    for d in range(D):
                        lv = l2[li_t, lt, d * ld_st]
                        sv = s2[c * D + d]
                        mv = m[c * D + d]
                        tot = sv + lv
                        dv = Cs[t, step, d] - mv
                        K += -0.5 * np.log(2.0 * np.pi * tot) - dv * dv / (2.0 * tot)
                        m[c * D + d] = (mv * lv + Cs[t, step, d] * sv) / tot
                        s2[c * D + d] = sv * lv / tot
                    LP[c] += K
            LL_on = (abs_step + 1) >= min_len   # mirrors `if current_step >= min_len`

            # ---- branch over the next state ----
            for c in range(P_now):
                s = c % S
                for j in range(S):
                    b = c * S + j
                    bLP[b] = LP[c] + log_tr[s, j]
                    if LL_on and abs_step > 0:
                        bLP[b] += Lp_stay[j]
                    add = pair_d2[s, j]
                    for d in range(D):
                        bm[b * D + d] = m[c * D + d]
                        bs2[b * D + d] = s2[c * D + d] + add
                    if want_cat:
                        for h in range(H):
                            for q in range(S):
                                bcat[(b * H + h) * S + q] = cat[(c * H + h) * S + q]

            n_new = n_act + 1 if n_act + 1 < L else L

            # ---- newborns ----
            for j in range(S):
                nsrc = 0
                for a in range(n_act):
                    for s in range(S):
                        if s != j:
                            idx[nsrc] = (a * S + s) * S + j
                            nsrc += 1
                nLP[j] = _fuse_into(bLP, bm, bs2, bcat, idx, nsrc, D, S, H, want_cat,
                                    nm, ns2, ncat, j, moment_matching, isotropic)
            # ---- stays ----
            n_copy = n_act if n_act < L else L - 2
            for a in range(n_copy):
                for s in range(S):
                    src = (a * S + s) * S + s
                    dst = (a + 1) * S + s
                    nLP[dst] = bLP[src]
                    for d in range(D):
                        nm[dst * D + d] = bm[src * D + d]
                        ns2[dst * D + d] = bs2[src * D + d]
                    if want_cat:
                        for h in range(H):
                            for q in range(S):
                                ncat[(dst * H + h) * S + q] = bcat[(src * H + h) * S + q]
            if n_act >= L:
                for s in range(S):
                    idx[0] = ((L - 2) * S + s) * S + s
                    idx[1] = ((L - 1) * S + s) * S + s
                    dst = (L - 1) * S + s
                    nLP[dst] = _fuse_into(bLP, bm, bs2, bcat, idx, 2, D, S, H, want_cat,
                                          nm, ns2, ncat, dst, moment_matching, isotropic)

            # ---- the state reached at this step is known for every target ----
            if want_cat:
                for a in range(n_new):
                    for s in range(S):
                        c = a * S + s
                        for q in range(S):
                            ncat[(c * H + abs_step + 1) * S + q] = 0.0
                        ncat[(c * H + abs_step + 1) * S + s] = 1.0

            for c in range(cap):
                LP[c] = nLP[c]
            for c in range(cap * D):
                m[c] = nm[c]
                s2[c] = ns2[c]
            if want_cat:
                for c in range(cap * H * S):
                    cat[c] = ncat[c]
            n_act = n_new

        if islast[t] == 0:
            # ---- hand the message to the next segment ----
            carry_nact[t] = n_act
            for c in range(cap):
                carry_LP[t, c] = LP[c]
            for c in range(cap * D):
                carry_m[t, c] = m[c]
                carry_s2[t, c] = s2[c]
            if want_cat:
                for c in range(cap * H * S):
                    carry_cat[t, c] = cat[c]
            continue

        # ---- last observation, then the end of track term ----
        last = nsteps[t]
        P_now = n_act * S
        lt = last * lt_st
        mx = -1.0e308
        for c in range(P_now):
            K = 0.0
            for d in range(D):
                lv = l2[li_t, lt, d * ld_st]
                tot = s2[c * D + d] + lv
                dv = Cs[t, last, d] - m[c * D + d]
                K += -0.5 * np.log(2.0 * np.pi * tot) - dv * dv / (2.0 * tot)
            LP[c] += K
            if islast[t] == 2:                 # 2 = the track ends by leaving or bleaching
                s = c % S
                acc = 0.0
                for j in range(S):
                    acc += np.exp(log_tr[s, j] + end_LL[j])
                LP[c] += np.log(acc)
            if LP[c] > mx:
                mx = LP[c]
        W = 0.0
        for c in range(P_now):
            W += np.exp(LP[c] - mx)
        out_LP[t] = mx + np.log(W)

        if want_cat:
            for h in range(H):
                for q in range(S):
                    acc = 0.0
                    for c in range(P_now):
                        acc += np.exp(LP[c] - mx) * cat[(c * H + h) * S + q]
                    out_cat[t, h, q] = acc / W
