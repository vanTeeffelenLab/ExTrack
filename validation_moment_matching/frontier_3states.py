#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cost / accuracy frontier of the two schemes on a 3 state model.

The historical buffer is nb_states**frame_len, so frame_len is capped by cost.
The (age, state) buffer is frame_len*nb_states, so a much longer memory is
affordable: at 3 states, `ages` with frame_len = 12 still holds fewer hypotheses
(36) than `sequences` with frame_len = 4 (81). This measures whether spending the
savings on a longer memory buys the accuracy back.

Tracks are short enough (8 points) that the fusion-free likelihood, 3**8 = 6561
sequences of states, is still computable and used as the reference.
"""

import time

import numpy as np

import common
import sweep_3states as S3
from common import tracking

NB_TRACKS = 300
TRACK_LEN = 8
KS = [0.4, 1.0, 2.0, 3.0]
CONFIGS = [('sequences', L) for L in [2, 3, 4, 5]] + \
          [('ages', L) for L in [2, 3, 4, 6, 8, 12]]


def total(LP):
    LP = np.asarray(LP)
    mx = LP.max(1)
    return mx + np.log(np.exp(LP - mx[:, None]).sum(1))


def run(Cs, LocErr, ds, Fs, TrMat, scheme, L, exact=False):
    if exact:
        common.set_mode('exact')
    else:
        common.set_mode('moment_matching')
    t0 = time.time()
    lp = tracking.Proba_Cs(Cs, LocErr, ds, Fs, TrMat, 1e-8, 0, common.BIG_FOV, 1, L,
                           Cs.shape[1], 0.2, 10 ** 9 if exact else 200,
                           'sequences' if exact else scheme)
    common.set_mode('moment_matching')
    return np.asarray(lp), time.time() - t0


print('3 states, %d tracks of %d points, transition probability %.2f per step, LocErr %.3f um'
      % (NB_TRACKS, TRACK_LEN, S3.TRANSITION_P, 0.02))
print('reference = no fusion at all (3**%d = %d sequences of states kept)'
      % (TRACK_LEN, 3 ** TRACK_LEN))
print('')
header = '  scheme      L   hypotheses  branches/step |' + \
    ''.join('   k=%-5.1f' % k for k in KS) + '  |   s/eval'
print(header)
print('  ' + '-' * (len(header) - 2))

data = {}
for k in KS:
    Cs, states, ds, TrMat, Fs = S3.simulate(k, NB_TRACKS, TRACK_LEN, 4242, 0.02)
    LocErr = np.array(0.02)[None, None, None]
    ref, _ = run(Cs, LocErr, ds, Fs, TrMat, None, TRACK_LEN + 2, exact=True)
    data[k] = (Cs, LocErr, ds, TrMat, Fs, ref)

for scheme, L in CONFIGS:
    hyp = min(3 ** L, 3 ** TRACK_LEN) if scheme == 'sequences' else L * 3
    br = hyp * 3
    line = '  %-10s %2d   %8d   %11d   |' % (scheme, L, hyp, br)
    tsum = 0
    for k in KS:
        Cs, LocErr, ds, TrMat, Fs, ref = data[k]
        got, dt = run(Cs, LocErr, ds, Fs, TrMat, scheme, L)
        tsum += dt
        line += ' %9.2e' % np.abs(got - ref).mean()
    print(line + '  | %8.3f' % (tsum / len(KS)))

print('')
print('  values are the mean |log likelihood error| per track, in nats')
print('  (hypotheses/branches for `sequences` are the ceiling before the adaptive')
print('   grouping, which brings them down by a data dependent amount)')
