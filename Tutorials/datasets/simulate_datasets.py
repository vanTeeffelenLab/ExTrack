#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the simulated replicates of this folder.

Two-state binding model, in the units ExTrack fits in (um, s):

    d0 = 0        um per step   immobile / bound state
    d1 = 0.06     um per step   mobile / unbound state
    p unbinding   0.05 per step   bound   -> mobile   (state 0 -> 1)
    p binding     0.10 per step   mobile  -> bound    (state 1 -> 0)

so the equilibrium occupancies are F0 = 0.10/0.15 = 2/3 bound and F1 = 1/3
mobile, and the tracks are started at those fractions.

`d` is the diffusion *length* per step, which is what the model carries
internally; `sim_FOV` takes diffusion coefficients, so it is handed
D = d**2 / (2*dt).

The transition probabilities need one correction. `sim_FOV` splits every frame
into `nb_sub_steps = 20` substeps and applies `TrMat / 20` at each of them, so
the matrix it is handed is *not* the per-frame transition matrix: over one frame
the composition gives visibly less than the requested rates (0.05 and 0.10 would
come out as 0.047 and 0.093). The 20th matrix root of the wanted per-frame matrix
is passed instead, so the realized per-frame probabilities are the requested
ones -- checked here against the simulated hidden states, not assumed.

Output: one csv per replicate in the 11-column TrackMate format of
`example_tracks.csv`, plus a companion `*_true_states.csv` holding the hidden
state of every peak (simulations know it; a real acquisition does not).

    python simulate_datasets.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import extrack                                          # noqa: E402

# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
D_LENGTHS = np.array([0.0, 0.06])       # um per step, states 0 (bound) and 1 (mobile)
P_UNBINDING = 0.05                      # per step, state 0 -> 1
P_BINDING = 0.10                        # per step, state 1 -> 0
DT = 0.02                               # s between frames
LOC_ERR = 0.02                          # um
PBL = 0.05                              # bleaching probability per step
CELL_DIMS = [1.0, None, None]           # depth of field in um (the GUI's default)
NB_TRACKS = 3000                        # simulated per replicate, before the FOV/bleaching filter
MIN_LEN, MAX_LEN = 5, 30
NB_REPLICATES = 3
FIELD = (80.0, 55.0)                    # um, the extent tracks are scattered over
SUB_STEPS = 20                          # hard coded in sim_FOV

TARGET_TRMAT = np.array([[1 - P_UNBINDING, P_UNBINDING],
                         [P_BINDING, 1 - P_BINDING]])
EQUILIBRIUM = np.array([P_BINDING, P_UNBINDING]) / (P_BINDING + P_UNBINDING)


def sim_trmat(target, nb_sub_steps=SUB_STEPS):
    """
    The matrix to hand `sim_FOV` so that the *realized* per-frame transition
    matrix is `target`. sim_FOV builds its substep matrix as target/nb_sub_steps
    with a corrected diagonal, so the off diagonals of the nb_sub_steps-th root,
    scaled back up, are what it needs.
    """
    root = np.real(fractional_matrix_power(target, 1.0 / nb_sub_steps))
    rates = root * nb_sub_steps
    rates[np.arange(len(target)), np.arange(len(target))] = 0
    rates[np.arange(len(target)), np.arange(len(target))] = 1 - rates.sum(1)
    return rates


def simulate(seed):
    np.random.seed(seed)
    tracks, states, sigmas = extrack.simulate_tracks.sim_FOV(
        nb_tracks=NB_TRACKS, max_track_len=MAX_LEN, min_track_len=MIN_LEN,
        LocErr=LOC_ERR, Ds=D_LENGTHS ** 2 / (2 * DT), nb_dims=2,
        initial_fractions=EQUILIBRIUM, TrMat=sim_trmat(TARGET_TRMAT),
        LocErr_std=0, dt=DT, pBL=PBL, cell_dims=CELL_DIMS)
    return tracks, states


def to_trackmate(tracks, states, seed):
    """
    The 11 columns of example_tracks.csv. Each track is offset to a random place
    in the field: ExTrack only ever reads displacements, so this changes nothing
    about the model and makes the file look like a real acquisition.
    """
    rng = np.random.default_rng(seed + 10 ** 6)
    rows, truth = [], []
    spot_id = 100000
    track_id = 0
    for key in sorted(tracks.keys(), key=int):
        block = tracks[key]
        state_block = states[key]
        for i in range(len(block)):
            xy = block[i] + rng.uniform([0.5, 0.5], [FIELD[0] - 0.5, FIELD[1] - 0.5])
            start = int(rng.integers(0, 120))
            for t in range(len(xy)):
                rows.append(('ID%d' % spot_id, spot_id, track_id,
                             round(float(rng.uniform(20, 76)), 3),
                             round(float(xy[t, 0]), 3), round(float(xy[t, 1]), 3),
                             0, int(round((start + t) * DT * 1000)), start + t,
                             0.25, 1))
                truth.append((spot_id, track_id, start + t, int(state_block[i, t])))
                spot_id += 1
            track_id += 1
    data = pd.DataFrame(rows, columns=['Label', 'ID', 'TRACK_ID', 'QUALITY',
                                       'POSITION_X', 'POSITION_Y', 'POSITION_Z',
                                       'POSITION_T', 'FRAME', 'RADIUS', 'VISIBILITY'])
    truth = pd.DataFrame(truth, columns=['ID', 'TRACK_ID', 'FRAME', 'TRUE_STATE'])
    return data.sort_values(['FRAME', 'TRACK_ID']).reset_index(drop=True), truth


def check(tracks, states):
    """what the replicate actually contains, measured rather than assumed"""
    st = np.concatenate([states[k].ravel() for k in states])
    switches = np.concatenate([(states[k][:, 1:] != states[k][:, :-1]).ravel() for k in states])
    from_0 = np.concatenate([(states[k][:, :-1] == 0).ravel() for k in states])
    p01 = switches[from_0].mean()
    p10 = switches[~from_0].mean()
    steps = np.concatenate([np.diff(tracks[k], axis=1).reshape(-1, 2) for k in tracks])
    pair = np.concatenate([((states[k][:, :-1] == 1) & (states[k][:, 1:] == 1)).ravel()
                           for k in tracks])
    # a mobile-to-mobile step carries d1**2 + 2*LocErr**2 of variance per dimension
    d1_hat = np.sqrt(max(steps[pair].var(0).mean() - 2 * LOC_ERR ** 2, 0))
    nb_tracks = sum(len(tracks[k]) for k in tracks)
    return dict(tracks=nb_tracks, peaks=sum(len(tracks[k]) * int(k) for k in tracks),
                lengths=(min(int(k) for k in tracks), max(int(k) for k in tracks)),
                p01=p01, p10=p10, f0=(st == 0).mean(), d1=d1_hat)


def main():
    print('target per-frame matrix:\n%s' % np.round(TARGET_TRMAT, 4))
    print('handed to sim_FOV (20th root, rescaled):\n%s' % np.round(sim_trmat(TARGET_TRMAT), 6))
    print('')
    header = ('  replicate | tracks  peaks   lengths |  p unbind  p bind   F0     d1')
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for rep in range(1, NB_REPLICATES + 1):
        tracks, states = simulate(seed=rep)
        data, truth = to_trackmate(tracks, states, seed=rep)
        name = 'simulated_tracks_replicate%d' % rep
        data.to_csv(os.path.join(HERE, name + '.csv'), index=False)
        truth.to_csv(os.path.join(HERE, name + '_true_states.csv'), index=False)
        c = check(tracks, states)
        print('  %-9d | %5d %7d   %2d-%-4d | %8.4f %7.4f %6.3f %6.4f'
              % (rep, c['tracks'], c['peaks'], c['lengths'][0], c['lengths'][1],
                 c['p01'], c['p10'], c['f0'], c['d1']))
    print('')
    print('  expected     %s          %8.4f %7.4f %6.3f %6.4f'
          % (' ' * 20, P_UNBINDING, P_BINDING, EQUILIBRIUM[0], D_LENGTHS[1]))


if __name__ == '__main__':
    main()
