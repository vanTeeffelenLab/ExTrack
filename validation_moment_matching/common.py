#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for the validation of the moment matched fusion.

The fusion of the branches of the tree of states is the only approximation of
ExTrack's likelihood: everything else is an exact integration over the hidden
positions. Disabling the fusion therefore gives the *exact* likelihood and the
*exact* state posterior of the very same model, which is the reference every
measurement here is made against.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from extrack import tracking  # noqa: E402
from extrack import simulate_tracks  # noqa: E402

assert os.path.dirname(os.path.abspath(tracking.__file__)) == os.path.join(REPO, 'extrack'), \
    'the installed extrack is being imported instead of the local one'


# ---------------------------------------------------------------------------
# fusion modes
# ---------------------------------------------------------------------------

_TRUE_FUSE = tracking.fuse_tracks_th


def _no_fusion(m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat, nb_Tracks, nb_states=2,
               nb_dims=2, do_preds=1, threshold=0.2, frame_len=6):
    """Keep every branch of the tree of states: no approximation at all."""
    return m_arr, s2_arr, LP, cur_Bs[:, :, :1], cur_Bs_cat


def set_mode(mode):
    """
    mode in {'exact', 'legacy', 'moment_matching', 'moment_matching_isotropic'}.

    'exact'  keeps all nb_states**track_len branches (only usable on short tracks),
    'legacy' is ExTrack's historical fusion (variance = weighted mean of the
             variances of the branches),
    'moment_matching' adds the spread of the means, one variance per dimension,
    'moment_matching_isotropic' averages that spread over the dimensions.
    """
    if mode == 'exact':
        tracking.fuse_tracks_th = _no_fusion
        tracking.set_fusion_method(moment_matching=True, isotropic_variance=False)
    elif mode == 'legacy':
        tracking.fuse_tracks_th = _TRUE_FUSE
        tracking.set_fusion_method(moment_matching=False)
    elif mode == 'moment_matching':
        tracking.fuse_tracks_th = _TRUE_FUSE
        tracking.set_fusion_method(moment_matching=True, isotropic_variance=False)
    elif mode == 'moment_matching_isotropic':
        tracking.fuse_tracks_th = _TRUE_FUSE
        tracking.set_fusion_method(moment_matching=True, isotropic_variance=True)
    else:
        raise ValueError(mode)


MODES = ['legacy', 'moment_matching']


# ---------------------------------------------------------------------------
# the 2 state models of the sweep
# ---------------------------------------------------------------------------

DT = 0.02          # s between frames
LOC_ERR = 0.02     # um
NB_DIMS = 2
BIG_FOV = [1e6]    # large enough that the probability to stay in the FOV is 1

D1_LENGTHS = [0.01, 0.015, 0.02, 0.03, 0.04]   # diffusion length of state 1, um per step
TRANSITION_PS = [0.02, 0.05, 0.1, 0.2]         # transition probability per step, both ways


def D_from_length(d, dt=DT):
    """diffusion coefficient (um2/s) giving a diffusion length d (um) over one step"""
    return d ** 2 / (2 * dt)


def length_from_D(D, dt=DT):
    return np.sqrt(2 * D * dt)


SIM_SUBSTEPS = 30   # hard coded in simulate_tracks.sim_noBias


def sim_rate_for(p, nb_sub_steps=SIM_SUBSTEPS):
    """
    Transition rate to hand to sim_noBias so that the *realized* probability to
    change state between two consecutive frames is exactly p.

    sim_noBias splits every frame into nb_sub_steps substeps and applies a
    transition matrix whose off diagonal term is rate/nb_sub_steps. For a
    symmetric 2 state chain the composition over one frame gives
    0.5 * (1 - (1 - 2*rate/nb_sub_steps)**nb_sub_steps), which is inverted here.
    """
    r_sub = 0.5 * (1 - (1 - 2 * p) ** (1.0 / nb_sub_steps))
    return nb_sub_steps * r_sub


def true_model(d1, p, d0=0.0):
    """ground truth of one point of the sweep"""
    rate = sim_rate_for(p)
    return dict(Ds=np.array([D_from_length(d0), D_from_length(d1)]),
                sim_TrMat=np.array([[1 - rate, rate], [rate, 1 - rate]]),
                TrMat=np.array([[1 - p, p], [p, 1 - p]]),   # realized per frame matrix
                Fs=np.array([0.5, 0.5]),
                LocErr=LOC_ERR,
                d0=d0, d1=d1, p=p)


def simulate(d1, p, nb_tracks, track_len, seed, d0=0.0):
    """
    Fixed length tracks, no bleaching and no field of view, so that the only terms
    of the likelihood that matter are the ones the fusion acts on.
    """
    np.random.seed(seed)
    m = true_model(d1, p, d0)
    tracks, states = simulate_tracks.sim_noBias(track_lengths=[track_len],
                                                track_nb_dist=[nb_tracks],
                                                LocErr=LOC_ERR,
                                                Ds=m['Ds'],
                                                TrMat=m['sim_TrMat'],
                                                initial_fractions=m['Fs'],
                                                dt=DT,
                                                nb_dims=NB_DIMS)
    return tracks, states, m


def simulate_well_specified(d1, p, nb_tracks, track_len, seed, d0=0.0, nb_dims=NB_DIMS,
                            loc_err=LOC_ERR):
    """
    Draw tracks from ExTrack's own model, so that the fitted parameters can be
    compared to the truth without any model mismatch:

      * the states follow a Markov chain *between frames*, with probability p to
        change state from one frame to the next,
      * the displacement between frames t and t+1 has variance
        (d2[b_t] + d2[b_(t+1)])/2 per dimension, which is the "transition in the
        middle of the step" convention of ds_froms_states,
      * each position is observed with a Gaussian error of standard deviation
        loc_err.

    sim_noBias instead splits every frame into 30 substeps, so a transition can
    happen anywhere inside a frame and a particle can even switch twice; ExTrack's
    frame level model cannot represent that and absorbs it as extra transitions
    (measured: a requested p = 0.1 comes back as 0.14 whatever the fusion). That
    mismatch is identical for every fusion mode but it dominates the distance to
    the truth, which is why the sweep uses this simulator and keeps sim_noBias as a
    separate realism check.
    """
    rng = np.random.default_rng(seed)
    d2 = np.array([d0, d1], dtype=float) ** 2
    TrMat = np.array([[1 - p, p], [p, 1 - p]])

    states = np.empty((nb_tracks, track_len), dtype=int)
    states[:, 0] = (rng.random(nb_tracks) > 0.5).astype(int)   # steady state = [0.5, 0.5]
    for t in range(1, track_len):
        stay = rng.random(nb_tracks) < TrMat[states[:, t - 1], states[:, t - 1]]
        states[:, t] = np.where(stay, states[:, t - 1], 1 - states[:, t - 1])

    step_var = (d2[states[:, :-1]] + d2[states[:, 1:]]) / 2.0        # (tracks, T-1)
    steps = rng.normal(size=(nb_tracks, track_len - 1, nb_dims)) * np.sqrt(step_var)[:, :, None]
    positions = np.concatenate([np.zeros((nb_tracks, 1, nb_dims)), np.cumsum(steps, 1)], 1)
    positions = positions + rng.normal(0, loc_err, positions.shape)

    m = true_model(d1, p, d0)
    m['LocErr'] = loc_err
    return ({str(track_len): positions}, {str(track_len): states}, m)


def params_from_truth(m, vary=True):
    """lmfit parameters positioned exactly on the ground truth"""
    params = tracking.generate_params(nb_states=2,
                                      LocErr_type=1,
                                      nb_dims=NB_DIMS,
                                      LocErr_bounds=[0.002, 0.2],
                                      D_max=10,
                                      Fractions_bounds=[0.001, 0.999],
                                      estimated_LocErr=[m['LocErr']],
                                      estimated_Ds=list(m['Ds']),
                                      estimated_Fs=list(m['Fs']),
                                      estimated_transition_rates=[m['TrMat'][0, 1],
                                                                  m['TrMat'][1, 0]])
    # the fit uses Matrix_type = 1, i.e. TrMat = 1 - exp(-p): invert it so that the
    # transition probabilities of the model are exactly the simulated ones
    params['p01'].value = -np.log(1 - m['TrMat'][0, 1])
    params['p10'].value = -np.log(1 - m['TrMat'][1, 0])
    params['pBL'].value = 1e-8
    params['pBL'].vary = False
    for name in params:
        if params[name].expr is None and name != 'pBL':
            params[name].vary = vary
    return params


def tracks_dict(tracks):
    return {k: np.asarray(v) for k, v in tracks.items()}
