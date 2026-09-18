#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batching tracks of different lengths, after ExaTrack's `segment_tracks`.

ExTrack keeps its tracks in a dict keyed by track length and runs one recursion
per length. A data set with lengths 5 to 20 therefore pays 16 separate
recursions, most of them on a few hundred tracks, and the per-call cost is
carried 16 times. ExaTrack instead cuts every track into segments of a fixed
length, packs the segments of *all* tracks into common batches, and carries the
hidden state from one batch to the next; a per-track `isfirst` flag says whether
a segment starts a track (initialise) or continues one (use the carried state).

Transposed here:

  * consecutive segments **share their boundary point**, exactly as in ExaTrack:
    segment i covers points `[i*(L-1), i*(L-1) + L - 1]`, so the displacements it
    folds are contiguous with the previous segment's. That is what makes the
    carried message -- the predictive Gaussian of the true position -- line up;
  * tracks are **sorted by decreasing length** and packed in that order, so the
    tracks still alive at segment s are always a prefix of the batch. Unused
    slots collect at the end of the last batches instead of being scattered
    through all of them;
  * the last segment of a track is usually shorter than `segment_length`. Rather
    than padding it and masking the arithmetic away, each track carries its own
    number of steps, and a track stops contributing the moment it ends. Nothing
    is computed for a slot that has no data.

`mask` and `isfirst` are still produced, in ExaTrack's layout, because they are
what the caller reasons about; the recursion consumes the per-track step counts
derived from them.
"""

import numpy as np


def flatten_tracks(all_tracks, input_LocErr=None):
    """
    Turn ExTrack's {length: (nb_tracks, length, nb_dims)} dict into a flat list,
    keeping the (length key, row) of every track so results can be scattered back.
    """
    tracks, locerrs, origin = [], [], []
    for key in sorted(all_tracks.keys(), key=int):
        block = np.asarray(all_tracks[key])
        if len(block) == 0:
            continue
        errs = None if input_LocErr is None else np.asarray(input_LocErr[key])
        for i in range(len(block)):
            tracks.append(block[i])
            locerrs.append(None if errs is None else errs[i])
            origin.append((key, i))
    return tracks, locerrs, origin


def segment_tracks(tracks, segment_length, batch_size, locerrs=None,
                   min_segment_steps=1):
    """
    Pack tracks into batches of segments, longest first.

    tracks : list of (track_len, nb_dims) arrays.
    segment_length : number of *points* per segment; a segment folds
        `segment_length - 1` displacements and shares its first point with the
        previous segment. `None` or 0 means "one segment per track", i.e. no
        cutting, which is what the rest of the machinery falls back to.
    batch_size : maximum number of tracks per batch.
    locerrs : optional list of per-peak localization errors, same shapes.
    min_segment_steps : a trailing segment folding fewer displacements than this
        is merged into the previous one rather than started.

    Returns a list of batches, each a dict with

        Cs        (n_active, seg_points, nb_dims)   the points of this segment
        LocErr    (n_active, seg_points, nb_err) or None
        nsteps    (n_active,)   displacements folded by each track here
        isfirst   (n_active,)   1 when the segment starts its track
        islast    (n_active,)   1 when the track ends inside this segment
        mask      (n_active, seg_points)  1 on the points a track really owns
        rows      (n_active,)   index into the sorted track list
        chunk     which chunk of `batch_size` tracks this batch belongs to

    Batches are ordered chunk by chunk and, inside a chunk, segment by segment,
    which is the order the carried state has to be consumed in.
    """
    lengths = np.array([len(t) for t in tracks])
    order = np.argsort(-lengths, kind='stable')          # longest first
    nb_dims = tracks[0].shape[1]

    batches = []
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        chunk_len = lengths[chunk]
        steps = chunk_len - 1                            # displacements per track
        seg_steps = (segment_length - 1) if segment_length else int(steps.max())
        seg_steps = max(int(seg_steps), 1)

        done = np.zeros(len(chunk), dtype=int)           # displacements already folded
        seg_id = 0
        while np.any(done < steps):
            active = np.where(done < steps)[0]           # a prefix, tracks being sorted
            remaining = steps[active] - done[active]
            take = np.minimum(remaining, seg_steps)
            # a trailing sliver is folded into this segment instead of a new one
            sliver = (remaining - take) > 0
            sliver &= (remaining - take) < min_segment_steps
            take[sliver] = remaining[sliver]

            seg_points = int(take.max()) + 1
            Cs = np.zeros((len(active), seg_points, nb_dims))
            mask = np.zeros((len(active), seg_points))
            err = None
            if locerrs is not None and locerrs[0] is not None:
                nb_err = np.asarray(locerrs[chunk[active[0]]]).shape[-1]
                err = np.zeros((len(active), seg_points, nb_err))
            for k, a in enumerate(active):
                t = tracks[chunk[a]]
                lo = done[a]
                hi = lo + take[k] + 1                    # +1: the shared boundary point
                seg = t[lo:hi]
                Cs[k, :len(seg)] = seg
                Cs[k, len(seg):] = seg[-1]               # padding, never folded
                mask[k, :len(seg)] = 1
                if err is not None:
                    e = np.asarray(locerrs[chunk[a]])[lo:hi]
                    err[k, :len(e)] = e
                    err[k, len(e):] = e[-1]

            batches.append(dict(
                Cs=Cs, LocErr=err, mask=mask,
                nsteps=take.astype(np.int64),
                isfirst=(done[active] == 0).astype(np.int64),
                islast=((done[active] + take) == steps[active]).astype(np.int64),
                rows=chunk[active].astype(np.int64),
                start=done[active].astype(np.int64),
                chunk=start // batch_size, segment=seg_id))
            done[active] += take
            seg_id += 1

    return batches, order, lengths


def describe(batches, tracks):
    """a one line summary of how well a packing uses its batches"""
    slots = sum(b['Cs'].shape[0] * (b['Cs'].shape[1] - 1) for b in batches)
    used = sum(int(b['nsteps'].sum()) for b in batches)
    return ('%d batches, %d track-steps in %d slots (%.1f %% used), '
            'largest batch %d x %d'
            % (len(batches), used, slots, 100.0 * used / max(slots, 1),
               max(b['Cs'].shape[0] for b in batches),
               max(b['Cs'].shape[1] for b in batches)))
