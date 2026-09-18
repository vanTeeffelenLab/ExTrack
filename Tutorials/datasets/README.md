# Simulated datasets — 3 replicates of a 2-state binding model

Three independent replicates of the same model, in the 11-column TrackMate csv
format of `../example_tracks.csv`, so they read with ExTrack's default headers
(`POSITION_X`, `POSITION_Y`, `FRAME`, `TRACK_ID`) and can be dropped straight
into the GUI — including its batch analyses, which take this folder as their
path.

| file | content |
|---|---|
| `simulated_tracks_replicate1.csv` … `3.csv` | the tracks, TrackMate format |
| `simulated_tracks_replicate1_true_states.csv` … `3.csv` | the hidden state of every peak (`ID`, `TRACK_ID`, `FRAME`, `TRUE_STATE`) |
| `simulate_datasets.py` | regenerates all of the above |

## The model

| parameter | value |
|---|---|
| `d0` (bound, state 0) | **0** µm per step |
| `d1` (mobile, state 1) | **0.06** µm per step (= D1 0.09 µm²/s at this dt) |
| p unbinding (0 → 1) | **0.05** per step |
| p binding (1 → 0) | **0.10** per step |
| localization error | 0.02 µm |
| frame time `dt` | 0.02 s |
| bleaching | 0.05 per step |
| depth of field | 1 µm (the GUI's default) |
| initial fractions | the equilibrium, F0 = 2/3 bound, F1 = 1/3 mobile |

`d` is the diffusion *length* per step, which is what the model carries
internally; the simulator is handed `D = d²/(2·dt)`.

## Two things done deliberately

**The transition probabilities are the realized ones.** `sim_FOV` splits every
frame into 20 substeps and applies `TrMat/20` at each, so the matrix it is
*given* is not the per-frame transition matrix: passing 0.05 and 0.10 directly
would have produced 0.047 and 0.093 in the data. The generator passes the 20th
matrix root of the wanted per-frame matrix instead, and checks the result
against the simulated hidden states rather than assuming it.

**Tracks are scattered over an 80 × 55 µm field** with random start frames, so
the files look like a real acquisition. ExTrack only ever reads displacements,
so this changes nothing about the model.

## What the data actually contains

Measured on the simulated hidden states (not assumed):

| replicate | tracks | peaks | lengths | p unbind | p bind | F0 | d1 |
|---|---|---|---|---|---|---|---|
| 1 | 2615 | 44329 | 5–30 | 0.0484 | 0.1032 | 0.681 | 0.0594 |
| 2 | 2645 | 44644 | 5–30 | 0.0476 | 0.1058 | 0.687 | 0.0590 |
| 3 | 2575 | 43942 | 5–30 | 0.0498 | 0.1030 | 0.677 | 0.0593 |
| *target* | | | | *0.0500* | *0.1000* | *0.667* | *0.0600* |

The observed bound fraction sits slightly above the equilibrium 2/3 because
bound particles neither leave the field of view nor get cut short as often as
mobile ones — a censoring effect of the depth-of-field model, present in real
data too.

## Recovered by ExTrack

Fitted with 2 states, `frame_len` 6, `threshold` 0.2, depth of field 1,
localization error fitted, tracks of 5–30 points:

| replicate | d0 | d1 | p unbind | p bind | LocErr | F0 |
|---|---|---|---|---|---|---|
| 1 | 0.0004 | 0.0601 | 0.0528 | 0.1004 | 0.0200 | 0.644 |
| 2 | 0.0005 | 0.0597 | 0.0521 | 0.1049 | 0.0201 | 0.637 |
| 3 | 0.0007 | 0.0598 | 0.0535 | 0.1011 | 0.0201 | 0.648 |
| *true* | *0* | *0.0600* | *0.0500* | *0.1000* | *0.0200* | *0.667* |

Diffusion lengths and localization error come back to well under 1 %, the
transition probabilities to a few per cent. The small upward bias on the
unbinding rate is expected and not a defect of the data: the simulator lets a
particle switch state *inside* a frame (20 substeps), while ExTrack's model
places at most one transition per frame at its middle, and it absorbs the
difference as slightly more transitions. `F0` is the fitted *initial* fraction,
compared here against the equilibrium.
