# Moment matched fusion, and the (age, state) buffer

Two changes to `extrack/tracking.py`, and the measurements that justify them.

## 1. The fusion is now moment matching

`fuse_tracks_th` (and `fuse_tracks_general`, `fuse_tracks`) replaces a group of
branches of the tree of states by a single Gaussian. It used to take

```
mu   = sum_b w_b m_b            (correct: the mean of the mixture)
sig2 = sum_b w_b s2_b           (too small: this is only the *within* branch part)
```

The variance of a mixture is the *law of total variance*: the average of the
branch variances **plus the spread of the branch means**. The second term was
missing, so every fused Gaussian was too narrow and the recursion behaved as if
the particle's position were known better than it is.

`fuse_gaussians` now computes, per spatial dimension and with scalar formulas only
(no covariance matrix is ever formed, so the message stays in the family the
recursion propagates):

```
LP     = log sum_b exp(LP_b)
mu_j   = sum_b w_b m_bj
sig2_j = sum_b w_b s2_b + sum_b w_b (m_bj - mu_j)**2
```

`tracking.set_fusion_method(moment_matching=False)` restores the old behaviour;
`isotropic_variance=True` averages the spread over the dimensions instead of
keeping one variance per dimension.

Two smaller fixes came with it: `log_integrale_dif`'s scalar shortcut now tests
the *total* variance (it silently dropped the per dimension localization errors
before), and `fuse_tracks_general` no longer calls `np.product`, removed in numpy 2.

## 2. `sequence_scheme = 'ages'`, transposed from ExaTrack

The historical scheme carries one hypothesis per *sequence of states* over the
last `frame_len` frames (`nb_states ** frame_len` of them), pruned at every step
by a python grouping loop of cost O(nb_branches**2) whose outcome depends on the
data. ExaTrack keeps instead a buffer of fixed size `frame_len * nb_states`
indexed by `(a, s)` = (steps since the last transition, current state), the oldest
slab meaning `a >= frame_len-1`. One step makes exactly `frame_len * nb_states**2`
branches and folds them straight back:

```
(a, s) --> (a+1, s)   staying: the age advances, nothing is fused, except in the
                      oldest slab where the two oldest ages merge
(a, s) --> (0, j)     transitioning: every source arriving in j is fused into the
                      single newborn (0, j)
```

What it keeps is what the carried Gaussian actually depends on — how much
diffusion has accumulated since the last transition. What it gives up is the
identity of the states before the current segment, and the exact age beyond
`frame_len-1`.

```python
extrack.tracking.param_fitting(..., sequence_scheme='ages')
extrack.tracking.predict_Bs(..., sequence_scheme='ages')
```

Default is `'sequences'`, i.e. unchanged behaviour. `'ages'` requires
`nb_substeps = 1` and one diffusion length per state.

## Files

| file | what it does |
|---|---|
| `common.py` | fusion mode switch (including a fusion-free `exact` mode), simulators, shared parameters |
| `bruteforce.py` | independent likelihood and state posterior by enumeration of all state sequences, written with matrices so it shares nothing with the recursion |
| `test_fuse_gaussians.py` | 18 unit checks on the fusion itself |
| `test_ages_scheme.py` | the (age, state) scheme: exactness where it must be exact, buffer size, accuracy |
| `bias_of_the_maximum.py` | where each fusion puts the maximum of the likelihood — no optimizer, no sampling noise |
| `sweep_2states.py` / `analyse_2states.py` | the 2 state grid: fits and state predictions |
| `sweep_3states.py` / `analyse_3states.py` | the 3 state grid: what the coarser buffer costs |

The **fusion-free mode is the reference for everything**: disabling the fusion
leaves ExTrack's likelihood exact, and it agrees with `bruteforce.py` to 1e-14.

`report.html` is a rendered summary of every measurement below (also published as
an artifact).

| file | what it does |
|---|---|
| `frontier_3states.py` | cost / accuracy frontier of the two schemes at 3 states |
| `results_2states_well_specified.json` | the 60 fits of the 2 state grid |
| `results_3states.json` | the 30 fits of the 3 state grid |
| `bias_T7_L6.log`, `bias_T12_L4.log` | the argmax-bias runs |


## 3. Numba acceleration

`extrack/numba_kernels.py` holds compiled twins of the three hot parts of the
likelihood, and `tracking.py` calls them when numba is installed:

| kernel | replaces |
|---|---|
| `step_kernel` | the `np.repeat` + `log_integrale_dif` + `LP += LT + LC + LL` of one recurrence step |
| `final_kernel` | the last observation of a track |
| `group_kernel` | the O(nb_branches**2) python grouping loop of `fuse_tracks_th` |
| `fuse_kernel`, `fuse_cat_kernel`, `mean_cat_kernel` | the moment matched fusion and the state histories |
| `ages_kernel` | the whole `(age, state)` recursion, one call per data set |

Both fusion schemes are covered (`sequence_scheme='sequences'` and `'ages'`) and
both fusion formulas (moment matched and legacy). The numpy code is untouched and
remains the reference and the fallback:

```python
extrack.tracking.set_numba(False)             # force the numpy path
extrack.tracking.set_numba('auto', threads=8) # the default, with a thread count
print(extrack.tracking.numba_status())
```

Why it is worth compiling: the numpy version spends only a small part of its time
on arithmetic. Measured on one likelihood evaluation, 2000 tracks of 12 points, 3
states, `frame_len` 4 -- 13-17 % in the python grouping loop, 20-33 % in the
fusion, 55-63 % in the recursion body, and the array kernels move ~0.3 GB/s
against a 10-20 GB/s single core, i.e. they are dispatch- and temporary-bound.

**Speed-up over numpy** (2000 tracks, 8 threads, medians):

| model | `sequences` | `ages` |
|---|---|---|
| 2 states, frame_len 6, 20 points | 14.5x | 17.0x |
| 3 states, frame_len 4, 12 points | 11.3x | 20.2x |
| 4 states, frame_len 4, 12 points | 11.9x | 25.9x |

The useful thread range is narrow: the recursion is a short parallel region
entered once per time step, so 8 threads is the best point on a 24-core
i9-14900K and 16 or more is slower again. The first call in a process pays a few
seconds of compilation, cached on disk afterwards.

One change to the numpy code came out of this and is not a numba matter: with
`do_preds = 0` every track carried an identical copy of the state-history array,
`(nb_tracks, nb_branches, frame_len, nb_states)`, rebuilt at every step. Only one
row is kept now. It is exact -- the histories are provably identical across
tracks in that mode -- and it is worth ~1.3x on the numpy path alone.

| file | what it does |
|---|---|
| `test_numba.py` | numba against numpy: likelihood, posteriors and the grouping decisions themselves, over 9 configurations x 2 schemes x 2 fusion formulas |
| `profile_likelihood.py` | where the numpy time goes |
| `prototype/` | a C++ and a numba prototype of the `ages` recursion, used to decide whether compiling was worth it at all |
| `Tutorial_ExTrack_executed.ipynb` | the package tutorial run end to end against this working tree |


### Running the package tutorial

`run_tutorial_cells.py` executes `Tutorials/Tutorial_ExTrack.ipynb` cell by cell
in one shared namespace and reports each cell separately, which `nbconvert`
cannot do (it stops at the first failure or timeout and says nothing about the
rest).

    python run_tutorial_cells.py --skip 50 52 54

**25 of 25 runnable cells pass, 0 failures, 316 s in total** against this working
tree: readers (csv and TrackMate xml), simulation, `param_fitting`,
`visualize_states_durations`, `predict_Bs`, the exporters, the plots,
`position_refinement`, the uncertainty scan and the per-peak localization error
path. The slowest are cell 48 (133 s), cell 36 (85 s), cell 46 (35 s), cell 17
(32 s) and cell 15, the tutorial's main fit, at **15 s**.

Cell 50 is skipped above because it is inherently very long, and it is worth
knowing why before waiting on it: it fits 2, 3 and 4 state models to 5000 tracks
with `method='powell'`, a derivative-free optimizer, and the 4 state model has
**21 free parameters**; each `n` is then refitted twice more. One likelihood
evaluation of that data set costs

| states | free params | numpy | numba | gain |
|---|---|---|---|---|
| 2 | 7 | 0.224 s | 0.024 s | 9.3x |
| 3 | 13 | 0.717 s | 0.213 s | 3.4x |
| 4 | 21 | 1.654 s | 0.159 s | 10.4x |

so the cell needs thousands of evaluations and runs for hours *with* the kernels,
and considerably longer without them. `nbconvert` defaults to a 30 s per-cell
timeout and needs `--ExecutePreprocessor.timeout` raised well past an hour for
this notebook. (Cells 52 and 54 only plot what cell 50 produces.)


## 4. Batching tracks of different lengths (after ExaTrack's `segment_tracks`)

ExTrack keys its tracks by length and runs one recursion per length: a data set
with lengths 5 to 100 pays 96 of them. `extrack/segmentation.py` transposes
ExaTrack's approach -- cut every track into segments of a fixed length, pack the
segments of all tracks into shared batches, and hand the message from one batch
to the next -- with three things adapted to ExTrack:

* consecutive segments **share their boundary point**, as in ExaTrack, so the
  displacements they fold are contiguous and the carried message (the predictive
  Gaussian of the true position) lines up across the cut;
* an **`isfirst` flag per track** says whether a segment initialises the
  recursion or resumes it from the carry buffers;
* tracks are **sorted by decreasing length**, so the tracks still alive at
  segment s are always a prefix of the batch and the idle slots collect at the
  end of the last batches. Measured slot occupancy on a 5..20 data set: 88 % with
  no cutting, 92-98 % with segments of 3 to 10 points.

Rather than padding a track's last segment and masking the arithmetic away, each
track carries **its own step count**, so a slot with no data costs nothing.

```python
packing = extrack.tracking.pack_tracks(all_tracks, segment_length=8, batch_size=1000)
LP, preds = extrack.tracking.Proba_Cs_batched(all_tracks, LocErr, ds, Fs, TrMat,
                                              pBL, cell_dims, 1, frame_len, min_len,
                                              threshold, max_nb_states, packing=packing)
```

The packing depends only on the track lengths, never on the parameters, so a fit
must build it **once**; rebuilding it inside the likelihood costs more than the
recursion (measured: 0.15-0.64x, i.e. a slowdown, when it is not hoisted).

The `ages` scheme batches unconditionally: its buffer is `frame_len*nb_states`
hypotheses whatever the track, so a fresh track and a resumed one can share a
batch. The `sequences` scheme is batched by `tracking.sequences_batch`, under one
condition -- see below.

### Bitwise identity

`test_segmentation.py` checks `np.array_equal`, not a tolerance:

| check | result |
|---|---|
| one length, 10 segment lengths x 4 configurations, against the unsegmented run | **bitwise identical** |
| tracks of many lengths in shared batches, 5 data sets x 6 packings, against ExTrack's per length path | **bitwise identical** |
| state posteriors, same treatment | **bitwise identical** (max difference exactly 0) |

### Which segment length

Time for one likelihood over the whole data set, packing hoisted, 8 threads:

| data set | length groups | per length | best batched | gain |
|---|---|---|---|---|
| lengths 5..100, 4000 tracks | 96 | 0.0809 s | 0.0429 s at segment 5 | **1.88x** |
| lengths 5..60, 3000 tracks | 56 | 0.0611 s | 0.0302 s at segment 10 | **2.02x** |
| lengths 5..20, 4000 tracks | 16 | 0.0149 s | 0.0118 s at segment 5-10 | 1.26x |

**Segments of 5 to 10 points are the best point**, and the choice matters: on the
5..100 data set, not cutting at all gives 1.30x against 1.88x at segment 5, and
segments of 20 give 1.63x. Short segments keep each kernel call's arrays compact;
below ~5 the per-segment call and carry traffic start to dominate.

The gain tracks the number of length groups, which is what the change addresses:
16 groups buy 1.26x, 96 groups buy 1.88x. It is not larger because the numba
`ages` kernel already loops over tracks independently, so the per-length grouping
was never costing an order of magnitude once that loop existed.


### The `sequences` scheme, and where its batching is exact

That scheme shares one *dynamic* set of branches across a batch, so it cannot mix
a fresh track with a resumed one. `tracking.sequences_batch` instead runs a single
recursion over the whole (length-sorted) batch and lets each track **leave the
moment its own recursion ends** -- before the fusion of that step, which is
exactly what `if current_step < nb_locs - 1` does in the per length version. The
rows being sorted, the survivors are always a prefix and leaving is a slice.

Whether that is bitwise identical to the per length path turns on which of
`fuse_tracks_th`'s two grouping routes fires:

| route | groups on | batch dependent? |
|---|---|---|
| `state_mask` | branches sharing their last `frame_len` states | **no**, it is structural |
| `m_mask * s_mask * cur_state_mask` | means and sigmas within `threshold`, sampled over the batch's 30 first tracks | **yes** |

So the adaptive route is the only obstacle, and switching it off -- a `threshold`
small enough that a branch only matches itself, `1e-12` here -- makes the batching
exact. Not zero: at exactly 0 a branch fails its own test, lands in no group, and
the reference implementation raises.

`test_segmentation_sequences.py`:

| check | result |
|---|---|
| 6 data sets (lengths 2..10, 2 and 3 states), threshold 1e-12, `frame_len >= longest` | **bitwise identical** |
| the same with the state posteriors | **bitwise identical** (max difference exactly 0) |
| threshold 1e-12 at `frame_len` 3, 4, 5, 6, 8, 12 -- i.e. with the `frame_len` truncation firing -- likelihood and posteriors | **bitwise identical** at all six |

That last row is the useful one: the `frame_len` truncation, which is the
scheme's real approximation, does *not* have to be switched off. Only the
adaptive threshold does. With it back at its production value the two paths
differ by a little rather than not at all:

| frame_len | threshold | max \|dlogL\| | mean \|dlogL\| |
|---|---|---|---|
| 12 | 0.2 | 2.9e-3 | 3.7e-5 |
| 6 | 0.2 | 2.9e-3 | 3.5e-5 |
| 4 | 0.2 | 7.6e-4 | 1.6e-5 |

in nats per track -- far below the fusion error itself, but not zero, and the
tests say so rather than hiding it behind a tolerance.

Practical reading: batch the `sequences` scheme with `threshold` at 1e-12 when the
likelihood must be reproducible to the bit, which costs the adaptive pruning and
therefore needs tracks short enough for `nb_states ** frame_len` branches; leave
`threshold` at 0.2 and accept ~1e-5 nats per track otherwise.


## 5. The fusion model in the GUI

`ExTrack_GUI.py` lets the user pick the fusion model on the **Model Fitting** and
**State Labeling** windows: a `Fusion model` dropdown (Multi-transition /
Mono-transition) above the save path, with a '?' cell on the same row (like the
other hyperparameters, see section 6) expanding the trade-off:

> Multi-transition: every sequence of states within the window is considered, so
> several transitions per window can be resolved. More accurate, but it scales
> poorly with the number of states: time proportional to
> `nb_states ** window_length`.
> Mono-transition: only the time since the last transition is kept. Time
> proportional to `window_length * nb_states**2`, so it stays fast with many
> states or long windows, at the cost of a coarser approximation.

Multi-transition is `sequence_scheme='sequences'` (the historical behaviour and
the default), Mono-transition is `'ages'`. The choice is persisted in the GUI's
`params` dict, so it carries from the fitting window to the labeling window.
Mono-transition with `Number of substeps > 1` is refused with an error window
before any fitting starts, since the (age, state) recursion does not support
substeps. The histogram and refinement windows are untouched: their backends only
exist in multi-transition form, so they show no dead option.

`test_gui_fusion_model.py` runs the real GUI module with `mainloop` stubbed,
builds the actual fitting window on `Tutorials/example_tracks.csv`, and checks
(25 checks, all passing): the dropdown's options, default and grid placement; the
info box text; that `run_predictions` with each model reproduces a direct
`predict_Bs(sequence_scheme=...)` call on the same data; that `run_fitting`
passes the scheme through for both models; and that the substeps guard refuses
Mono-transition + substeps without calling the fitter, while leaving
Multi-transition + substeps working.


## 6. Hyperparameter info cells and progress windows in the GUI

Two more GUI additions in `ExTrack_GUI.py`, verified by
`test_gui_info_progress.py` (19 checks, all passing; `test_gui_fusion_model.py`
still passes unchanged):

**'?' info cells.** Every hyperparameter on the four analysis windows (8 on
fitting, 7 on state labeling, 5 on lifetime histograms, 6 on refinement) carries
a small `?` button in the column right of its value; clicking it expands a boxed
explanation next to that row, clicking again collapses it. The texts live in one
`HYPERPARAM_INFO` dict and state what each parameter does and how it trades
accuracy against cost (the window length entry repeats the two fusion-model
scalings). The fusion-model row carries the same kind of '?' cell, expanding the
multi- vs mono-transition panel on its own row (`HYPERPARAM_INFO['fusion_model']`),
so 9 cells on the fitting window and 8 on labeling.

**Transient progress windows.** Each analysis (`run_fitting`, `run_predictions`,
`run_lifetime`, `run_refinement`) is now a thin wrapper around its original body
(`_run_*_core`): a window reading "Fitting on-going..." / "State labeling
on-going..." / "Lifetime histograms on-going..." / "Position refinement
on-going..." is drawn before the computation starts (the analyses run on the
interface's own thread, so the window is painted once with `update()` and stays
up, frozen, during the work), and at the end the same window switches to
"... finished." with the save path and an OK button. The test verifies the
on-going text from *inside* the computation, via a spy planted on the backend.
Two edge paths are covered: an analysis refused by a guard (mono-transition +
substeps) closes its progress window without claiming success, and a raising
backend turns the window into "... failed: <error>" and re-raises, so the
console traceback is preserved.


## 7. Initial vs equilibrium fractions, and error panels on loading

Verified by `test_gui_fractions_errors.py` (19 checks, all passing; the two other
GUI suites unchanged).

**Fractions.** The Parameter Window's "Fractions" row is the occupancy at the
*first time point* of the tracks (it feeds `estimated_Fs`), so it is now labeled
**Initial fractions**, with a '?' cell saying exactly that. Below it, a new
read-only **Equilibrium fractions** row shows the steady state implied by the
transition entries, recomputed live at every keystroke (degrading to `?` while an
entry is mid-edit or non-numerical). `equilibrium_fractions` uses the model's own
convention (`extract_params`, Matrix_type 1: off-diagonal `1 - exp(-rate)`, so it
matches what the fit actually assumes rather than the raw rates) and solves the
stationary vector exactly rather than by iteration, so arbitrarily slow rates are
fine. Checked against `T**200000` to 2e-12.

**Error panels.** Dataset loading behind all four analysis windows now goes
through one `load_dataset_or_error` helper: a path that does not exist, a
directory with no csv, a file whose headers do not match, and length filters
that keep zero tracks all raise the GUI's error panel (with the underlying
exception quoted) instead of a console traceback -- and the panel's Previous
button returns to the main window. This also fixes a pre-existing bug in the
fitting window, whose bare `except:` printed to the console, failed to return,
and then crashed on the undefined `tracks` variable.


## 8. State Labeling plot menu, and the duplicated tracks in the prediction plot

Verified by `test_gui_labeling_plot.py` (6 checks; the three other GUI suites
unchanged).

**Menu alignment.** The 'Plot labeled tracks' dropdown was the one value widget
gridded with `sticky='e'` and its own padding, which pushed it out of the column
the other hyperparameter values sit in. It is now gridded exactly like the
entries (plain column 1), on its row 8 with the others.

**Duplicated tracks in the plot.** Cause found and measured: each of the
8 x 8 = 64 subplot slots drew `ID = np.random.randint(len(track_list))`
independently, i.e. sampled WITH replacement. `example_tracks.csv` loads **35
tracks** at the GUI's default lengths 5-15, so with 64 slots every track
appeared ~1.8 times on average and repeats were guaranteed; even a 500-track
data set would repeat with probability 0.985 (birthday effect). The plot now
draws `min(64, nb_tracks)` tracks sampled **without replacement**
(`np.random.choice(..., replace=False)`), so a small data set shows each track
once and a large one fills the 64 slots with 64 distinct tracks. The test
records the actual `plt.plot` calls: 10 loaded tracks give exactly 10 plotted
tracks (previously 64), 100 give 64, and all plotted tracks are pairwise
distinct after mean-centering (which cancels the grid offset).


## 9. Batch analyses over a folder

Three new entries in the main window's Analysis Type dropdown -- **Batch
Fitting**, **Batch Fitting + Labeling**, **Batch All** -- take a *folder* path
and process every `.csv` (with the informed headers) and TrackMate `.xml` file
it contains. Per file, the stages run in order: fitting, then depending on the
mode, state labeling, lifetime histograms and position refinement; the fitted
parameters of a file feed that same file's later stages, exactly as running the
single-file analyses in that order would. One output csv per stage lands in the
save folder, prefixed by the input file name (`x_fitting_results.csv`,
`x_labeling.csv`, `x_lifetime_histograms.csv`, `x_refined_positions.csv`).

Semantics worth knowing: every file starts from the same user-set parameters
(snapshotted before the batch and restored at the end -- a batch never leaves
the fitted values of its last file behind); a file that fails to read or fit is
reported in the completion window and the batch continues; plots are disabled;
one progress window updates with the current file and stage; a non-folder path
or a folder without csv/xml raises the error panel; and the mono-transition +
substeps guard fires once, before any file runs.

`test_gui_batch.py` (15 checks, all passing) drives a folder holding two
synthetic csvs, one corrupt csv and the tutorial's real TrackMate xml through
'Batch All' (all four outputs per readable file, the corrupt one reported
failed, parameters restored, completion message with counts) and 'Batch
Fitting' through the window's own Start button (fitting output only, no
labeling window-length row shown in fitting-only mode). The four previous GUI
suites were re-run unchanged: 25 + 19 + 19 + 6 checks, all passing.


## 10. Browse dialogs anchored to the current path, path saved across instances

Verified by `test_gui_path_persistence.py` (16 checks; the five other GUI suites
re-run unchanged: 25 + 19 + 19 + 6 + 15).

All three Browse dialogs (dataset path, save path, save folder) now open in the
folder containing whatever their field currently holds (`initialdir_from`: a
folder anchors to itself, a file to its folder, a missing file to its surviving
folder, garbage to the home folder), instead of always starting at the user
folder. Cancelling a dialog keeps the field as it was instead of clearing it,
which the old `browser()` did.

The dataset path is remembered from one instance of the GUI to the next in
`~/.extrack_gui.json` (location overridable with the `EXTRACK_GUI_CONFIG`
environment variable, which is how the tests keep away from the real file). It
is saved both when a file is picked through Browse and when a typed path is
validated with Next; at startup the field preloads the saved path if it still
exists, its folder if only the file disappeared, and the working directory
otherwise. A failed settings write prints a message and never breaks the GUI.
The persistence test is literal: the GUI module is executed twice, and the
second instance comes up on the path the first one saved.
