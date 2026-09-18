ExTrack
-------

This repository contains the necessary scripts to run the method ExTrack. ExTrack is a method to detemine kinetics of particles able to transition between different motion states. It can assess diffusion coefficients, transition rates, localization error as well as annotating the probability for any track to be in each state for every time points. It can produce histograms of durations in each state to highlight none-markovian transition kinetics. Eventually it can be used to refine the localization precision of tracks by considering the most likely positions which is especially efficient when the particle do not move.

More details on the methods are available in the Journal of Cell Biology https://rupress.org/jcb/article/222/5/e202208059/213911/ExTrack-characterizes-transition-kinetics-and.

ExTrack has been designed and implemented by François Simon in the laboratory of Sven van Teeffelen at University of Montreal. ExTrack is primarely implemented as a python package and as a stand-alone software. The stand alone version of ExTrack can be download at https://zenodo.org/records/15133436. See the wiki https://github.com/vanTeeffelenLab/ExTrack/wiki or the pdf ExTrack_GUI_manual.pdf (not implemented yet) in this repository for detailed informations on how to use the software. Currently supported OS: Windows.

See the Wiki section for more information on how to install and use ExTrack (python package and stand-alone software).

https://pypi.org/project/extrack/

# Dependencies

- numpy
- lmfit
- xmltodict
- matplotlib
- pandas

Optional:

- numba: compiles the inner recursion. The kernels are reported at 11 to 26
  times the numpy path on the recursion itself; end to end a full fit gains
  less, because the optimizer and the parameter handling do not speed up --
  measured at 2.9x on the fitting cell of the tutorial (93.3 s against 32.2 s),
  with the fitted likelihood identical to the last digit. ExTrack runs without
  it (the numpy code stays the reference); install it and it is picked up
  automatically.
  `extrack.tracking.set_numba(mode, threads)` turns it on or off and sets the
  thread count, `extrack.tracking.numba_status()` reports what is in use.
- jupyter, for the tutorial notebooks.
- cupy, for the GPU path (see Parallelization below).

# Installation (from pip)

(needs to be run in anaconda prompt for anaconda users on windows)

## Install dependencies

`pip install numpy lmfit xmltodict matplotlib pandas`

and, optionally, `pip install numba`

## Install ExTrack

`pip install extrack`

https://pypi.org/project/extrack/

The release on PyPI can lag behind this repository. In particular, the position
refinement in this repository has been verified position by position against an
exact reference (every sequence of states enumerated, the posterior of each
solved in closed form) for 1, 2, 3 and 4 states -- see
`validation_moment_matching/test_refinement_multistate.py`. Install from this
GitHub repository to get that version.

## Input file format

ExTrack can deal with tracks saved with TrackMate xml format or csv format by using the integrated readers https://github.com/vanTeeffelenLab/ExTrack/wiki/Loading-data-sets.

# Installation from this GitHub repository

## From Unix/Mac:

`sudo apt install git` (if git is not installed)

`git clone https://github.com/vanTeeffelenLab/ExTrack.git`

`cd ExTrack`

`sudo python setup.py install`

## From Windows using anaconda prompt:

Need to install git if not already installed.

`git clone https://github.com/vanTeeffelenLab/ExTrack.git` One can also just manually download the package if git is not installed. Once extracted the folder may be named ExTrack-main

`cd ExTrack` or cd `ExTrack-main`

`python setup.py install` from the ExTrack directory

# Tutorial

Tutorials for the python package of ExTrack are available.

A first tutorial allows the user to have an overview of all the possibilities of the different modules of ExTrack (https://github.com/vanTeeffelenLab/ExTrack/blob/main/Tutorials/Tutorial_ExTrack.ipynb). This jupyter notebook tutorial shows the whole pipeline:
- Loading data sets (https://github.com/vanTeeffelenLab/ExTrack/wiki/Loading-data-sets).
- Initialize parameters of the model (https://github.com/vanTeeffelenLab/ExTrack/wiki/Parameters-for-fitting).
- Fitting.
- Probabilistic state annotation.
- Histograms of state duration.
- Position refinement.
- Saving results.

from loading data sets to saving results
at these location: 
- tests/test_extrack.py
- or Tutorials/tutorial_extrack.ipynb

These contain the most important modules in a comprehensive framework. We recommand following the tutorial tutorial_extrack.ipynb which uses Jupyter notebook as it is more didactic. One has to install jupyter to use it: `pip install jupyter` in the anaconda prompt for conda users.

# Usage
## Units
The distance units of the parameters (input and output) are the same unit as the units of the tracks. Our initial parameters are chosen to work for micron units but initial parameters can be changed to match other units. The rate parameters are rates per frame. Rates per second can be inferred from the rates per frame by dividing them by the time in between frames.

## Main functions

extrack.tracking.param_fitting : performs the fit to infer the parameters of a given data set.

extrack.visualization.visualize_states_durations : plot histograms of the duration in each state.

extrack.tracking.predict_Bs : predicts the states of the tracks.

## Extra functions

extrack.simulate_tracks.sim_FOV : allows to simulate tracks.

extrack.exporters.extrack_2_pandas : turn the outputs from ExTrack to a pandas dataframe. outputed dataframe can be save with dataframe.to_csv(save_path)

extrack.exporters.save_extrack_2_xml : save extrack data to xml file (trackmate format).

extrack.visualization.visualize_tracks : show all tracks in a single plot.

extrack.visualization.plot_tracks : show the longest tracks on separated plots

## Caveats

# References

# License
This program is released under the GNU General Public License version 3 or upper (GPLv3+).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Parallelization

Multiple CPU Parallelization can be performed in get_2DSPT_params with the argument worker the number of cores used for the job (equal to 1 by default).
Warning: Do not work on windows.

On Windows (and anywhere else), the numba kernels are the way to use several
cores: `extrack.tracking.set_numba('auto', threads = 8)`. The recursion is a
short parallel region entered once per time step, so the useful range is narrow
-- measured on a 24 core i9-14900K the best point is around 8 threads, and 16 or
more is slower again because the thread pool then costs more than the step.

GPU parallelization used to be available but may not be compatible with the current CPU parallelization, GPU parallelization uses the package cupy which can be installed as described here : https://github.com/cupy/cupy. The cupy version will depend on your cuda version which itself must be compatible with your GPU driver and GPU. Usage of cupy requires a change in the module extrack/tracking (line 4) : GPU_computing = True

# Graphical User interface of ExTrack

The Graphical User interface of ExTrack can be used with the script ExTrack_GUI.py.

It offers four single-dataset analyses -- model fitting, state labeling, state
lifetime histograms and position refinement -- and three batch analyses that run
over every csv/xml file of a folder: `Batch Fitting`, `Batch Fitting + Labeling`
and `Batch All`. Browse takes either a file or a folder (a folder is what the
batch analyses need). A batch writes its results to a `Results` folder next to
the dataset folder rather than among the data: one track file per replicate
holding the state predictions, the refined positions and their localization
error together, plus a single `batch_fitting_summary.csv` gathering the fitted
parameters of every replicate, which is also shown as a table when the batch
ends.

# Stand-alone Version

The stand alone version of ExTrack can be download at https://zenodo.org/records/15133436. See the wiki https://github.com/vanTeeffelenLab/ExTrack/wiki or the pdf ExTrack_GUI_manual.pdf (not implemented yet) in this repository for detailed informations on how to use the software (on going).
Currently supported OS: Windows

To create a stand-alone version of ExTrack for your own OS, you can follow the following steps:
1) pip install pyinstaller
2) pyinstaller --onedir path\ExTrack_GUI.py
3) Copy the .ddl files starting with mkl into the dist\ExTrack_GUI\_internal (the mkl files can be found in C:\Users\Franc\anaconda3\Library\bin in my case)
4) execute dist\ExTrack_GUI.exe to run the stand alone software

# Authors
François Simon

# Bugs/suggestions
Sent an email at simon.francois \at protonmail.com
