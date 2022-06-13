ExTrack
-------

This repository contains the necessary scripts to run the method ExTrack. ExTrack is a method to detemine kinetics of particles able to transition between different motion states. It can assess diffusion coefficients, transition rates, localization error as well as annotating the probability for any track to be in each state for every time points. It can produce histograms of durations in each state to highlight no markovian transition kinetics. Eventually it can be used to refine the localization precision of tracks by considering the most likely positions which is especially efficient when the particle do not move.

ExTrack has been designed and implemented by François Simon. The python implementation of ExTrack can profite from GPU parallelization using the cupy library. An additionnal version of ExTrack is available on Fiji via Trackmate thanks to Jean-Yves Tinevez https://sites.imagej.net/TrackMate-ExTrack/. The fiji version can profite from CPU parallelization better performances.

https://pypi.org/project/extrack/

# Dependencies

- numpy
- lmfit
- xmltodict
- matplotlib
- pandas

Optional: jupyter, cupy

# Installation (from pip)

(needs to be run in anaconda prompt for anaconda users on windows)

## Install dependencies

`pip install numpy lmfit xmltodict matplotlib pandas`

## Install ExTrack

`pip install extrack`

https://pypi.org/project/extrack/

the version 1.4 has an oudated version for the histograms script, replace it with histograms.py from the git repository if you want it to work correctly.

## Input file format

all_tracks: dict describing the tracks with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = x, y position.

# Tutorial

Test codes can be found at these location: 
- tests/test_extrack.py
- or tests/tutorial_extrack.ipynb
These contain the most important modules in a comprehensive framework. 
The .ipynb file is more didactic. One has to install jupyter to use it: `pip install jupyter`.

**Document here how to open a Jupyter notebook**

# Installation from this Gitlab repository

## From Unix/Mac:

`sudo apt install git` (if git is not installed)

`git clone https://github.com/FrancoisSimon/ExTrack-python3.git`

`cd ExTrack-python3`

`sudo python setup.py install`

## From Windows using anaconda prompt:

Need to install git if not already installed.

`git clone https://github.com/FrancoisSimon/ExTrack-python3.git` One can also just manually download the package if git is not install

`cd ExTrack-python3`

`python setup.py install` from the ExTrack-python3 directory


# Usage
## Main functions

extrack.tracking.get_2DSPT_params : performs the fit to infer the parameters of a given data set.

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

GPU parallelization used to be available but may not be compatible with the current CPU parallelization, GPU parallelization uses the package cupy which can be installed as described here : https://github.com/cupy/cupy. The cupy version will depend on your cuda version which itself must be compatible with your GPU driver and GPU. Usage of cupy requires a change in the module extrack/tracking (line 4) : GPU_computing = True

# Deploying (developer only)

# Authors
Francois Simon

# Bugs/suggestions
Send to bugtracker or to email.
