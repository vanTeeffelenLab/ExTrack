
import extrack
import numpy as np
from matplotlib import pyplot as plt

dt = 0.02

# simulate tracks able to come and leave from the field of view :

all_tracks, all_Bs = extrack.simulate_tracks.sim_FOV(nb_tracks=2000,
                                                     max_track_len=60,
                                                     LocErr=0.02,
                                                     Ds = np.array([0,0.25]),
                                                     initial_fractions = np.array([0.6,0.4]),
                                                     TrMat = np.array([[0.9,0.1],[0.1,0.9]]),
                                                     dt = dt,
                                                     pBL = 0.1,
                                                     cell_dims = [1,None,None], # dimension limits in x, y and z respectively
                                                     min_len = 5)

# fit parameters of the simulated tracks : 
# increase the number of workers accordingly to your number of avaialable core for faster computing, not working on windows
model_fit = extrack.tracking.param_fitting(all_tracks,
                                              dt,
                                              cell_dims = [1],
                                              nb_substeps = 2,
                                              nb_states = 2,
                                              frame_len = 7,
                                              verbose = 1,
                                              workers = 1,
                                              method = 'powell',
                                              steady_state = False,
                                              vary_params = {'LocErr' : True, 'D0' : True, 'D1' : True, 'F0' : True, 'p01' : True, 'p10' : True, 'pBL' : True},
                                              estimated_vals = {'LocErr' : 0.025, 'D0' : 0, 'D1' : 0.5, 'F0' : 0.5, 'p01' : 0.05, 'p10' : 0.05, 'pBL' : 0.05})

# produce histograms of time spent in each state :

extrack.visualization.visualize_states_durations(all_tracks,
                                                 model_fit.params,
                                                 dt,
                                                 cell_dims = [1],
                                                 nb_states = 2,
                                                 max_nb_states = 400,
                                                 long_tracks = True,
                                                 nb_steps_lim = 20,
                                                 steps = False)

# ground truth histogram (actual labeling from simulations) :

seg_len_hists = extrack.histograms.ground_truth_hist(all_Bs,long_tracks = True,nb_steps_lim = 20)

plt.plot(np.arange(1,len(seg_len_hists)+1)[:,None]*dt, seg_len_hists/np.sum(seg_len_hists,0), ':')

# assesment of the slops of the histograms :

np.polyfit(np.arange(1,len(seg_len_hists))[3:15], np.log(seg_len_hists[3:15])[:,0], 1)
np.polyfit(np.arange(1,len(seg_len_hists))[3:15], np.log(seg_len_hists[3:15])[:,1], 1)

# NB : the slops do not exactly correspond to the transition rates as leaving the field of view biases the dataset but the decay is still linear

# simulation of fewer tracks to plot them and their annotation infered by ExTrack :

all_tracks, all_Bs = extrack.simulate_tracks.sim_FOV(nb_tracks=500,
                                                     max_track_len=60,
                                                     LocErr=0.02,
                                                     Ds = np.array([0,0.5]),
                                                     initial_fractions = np.array([0.6,0.4]),
                                                     TrMat = np.array([[0.9,0.1],[0.1,0.9]]),
                                                     dt = dt,
                                                     pBL = 0.1,
                                                     cell_dims = [1,], # dimension limits in x, y and z respectively
                                                     min_len = 11)

# performs the states probability predictions based on the most likely parameters :

pred_Bs = extrack.tracking.predict_Bs(all_tracks,
                                     dt,
                                     model_fit.params,
                                     cell_dims=[1],
                                     nb_states=2,
                                     frame_len=12)

# turn outputs from extrack to a more classical data frame format :
    
DATA = extrack.exporters.extrack_2_pandas(all_tracks, pred_Bs, frames = None, opt_metrics = {})

# show all tracks :

extrack.visualization.visualize_tracks(DATA,
                                       track_length_range = [10,np.inf],
                                       figsize = (5,10))

# show the longest tracks in more details :

extrack.visualization.plot_tracks(DATA,
                                  max_track_length = 50, 
                                  nb_subplots = [5,5],
                                  figsize = (10,10), 
                                  lim = 1)

# download tracks from csv file :
path = '/home/oem/Downloads/tracks.csv'
all_tracks, frames, opt_metrics = extrack.readers.read_table(path,
                                                             opt_colnames=['QUALITY', 'RADIUS'])

# download tracks from xml file (trackmate format) :
path = '/home/oem/Downloads/tracks.csv'
all_tracks, frames, opt_metrics = extrack.readers.read_trackmate_xml(path,
                                                                     opt_metrics_names=[]) # opt_metrics_names corresponds to a list of the metrics the user wants to get from the detections lines in the xml file

# predict states :

pred_Bs = extrack.tracking.predict_Bs(all_tracks,
                                      dt,
                                      model_fit.params,
                                      cell_dims=[1],
                                      nb_states=2,
                                      frame_len=12)

# save as xml file used for trackmate :

save_path = './tracks.xml'
extrack.exporters.save_extrack_2_xml(all_tracks, pred_Bs, model_fit.params, save_path, dt, all_frames = None, opt_metrics = opt_metrics)

DATA = extrack.exporters.extrack_2_pandas(all_tracks, pred_Bs, frames = None, opt_metrics = opt_metrics)

# save as csv file :
save_path = './tracks.csv'
DATA.to_csv(save_path)


# simulate and fit a 3 states model
all_tracks, all_Bs = extrack.simulate_tracks.sim_FOV(nb_tracks=10000,
                                                     max_track_len=60,
                                                     min_track_len = 5,
                                                     LocErr=0.02,
                                                     Ds = np.array([0,0.04,0.2]),
                                                     initial_fractions = np.array([0.3,0.3,0.4]),
                                                     TrMat = np.array([[0.85,0.1,0.05],
                                                                       [0.1,0.8, 0.1],
                                                                       [0.1, 0.05, 0.8]]),
                                                     dt = dt,
                                                     pBL = 0.1,
                                                     cell_dims = [1,None,None]) # dimension limits in x, y and z respectively

# fit parameters of the simulated tracks :

model_fit = extrack.tracking.param_fitting(all_tracks,
                                              dt,
                                              cell_dims = [1],
                                              nb_substeps = 1,
                                              nb_states = 3,
                                              frame_len = 5,
                                              verbose = 0,
                                              workers = 1,
                                              method = 'powell',
                                              steady_state = False,
                                              vary_params = {'LocErr' : True, 'D0' : True, 'D1' :  True, 'D2' : True, 'F0' : True, 'F1' : True, 'p01' : True, 'p02' : True, 'p10' : True,'p12' :  True,'p20' :  True, 'p21' : True, 'pBL' : True},
                                              estimated_vals = {'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'F0' : 0.33,  'F1' : 0.33, 'p01' : 0.1, 'p02' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p20' :0.1, 'p21' :0.1, 'pBL' : 0.1},
                                              min_values = {'LocErr' : 0.007, 'D0' : 0, 'D1' : 0.0000001, 'D2' :  0.000001, 'F0' : 0.001,  'F1' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p20' :0.001, 'p21' :0.001, 'pBL' : 0.001},
                                              max_values = {'LocErr' : 0.6, 'D0' : 1e-20, 'D1' : 1, 'D2' :  10, 'F0' : 0.999,  'F1' : 0.999, 'p01' : 1, 'p02' : 1, 'p10' : 1, 'p12' : 1, 'p20' : 1, 'p21' : 1, 'pBL' : 0.99})
    






