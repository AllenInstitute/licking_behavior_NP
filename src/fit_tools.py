# Import packages
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import minimize
from scipy.interpolate import interp1d

try: 
    from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
    from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
except Exception:
    print("AllenSDK not importable, some things won't work")
    pass


## This code base expects some variables with the following formats
## get list of all lick times, an array with the lick times, rounded to the nearest 10msec
#licks = dataset.licks.time.values
#dt = 0.01 # 10msec timesteps
#licks = np.round(licks,2)
#licksdt = np.round(licks*(1/dt))
## get start/stop time for session
#start_time = 1
#stop_time = int(np.round(dataset.running_speed.time.values[-1],2)*(1/dt))
## A time vector in dt increments from 0 to stop_time
# time_vec = np.arange(0,stop_time/100.0,dt)

#### General Functions
def loglikelihood(licksdt, latent,params=[],l2=0):
    '''
    Compute the negative log likelihood of poisson observations, given a latent vector

    Args:
        licksdt: a vector of lick times in dt-index points
        latent: a vector of the estimated lick rate in each time bin
        params: a vector of the parameters for the model
        l2: amplitude of L2 regularization penalty
    
    Returns: NLL of the model
    '''
    # If there are any zeros in the latent model, have to add "machine tolerance"
    latent[latent==0] += np.finfo(float).eps

    NLL = -sum(np.log(latent)[licksdt.astype(int)]) + sum(latent) + l2*np.sum(np.array(params)**2)
    return NLL

def compare_model(latent, time_vec, licks, stop_time, running_speed=None,rewards=None, flashes=None, change_flashes=None,running_acceleration=None):
    '''
    Evaluate fit by plotting prediction and lick times

    Args:
        Latent: a vector of the estimate lick rate
        time_vec: the timestamp for each time bin
        licks: the time of each lick in dt-rounded timevalues
        stop_time: the number of timebins
    
    Plots the lick raster, and latent rate
    
    Returns: the figure handle and axis handle
    '''
    fig,axes  = plt.subplots(2,1)  
    fig.set_size_inches(12,8) 
    if running_speed is not None:
        axes[0].plot(time_vec, running_speed / np.max(running_speed), 'r-',alpha = .2, label='running_speed') 
    if running_acceleration is not None:
        axes[0].plot(time_vec, 0.5+ (running_acceleration / (2*np.max(running_acceleration))), 'y-',alpha = .2, label='running_acceleration')
    if flashes is not None:
        axes[0].vlines(flashes, 0, 1, alpha = .2, color='g', label='flash',linewidth=4)
        axes[1].vlines(flashes, 0, 1, alpha = .2, color='g', label='flash',linewidth=4)
    if change_flashes is not None:
        axes[0].vlines(change_flashes, 0, 1, alpha = .6, color='c', label='change flash',linewidth=4)
        axes[1].vlines(change_flashes, 0, 1, alpha = .6, color='c', label='change flash',linewidth=4)
    axes[0].plot(time_vec,latent,'b',label='model')
    axes[1].plot(time_vec,latent,'b',label='model')
    axes[0].vlines(licks,.8, .9, alpha = 1, label='licks',linewidth=2)
    axes[1].vlines(licks,.8, .9, alpha = 1, label='licks',linewidth=2)
    if rewards is not None:
        axes[0].plot(rewards, np.zeros(np.shape(rewards))+0.05, 'ro', label='reward',markersize=10)
        axes[1].plot(rewards, np.zeros(np.shape(rewards))+0.005, 'ro', label='reward',markersize=10)
    axes[0].set_ylim([0, 1])
    axes[0].set_xlim(600,610)
    axes[0].legend(loc='upper left' )
    axes[0].set_xlabel('time (s)',fontsize=16)
    axes[0].set_ylabel('Licking Probability',fontsize=16)
    axes[1].set_ylim([0, .05])
    axes[1].set_xlim(600,610)
    axes[1].set_xlabel('time (s)',fontsize=16)
    axes[1].set_ylabel('Licking Probability',fontsize=16)
    axes[0].yaxis.set_tick_params(labelsize=16) 
    axes[0].xaxis.set_tick_params(labelsize=16)
    axes[1].yaxis.set_tick_params(labelsize=16) 
    axes[1].xaxis.set_tick_params(labelsize=16)
    plt.tight_layout()
    
    def on_key_press(event):
        xStep = 2
        x = axes[0].get_xlim()
        xmin = x[0]
        xmax = x[1]
        if event.key=='<' or event.key==',' or event.key=='left': 
            xmin -= xStep
            xmax -= xStep
        elif event.key=='>' or event.key=='.' or event.key=='right':
            xmin += xStep
            xmax += xStep
        axes[0].set_xlim(xmin,xmax)
        axes[1].set_xlim(xmin,xmax)
    kpid = fig.canvas.mpl_connect('key_press_event', on_key_press)
    return fig, axes

def compute_bic(nll, num_params, num_data_points):
    '''
    Computes the BIC of the model
    BIC = log(#num-data-points)*#num-params - 2*log(L)
        = log(x)*k + 2*NLL

    Args:
        nll: negative log likelihood of the model
        num_params: number of parameters in the model
        num_data_points: number of data points in model
    
    Returns the BIC score
    '''
    return np.log(num_data_points)*num_params + 2*nll

def evaluate_model(res,model_func, licksdt, stop_time):
    '''
    Evaluates the model

    Args:
        res: the optimization results from minimize()
        model_func: the function handle for the model
        licksdt: the lick times in dt-index
        stop_time: number of time bins

    Returns: res, with nll computed, latent estimate computed, BIC computed
    '''
    res.nll, res.latent = model_func(res.x)
    res.BIC = compute_bic(res.nll, len(res.x), len(res.latent))
    return res    

def build_filter(params,filter_time_vec, sigma, plot_filters=False, plot_nonlinear=False):
    '''
    Builds a filter out of basis functions

    puts len(params) gaussian bumps equally spaced across time_vec
    each gaussian is weighted by params, and is truncated outside of time_vec

    Args:
        params: The weights of each gaussian bumps
        filter_time_vec: the time vector of the timepoints to build the filter for
        sigma: the variance of each gaussian bump
        plot_filters: if True, plots each bump, and the entire function
    
    
    Returns: The filter, with length given by filter_time_vec
    
    Example:

    filter_time_vec = np.arange(dt,.21,dt)
    build_filter([2.75,-2,-2,-2,-2,3,3,3,.1], filter_time_vec, 0.025, plot_filters=True)
    '''
    def gaussian_template(mu,sigma):
        return (1/(np.sqrt(2*3.14*sigma**2)))*np.exp(-(filter_time_vec-mu)**2/(2*sigma**2))
    numparams = len(params)
    mean = (filter_time_vec[-1] - filter_time_vec[0])/(numparams-1)
    base = np.zeros(np.shape(filter_time_vec)) 
    if plot_filters:
        plt.figure()
    for i in range(0,len(params)):
        base += params[i]*gaussian_template(mean*i,sigma)    
        if plot_filters:
            plt.plot(filter_time_vec, params[i]*gaussian_template(mean*i,sigma))
    if plot_filters:
        plt.plot(filter_time_vec,base, 'k')
        if plot_nonlinear:
            plt.figure()
            plt.plot(filter_time_vec, np.exp(base), 'k')
    return base

def get_data(experiment_id, save_dir=r'/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/thursday_harbor'):

    '''
    Pull processed data.

    Args: 
        experiment_id (int): The experiment ID to get data for
        save_dir (str): dir containing processed NPZ files

    Returns: 
        data (dict of np.arr): A dictionary with arrays under the following keys: 

        running_timestamps = data['running_timestamps']
        running_speed = data['running_speed']
        lick_timestamps = data['lick_timestamps']
        stim_on_timestamps = data['stim_on_timestamps']
        stim_off_timestamps = data['stim_off_timestamps']
        stim_name = data['stim_name']
        stim_omitted = data['stim_omitted']
        reward_timestamps = data['reward_timestamps']

    '''
    output_fn = 'experiment_{}.npz'.format(experiment_id)
    full_path = os.path.join(save_dir, output_fn)
    data = np.load(full_path)
    return data

def get_sdk_data(experiment_id, load_dir=r'\\allen\aibs\technology\nicholasc\behavior_ophys'):
    
    """Uses AllenSDK to load data and return session object
    
    Arguments:
        experiment_id {int} -- [9 digit unique identifier for a behavior ophys session]
    
    Keyword Arguments:
        load_dir {file path} -- [path of saved NWB] (default: {r'\allen\aibs\technology\nicholasc\behavior_ophys'})
    
    Returns:
        data[dictionary] -- [session object]
    """
   

    full_filepath = os.path.join(load_dir, 'behavior_ophys_session_{}.nwb'.format(experiment_id))
    
    session = BehaviorOphysSession(api=BehaviorOphysNwbApi(full_filepath))
    running_timestamps = session.running_speed.timestamps
    running_speed = session.running_speed.values
 
    #lick information
    lick_timestamps = session.licks.values
    lick_timestamps = lick_timestamps[lick_timestamps>min(running_timestamps)]
     
    #rewards and water consumption
    reward_timestamps = session.rewards.index.values
    reward_volume = session.rewards.volume.values
    reward_autoreward = session.rewards.autorewarded.values
    reward_timestamps = reward_timestamps[reward_timestamps>min(running_timestamps)]

    #stimulus related 
    stim_flash_image = session.stimulus_presentations[["image_name"]].values
    stim_flash_start = session.stimulus_presentations[["start_time"]].values
    stim_flash_stop = session.stimulus_presentations[["stop_time"]].values
    stim_flash_image = stim_flash_image[stim_flash_start > min(running_timestamps)]
    stim_flash_stop = stim_flash_stop[stim_flash_start > min(running_timestamps)]
    stim_flash_start = stim_flash_start[stim_flash_start > min(running_timestamps)]

        #get true changes and exclude aborted & autrewarded trials
    changes_df = session.trials.loc[session.trials["stimulus_change"]==True, ["change_image_name", "change_time"]].copy()
    stim_change_image = changes_df.change_image_name.values
    stim_change_time = changes_df.change_time.values
    stim_change_image = stim_change_image[stim_change_time > min(running_timestamps)]
    stim_change_time = stim_change_time[stim_change_time > min(running_timestamps)]

    #alignment
    start_time = running_timestamps[0]   
    running_timestamps = running_timestamps-start_time
    lick_timestamps = lick_timestamps-start_time
    reward_timestamps = reward_timestamps - start_time
    stim_flash_start = stim_flash_start - start_time
    stim_change_time = stim_change_time - start_time
    stim_flash_stop = stim_flash_stop - start_time

    # Interpolate
    dt = 0.01
    timebase_interpolation = np.arange(0, max(running_timestamps),dt)
    f_running= interp1d(running_timestamps, running_speed)
    running_speed_interpolated = f_running(timebase_interpolation)
    running_speed = running_speed_interpolated 
  
    data={"running_timestamps": running_timestamps, "running_speed":running_speed, "lick_timestamps": lick_timestamps, 
      "reward_timestamps":reward_timestamps, 'reward_volume': reward_volume, "reward_autoreward":reward_autoreward,
      "stim_flash_image":stim_flash_image, "stim_flash_start": stim_flash_start,
        "stim_flash_stop": stim_flash_stop, "stim_change_image": stim_change_image, "stim_change_time": stim_change_time}

    return data

#### Specific Model Functions
# set up basic model, which has a constant lick rate
# mean_lick rate: scalar parameter that is the log(average-lick rate)
# licksdt: a vector of lick times in dt-index points
# stop_time: The index of the last time-bin
#
# Returns: the NLL of the model, and the latent rate
def mean_lick_model(mean_lick_rate,licksdt, stop_time):
    '''
    Depreciated, use licking model
    '''
    base = np.ones((stop_time,))*mean_lick_rate
    latent = np.exp(base)
    return loglikelihood(licksdt,latent), latent

# Wrapper function for optimization that only takes one input
def mean_wrapper_func(mean_lick_rate):
    '''
    Depreciated, use licking model
    '''
    return mean_lick_model(mean_lick_rate,licksdt,stop_time)[0]

# Model with Mean lick rate, and post-lick filter
# params[0]: mean lick rate
# params[1:]: post-lick filter
def mean_post_lick_model(params, licksdt,stop_time):
    '''
    Depreciated, use licking model
    '''
    mean_lick_rate = params[0]
    base = np.ones((stop_time,))*mean_lick_rate
    post_lick_filter = params[1:]
    post_lick = np.zeros((stop_time+len(post_lick_filter)+1,))
    for i in licksdt:
        post_lick[int(i)+1:int(i)+1+len(post_lick_filter)] +=post_lick_filter
    post_lick = post_lick[0:stop_time]
    latent = np.exp(base+post_lick)
    return loglikelihood(licksdt,latent), latent

def post_lick_wrapper_func(params):
    '''
    Depreciated, use licking model
    '''
    return mean_post_lick_model(params,licksdt,stop_time)[0]

# Model with Mean lick rate, and post-lick filter
# params[0]: mean lick rate
# params[1:]: post-lick filter parameters for basis function
def basis_post_lick_model(params, licksdt,stop_time,sigma):
    '''
    Depreciated, use licking model
    '''
    mean_lick_rate = params[0]
    base = np.ones((stop_time,))*mean_lick_rate
    filter_time_vec = np.arange(dt,.21,dt)
    post_lick_filter = build_filter(params[1:],filter_time_vec,sigma)
    post_lick = np.zeros((stop_time+len(post_lick_filter)+1,))
    for i in licksdt:
        post_lick[int(i)+1:int(i)+1+len(post_lick_filter)] +=post_lick_filter
    post_lick = post_lick[0:stop_time]
    latent = np.exp(base+post_lick)
    return loglikelihood(licksdt,latent), latent

def basis_post_lick_wrapper_func(params):
    '''
    Depreciated, use licking model
    '''
    return basis_post_lick_model(params,licksdt,stop_time,0.025)[0]


def licking_model(params, licksdt, stop_time, mean_lick_rate=True, dt = 0.01,
    post_lick=True,num_post_lick_params=10,post_lick_duration=.21, post_lick_sigma =0.025, 
    include_running_speed=False, num_running_speed_params=6,running_speed_duration = 0.25, running_speed_sigma = 0.025,running_speed=0,
    include_reward=False, num_reward_params=20,reward_duration =4, reward_sigma = 0.25 ,rewardsdt=[],
    include_flashes=False, num_flash_params=15,flash_duration=0.76, flash_sigma = 0.05, flashesdt=[],
    include_change_flashes=False, num_change_flash_params=30,change_flash_duration=1.6, change_flash_sigma = 0.05, change_flashesdt=[],
    include_running_acceleration=False, num_running_acceleration_params=10, running_acceleration_duration=1.01, running_acceleration_sigma = 0.2, running_acceleration=[],
    l2=0):
    '''
    Top function for fitting licking model. Can flexibly add new features
    
    Args:
        params,         vector of parameters
        licksdt,        dt-index of each lick time
        stop_time,      number of timebins
        mean_lick_rate, if True, include mean lick rate
        dt,             length of timestep
        For each feature:
        <feature>               if True, include this features
        num_<feature>_params    number of parameters for this feature
        <feature>_duration      length of the filter for this feature
        <feature>_sigma         width of each basis function for this feature

        l2,             penalty strength of L2 (Ridge) Regularization
    Returns:
        NLL for this model
        latent lick rate for this model
    '''
    base = np.zeros((stop_time,))
    param_counter = 0
    if mean_lick_rate:
        mean_lick_param = params[param_counter]
        param_counter +=1
        base += np.ones((stop_time,))*mean_lick_param
    if post_lick:
        param_counter, post_lick_params = extract_params(params, param_counter, num_post_lick_params)
        post_lick_response = linear_post_lick(post_lick_params,post_lick_duration,licksdt,dt,post_lick_sigma,stop_time)
        base += post_lick_response
    if include_running_speed:
        param_counter, running_speed_params = extract_params(params, param_counter, num_running_speed_params)
        running_speed_response = linear_running_speed(running_speed_params, running_speed_duration, running_speed, dt, running_speed_sigma, stop_time)
        base += running_speed_response
    if include_reward:
        param_counter, reward_params = extract_params(params, param_counter, num_reward_params)
        reward_response = linear_reward(reward_params, reward_duration, rewardsdt, dt, reward_sigma, stop_time)
        base += reward_response
    if include_flashes:
        param_counter, flash_params = extract_params(params, param_counter, num_flash_params)
        flash_response = linear_flash(flash_params, flash_duration, flashesdt, dt, flash_sigma, stop_time)
        base += flash_response
    if include_change_flashes:
        param_counter, change_flash_params = extract_params(params, param_counter, num_change_flash_params)
        change_flash_response = linear_change_flash(change_flash_params, change_flash_duration, change_flashesdt, dt, change_flash_sigma, stop_time)
        base += change_flash_response
    if include_running_acceleration:
        param_counter, running_acceleration_params = extract_params(params,param_counter, num_running_acceleration_params)
        running_acceleration_response = linear_running_acceleration(running_acceleration_params, running_acceleration_duration, running_acceleration, dt, running_acceleration_sigma, stop_time)
        base += running_acceleration_response
    if not (param_counter == len(params)):
        print(str(param_counter))
        print(str(len(params)))
        raise Exception('Not all parameters were used')

    # Clip to prevent overflow errors
    latent = np.exp(np.clip(base, -700, 700))
    return loglikelihood(licksdt,latent,params=params, l2=l2), latent

def extract_params(params, param_counter, num_to_extract):
    '''
    Extracts each feature's parameters from the vector of model parameters

    Args:
        params      the vector of all parameters
        param_counter, the current location in the parameter list
        num_to_extract, the number of parameters for this feature
    '''
    this_params = params[param_counter:param_counter+num_to_extract]
    param_counter += num_to_extract
    if not (len(this_params) == num_to_extract):
        raise Exception('Parameter mis-alignment')
    return param_counter, this_params

def linear_post_lick(post_lick_params, post_lick_duration, licksdt,dt,post_lick_sigma,stop_time):
    '''
    Computes the linear response function for the post-lick-triggered filter

    Args:
        post_lick_params,       vector of parameters, number of parameters determines number of basis functions
        post_lick_duration,     duration (s) of the filter
        licksdt,                times of the licks in dt-index units
        post_lick_sigma,        sigma parameter for basis functions
        stop_time,              number of timebins
    '''
    filter_time_vec = np.arange(dt,post_lick_duration,dt)
    post_lick_filter = build_filter(post_lick_params,filter_time_vec,post_lick_sigma)
    post_lick = np.zeros((stop_time+len(post_lick_filter)+1,))
    for i in licksdt:
        post_lick[int(i)+1:int(i)+1+len(post_lick_filter)] +=post_lick_filter
    post_lick = post_lick[0:stop_time]       
    return post_lick

def linear_running_speed(running_speed_params, running_speed_duration, running_speed, dt, running_speed_sigma, stop_time):
    '''
    Args:
        running_speed_params (np.array): Array of parameters
        running_speed_duration (int): Length of the running speed filter in seconds
        running_speed (np.array): Actual running speed values
        dt (float): length of the time bin in seconds
        running_speed_sigma (float): standard deviation of each Gaussian basis function to use in the filter
        stop_time (int): end bin number

    Returns:
        running_effect (np.array): The effect on licking from the previous running at each time point
    '''

    #  filter_time_vec = np.arange(dt, running_speed_duration, dt)
    #  running_speed_filter = build_filter(running_speed_params, filter_time_vec, running_speed_sigma)
    running_speed_filter = running_speed_params
    #  running_effect = np.convolve(np.concatenate([np.zeros(len(running_speed_filter)), running_speed]), running_speed_filter)[:stop_time]
    running_effect = np.convolve(running_speed, running_speed_filter)[:stop_time]
    
    # Shift our predictions to the next time bin
    running_effect = np.r_[0, running_effect][:-1]
    return running_effect

def linear_running_acceleration(running_acceleration_params, running_acceleration_duration, running_acceleration, dt, running_acceleration_sigma, stop_time):
    '''
    Args:
        running_acceleration_params (np.array): Array of parameters
        running_acceleration_duration (int): Length of the running acceleration filter in seconds
        running_acceleration (np.array): Actual running acceleration values
        dt (float): length of the time bin in seconds
        running_acceleration_sigma (float): standard deviation of each Gaussian basis function to use in the filter
        stop_time (int): end bin number

    Returns:
        running_effect (np.array): The effect on licking from the previous running at each time point
    '''

    filter_time_vec = np.arange(dt, running_acceleration_duration, dt)
    running_acceleration_filter = build_filter(running_acceleration_params, filter_time_vec, running_acceleration_sigma)
    #running_acceleration_filter = running_acceleration_params
    running_effect = np.convolve(np.concatenate([np.zeros(len(running_acceleration_filter)), running_acceleration]), running_acceleration_filter)[:stop_time]
    
    # Shift our predictions to the next time bin
    running_effect = np.r_[0, running_effect[1:]]
    return running_effect




def linear_reward(reward_params, reward_duration, rewardsdt, dt, reward_sigma, stop_time):
    '''
    Computes the linear response function for the reward-triggered filter

    Args:
        reward_params,    vector of parameters, number of parameters determines number of basis functions
        reward_duration,  duration (s) of the filter
        rewardsdt,       times of the rewards in dt-index units
        reward_sigma,     sigma parameter for basis functions
        stop_time,              number of timebins
    '''
    filter_time_vec =np.arange(dt, reward_duration,dt)
    reward_filter = build_filter(reward_params, filter_time_vec, reward_sigma)
    base = np.zeros((stop_time+len(reward_filter)+1,))
    for i in rewardsdt:
        base[int(i)+1:int(i)+1+len(reward_filter)] += reward_filter
    base = base[0:stop_time]
    return base

def linear_flash(flash_params, flash_duration, flashesdt, dt, flash_sigma, stop_time):
    '''
    Computes the linear response function for the image-triggered filter

    Args:
        flash_params,    vector of parameters, number of parameters determines number of basis functions
        flash_duration,  duration (s) of the filter
        flashesdt,       times of the flashes in dt-index units
        flash_sigma,     sigma parameter for basis functions
        stop_time,              number of timebins
    '''
    filter_time_vec =np.arange(dt, flash_duration,dt)
    flash_filter = build_filter(flash_params, filter_time_vec, flash_sigma)
    base = np.zeros((stop_time+len(flash_filter)+1,))
    for i in flashesdt:
        base[int(i)+1:int(i)+1+len(flash_filter)] += flash_filter
    base = base[0:stop_time]
    return base

def linear_change_flash(change_flash_params, change_flash_duration, change_flashesdt, dt, change_flash_sigma, stop_time):
    '''
    Computes the linear response function for the change-image-triggered filter

    Args:
        change_flash_params,    vector of parameters, number of parameters determines number of basis functions
        change_flash_duration,  duration (s) of the filter
        change_flashesdt,       times of the change flashes in dt-index units
        change_flash_sigma,     sigma parameter for basis functions
        stop_time,              number of timebins
    '''
    filter_time_vec =np.arange(dt, change_flash_duration,dt)
    change_flash_filter = build_filter(change_flash_params, filter_time_vec, change_flash_sigma)
    base = np.zeros((stop_time+len(change_flash_filter)+1,))
    for i in change_flashesdt:
        base[int(i)+1:int(i)+1+len(change_flash_filter)] += change_flash_filter
    base = base[0:stop_time]
    return base

class Model(object):

    '''
    Object for defining, training, and analyzing a GLM. 

    What does it need to do? 

    1. User has to define all the filters
    2. Pass in the data (training set vs. test set?)
    2. Obj can create good initial params
    3. Method to start training
    4. Method for plotting predictions
    5. Method for plotting the filters

    '''

    def __init__(self,
                 licks, 
                 running_timestamps,
                 running_speed=None, 
                 rewards=None, 
                 flashes=None, 
                 change_flashes=None,
                 dt=0.01,
                 start_time=1,
                 mean_lick_rate=True, 
                 post_lick=False,
                 num_post_lick_params=10,
                 post_lick_duration=.21,
                 post_lick_sigma =0.025,
                 include_running_speed=False,
                 num_running_speed_params=6,
                 running_speed_duration=0.25,
                 running_speed_sigma=0.025,
                 include_reward=False,
                 num_reward_params=20,
                 reward_duration =4,
                 reward_sigma=0.25,
                 include_flashes=False,
                 num_flash_params=15,
                 flash_duration=0.76,
                 flash_sigma=0.05,
                 include_change_flashes=False,
                 num_change_flash_params=15,
                 change_flash_duration=0.76,
                 change_flash_sigma=0.05,
                 include_running_acceleration=False, 
                 num_running_acceleration_params=10, 
                 running_acceleration_duration=1.01, 
                 running_acceleration_sigma = 0.2, 
                 running_acceleration=[],
                 l2=0,
                 initial_params=None):

        self.dt = dt
        self.start_time = start_time

        self.stop_time = int(np.round(running_timestamps[-1],2)*(1/dt))
        licks = licks[licks < self.stop_time/100]
        self.licks = np.round(licks,2)
        self.licksdt = np.round(licks*(1/dt))

        self.running_timestamps = running_timestamps
        self.running_speed = np.round(running_speed, 2)
        self.rewards = np.round(rewards, 2)
        self.flashes = np.round(flashes, 2)
        self.change_flashes = np.round(change_flashes, 2)

        if rewards is not None:
            self.rewardsdt = np.round(rewards*(1/dt))
        else:
            self.rewardsdt = None
        if flashes is not None:
            self.flashesdt = np.round(flashes*(1/dt))
        else:
            self.flashesdt = None
        if change_flashes is not None:
            self.change_flashesdt = np.round(change_flashes*(1/dt))
        else:
            self.change_flashesdt = None
        self.time_vec = np.arange(0,self.stop_time/100.0,dt)

        self.mean_lick_rate = mean_lick_rate
        self.post_lick = post_lick
        self.num_post_lick_params = num_post_lick_params
        self.post_lick_duration = post_lick_duration
        self.post_lick_sigma = post_lick_sigma
        self.include_running_speed = include_running_speed
        self.num_running_speed_params = num_running_speed_params
        self.running_speed_duration = running_speed_duration
        self.running_speed_sigma = running_speed_sigma
        self.include_reward = include_reward
        self.num_reward_params = num_reward_params
        self.reward_duration = reward_duration
        self.reward_sigma = reward_sigma
        self.include_flashes = include_flashes
        self.num_flash_params = num_flash_params
        self.flash_duration = flash_duration
        self.flash_sigma = flash_sigma
        self.include_change_flashes = include_change_flashes
        self.num_change_flash_params = num_change_flash_params
        self.change_flash_duration = change_flash_duration
        self.change_flash_sigma = change_flash_sigma
        self.include_running_acceleration = include_running_acceleration
        self.num_running_acceleration_params = num_running_acceleration_params
        self.running_acceleration_duration = running_acceleration_duration
        self.running_acceleration_sigma = running_acceleration_sigma
        self.running_acceleration = running_acceleration
        self.l2 = l2

        paramlist = self.make_param_list()

        # Setup initial params
        if initial_params is None:

            paramlist = []
            if self.mean_lick_rate:
                paramlist.append([-0.5])
            if self.post_lick:
                paramlist.append(np.zeros(self.num_post_lick_params))
            if self.include_running_speed:
                paramlist.append(np.zeros(self.num_running_speed_params))
            if self.include_reward:
                paramlist.append(np.zeros(self.num_reward_params))
            if self.include_flashes:
                paramlist.append(np.zeros(self.num_flash_params))
            if self.include_change_flashes:
                paramlist.append(np.zeros(self.num_change_flash_params))
            if self.include_running_acceleration:
                paramlist.append(np.zeros(self.num_running_acceleration_params))

            self.initial_params = np.concatenate(paramlist)
        else:
            self.initial_params = initial_params

    def wrapper_full(self, params):
        return licking_model(params,
                             licksdt=self.licksdt,
                             stop_time=self.stop_time,
                             mean_lick_rate=self.mean_lick_rate,
                             dt=self.dt,
                             post_lick=self.post_lick,
                             num_post_lick_params=self.num_post_lick_params,
                             post_lick_duration=self.post_lick_duration,
                             post_lick_sigma=self.post_lick_sigma,
                             include_running_speed=self.include_running_speed,
                             num_running_speed_params=self.num_running_speed_params,
                             running_speed_duration=self.running_speed_duration,
                             running_speed_sigma=self.running_speed_sigma,
                             running_speed=self.running_speed,
                             include_reward=self.include_reward,
                             num_reward_params=self.num_reward_params,
                             reward_duration=self.reward_duration,
                             reward_sigma=self.reward_sigma,
                             rewardsdt=self.rewardsdt,
                             include_flashes=self.include_flashes,
                             num_flash_params=self.num_flash_params,
                             flash_duration=self.flash_duration,
                             flash_sigma=self.flash_sigma,
                             flashesdt=self.flashesdt,
                             include_change_flashes=self.include_change_flashes,
                             num_change_flash_params=self.num_change_flash_params,
                             change_flash_duration=self.change_flash_duration,
                             change_flash_sigma=self.change_flash_sigma,
                             change_flashesdt=self.change_flashesdt,
                             include_running_acceleration= self.include_running_acceleration, 
                             num_running_acceleration_params = self.num_running_acceleration_params, 
                             running_acceleration_duration=self.running_acceleration_duration, 
                             running_acceleration_sigma = self.running_acceleration_sigma, 
                             running_acceleration=self.running_acceleration,
                             l2=self.l2)

    def fit(self):

        print("Fitting model with {} params".format(len(self.initial_params)))

        def wrapper_func(params):
            return self.wrapper_full(params)[0]

        start_time = time.time()
        # TODO: Make this async?
        res = minimize(wrapper_func, self.initial_params)
        self.res = evaluate_model(res, self.wrapper_full,
                                  self.licksdt, self.stop_time)
        elapsed_time = time.time() - start_time
        print("Done! Elapsed time: {:02f} sec".format(time.time()-start_time))

    def compare(self):
        compare_model(self.res.latent,
                      self.time_vec,
                      self.licks,
                      self.stop_time,
                      self.running_speed)

    def plot_filter(self, filter_to_plot):

        if filter_to_plot not in self.model_filters:
            print("Model doesn't have that filter")
            return None

        # Plot the filter
        filter_to_plot_ind = self.model_filters.index(filter_to_plot)
        filter_start = self.filter_params_start[filter_to_plot_ind]
        filter_end = self.filter_params_end[filter_to_plot_ind]
        filter_x = self.res.x[filter_start:filter_end]
        filter_duration = self.filter_durations[filter_to_plot_ind]
        filter_sigma = self.filter_sigmas[filter_to_plot_ind]

        #These filters don't use basis functions
        if filter_to_plot in ['mean_lick', 'running_speed']:
            ax = plt.subplot(111)
            ax.plot(filter_x, 'k-') #TODO: This is just the linear filter for now.
            return [ax] #TODO: We can return the handles to both figs here
        else:
            build_filter(filter_x,
                         np.arange(self.dt, filter_duration, self.dt),
                         filter_sigma,
                         plot_filters=True,
                         plot_nonlinear=True)
            ax = plt.gca()
            return [ax]

    def linear_filter(self, filter_name):
        '''
        Return the linear representation of the model filter
        '''

        if filter_name not in self.model_filters:
            print("Model doesn't have that filter")
            return None

        # Plot the filter
        filter_name_ind = self.model_filters.index(filter_name)
        filter_start = self.filter_params_start[filter_name_ind]
        filter_end = self.filter_params_end[filter_name_ind]
        filter_x = self.res.x[filter_start:filter_end]
        filter_duration = self.filter_durations[filter_name_ind]
        filter_sigma = self.filter_sigmas[filter_name_ind]

        #These filters don't use basis functions
        if filter_name in ['mean_lick_rate', 'running_speed']:
            return filter_x
        else:
            base = build_filter(filter_x,
                         np.arange(self.dt, filter_duration, self.dt),
                         filter_sigma,
                         plot_filters=False,
                         plot_nonlinear=False)
            return base

    def plot_all_filters(self, nonlinear=True):
        plt.clf()
        nFilters = len(self.model_filters)
        time_vec = np.arange(self.dt, 10, self.dt)
        for indFilter, filter_name in enumerate(self.model_filters):
            base = self.linear_filter(filter_name)
            if nonlinear:
                base = np.exp(np.clip(base, -700, 700))
                
            # Plot the filter
            plt.subplot(4, 2, indFilter+1)
            plt.title(filter_name)
            if len(base) == 1:
                plt.plot(base,'o',color='0.5')
            else:
                my_time_vec = time_vec[0:len(base)] 
                plt.plot(my_time_vec, base, color='0.5')
                if nonlinear:
                    plt.plot(my_time_vec, np.ones(np.shape(base)),'k--',alpha=0.3)
                    plt.ylim(ymin=0)
        plt.tight_layout()
        plt.show()

    def make_param_list(self):
        '''
        Saves information about the filter parameters as attrs, and returns
        a list that can be used to set default params if you want.
        '''
        model_filters = []
        filter_params_start = []
        filter_params_end = []
        filter_durations = []
        filter_sigmas = []
        paramlist = []

        ind_param = 0
        if self.mean_lick_rate:
            paramlist.append([-0.5])
            model_filters.append('mean_lick_rate')
            filter_params_start.append(ind_param)
            end_param_ind = ind_param + 1
            filter_params_end.append(end_param_ind)
            ind_param = end_param_ind
            filter_durations.append(None)
            filter_sigmas.append(None)
        if self.post_lick:
            paramlist.append(np.zeros(self.num_post_lick_params))
            model_filters.append('post_lick')
            filter_params_start.append(ind_param)
            end_param_ind = ind_param + self.num_post_lick_params
            filter_params_end.append(end_param_ind)
            ind_param = end_param_ind
            filter_durations.append(self.post_lick_duration)
            filter_sigmas.append(self.post_lick_sigma)
        if self.include_running_speed:
            paramlist.append(np.zeros(self.num_running_speed_params))
            model_filters.append('running_speed')
            filter_params_start.append(ind_param)
            end_param_ind = ind_param + self.num_running_speed_params
            filter_params_end.append(end_param_ind)
            ind_param = end_param_ind
            filter_durations.append(self.running_speed_duration)
            filter_sigmas.append(self.running_speed_sigma)
        if self.include_reward:
            paramlist.append(np.zeros(self.num_reward_params))
            model_filters.append('reward')
            filter_params_start.append(ind_param)
            end_param_ind = ind_param + self.num_reward_params
            filter_params_end.append(end_param_ind)
            ind_param = end_param_ind
            filter_durations.append(self.reward_duration)
            filter_sigmas.append(self.reward_sigma)
        if self.include_flashes:
            paramlist.append(np.zeros(self.num_flash_params))
            model_filters.append('flash')
            filter_params_start.append(ind_param)
            end_param_ind = ind_param + self.num_flash_params
            filter_params_end.append(end_param_ind)
            ind_param = end_param_ind
            filter_durations.append(self.flash_duration)
            filter_sigmas.append(self.flash_sigma)
        if self.include_change_flashes:
            paramlist.append(np.zeros(self.num_change_flash_params))
            model_filters.append('change_flash')
            filter_params_start.append(ind_param)
            end_param_ind = ind_param + self.num_change_flash_params
            filter_params_end.append(end_param_ind)
            ind_param = end_param_ind
            filter_durations.append(self.change_flash_duration)
            filter_sigmas.append(self.change_flash_sigma)
        if self.include_running_acceleration:
            model_filters.append('running_acceleration')
            filter_params_start.append(ind_param)
            end_param_ind = ind_param + self.num_running_acceleration_params
            filter_params_end.append(end_param_ind)
            ind_param = end_param_ind
            filter_durations.append(self.running_acceleration_duration)
            filter_sigmas.append(self.running_acceleration_sigma)

        # Save these lists of filter info for later plotting
        self.model_filters = model_filters
        self.filter_params_start = filter_params_start
        self.filter_params_end = filter_params_end
        self.filter_durations = filter_durations
        self.filter_sigmas = filter_sigmas
        return paramlist

    def initial_params_from_file_res(self, Fn):
        '''
        Get the initial params from a previous run. 
        Importantly, this has to accomodate different parameter sets for the runs.
        Only set initial params for the filters that we had in the last run.

        TODO: What happens if we change param number for a filter between runs and want to 
        use this for the other filters? 
        '''

        # To decrease the effect of changing the class def, just rebuild the thing
        inst_previous = Model.from_file_rebuild(Fn)
        self.initial_params = inst_previous.res.x

        #  # TODO: Make it work for diff params
        #  for filter_name in self.model_filters:
        #  
        #      #See if the filter existed in the previous model
        #      if filter_name in inst_previous.model_filters:
        #          pass
        #      else:
        #          # Don't mess with the initial params for this filter, 
        #          # which are set at init time either by passing for calc.
        #          # zeros
        #          pass


    def save(self, Fn):
        '''
        Fn: output pickle file path
        '''
        with open(Fn, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def from_file_direct(cls, Fn):
        '''
        Construct object instance from saved pickle file
        Directly updates the instance __dict__ w/o calling init.

        Fn: Pickle file path on disk
        '''
        inst = cls.__new__(cls)
        with open(Fn, 'rb') as f:
            inst.__dict__.update(pickle.load(f))
        return inst

    @classmethod
    def from_file_rebuild(cls, Fn):

        init_args = [
            'licks', 'running_timestamps', 'running_speed', 'rewards', 'flashes', 
            'change_flashes', 'dt', 'start_time', 'mean_lick_rate', 'post_lick',
            'num_post_lick_params', 'post_lick_duration', 'post_lick_sigma',
            'include_running_speed', 'num_running_speed_params',
            'running_speed_duration', 'running_speed_sigma', 'include_reward',
            'num_reward_params', 'reward_duration', 'reward_sigma', 'include_flashes',
            'num_flash_params', 'flash_duration', 'flash_sigma', 'include_change_flashes',
            'num_change_flash_params', 'change_flash_duration', 'change_flash_sigma',
            'l2', 'initial_params'
        ]
        with open(Fn, 'rb') as f:
            argdict = pickle.load(f)
        initdict = {key:argdict[key] for key in init_args}
        inst = cls(**initdict)
        inst.__dict__.update(argdict)
        return inst


def extract_data(data,dt):
    licks = data['lick_timestamps']
    running_timestamps = data['running_timestamps']
    running_speed = data['running_speed']
    rewards = np.round(data['reward_timestamps'],2)
    flashes=np.round(data['stim_on_timestamps'],2)
    rewardsdt = np.round(rewards*(1/dt))
    flashesdt = np.round(flashes*(1/dt))
    
    stims = data['stim_id']
    stims[np.array(stims) == 8 ] = 100
    diffs = np.diff(stims)
    diffs[(diffs > 50) | (diffs < -50 )] = 0
    diffs[ np.abs(diffs) > 0] = 1
    diffs = np.concatenate([[0], diffs])
    
    change_flashes = flashes[diffs == 1]
    change_flashesdt = np.round(change_flashes*(1/dt))
    # get start/stop time for session
    start_time = 1
    stop_time = int(np.round(running_timestamps[-1],2)*(1/dt))
    licks = licks[licks < stop_time/100]
    licks = np.round(licks,2)
    licksdt = np.round(licks*(1/dt))
    time_vec = np.arange(0,stop_time/100.0,dt)
    running_acceleration = compute_running_acceleration(running_speed)
    return licks, licksdt, start_time, stop_time, time_vec, running_speed, rewardsdt, flashesdt, change_flashesdt, running_acceleration

def extract_change_flashes(data):
    stims = data['stim_id']
    flashes=data['stim_on_timestamps']
    stims[np.array(stims) == 8 ] = 100 # ID 8 is omitted flash
    diffs = np.diff(stims)
    diffs[(diffs > 50) | (diffs < -50 )] = 0 # Don't count omitted change as change
    diffs[ np.abs(diffs) > 0] = 1
    diffs = np.concatenate([[0], diffs])
    change_flashes = flashes[diffs == 1]
    return change_flashes
    
def compute_running_acceleration(running_speed):
    running_speed_sm = running_mean(running_speed,5)
    running_speed_sm = np.concatenate([running_speed[0:2],running_speed_sm, running_speed[-2:]])
    acc = np.concatenate([[0], np.diff(running_speed_sm)])
    return acc

def running_mean(x,N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def extract_sdk_data(data,dt):
    licks = data['lick_timestamps']
    running_timestamps = data['running_timestamps']
    running_speed = data['running_speed']
    rewards = np.round(data['reward_timestamps'],2)
    flashes=np.round(data['stim_flash_start'],2)
    rewardsdt = np.round(rewards*(1/dt))
    flashesdt = np.round(flashes*(1/dt))
    change_flashes = np.round(data["stim_change_time"],2)
    change_flashesdt = np.round(change_flashes*(1/dt))
    # get start/stop time for session
    start_time = 1
    stop_time = int(np.round(running_timestamps[-1],2)*(1/dt))
    licks = licks[licks < stop_time/100]
    licks = np.round(licks,2)
    licksdt = np.round(licks*(1/dt))
    time_vec = np.arange(0,stop_time/100.0,dt)
    return licks, licksdt, start_time, stop_time, time_vec, running_speed, rewardsdt, flashesdt, change_flashesdt