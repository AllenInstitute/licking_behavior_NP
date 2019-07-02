import numpy as np
from datetime import datetime, timedelta
from os import makedirs
from psytrack.hyperOpt import hyperOpt
from psytrack.helper.invBlkTriDiag import getCredibleInterval
from psytrack.helper.helperFunctions import read_input
import os
import matplotlib.pyplot as plt
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api import behavior_ophys_api as boa
import copy

def get_data(experiment_id,load_dir = r'/allen/aibs/technology/nicholasc/behavior_ophys'):
    '''
        Loads data from SDK interface
        ARGS: experiment_id to load
    '''
    # full_filepath = os.path.join(load_dir, 'behavior_ophys_session_{}.nwb'.format(experiment_id))
    session = BehaviorOphysSession(api=boa.BehaviorOphysLimsApi(experiment_id)) 
    return session

def check_grace_windows(session,time_point):
    '''
        Returns true if the time point is inside the grace period after reward delivery from an earned reward or auto-reward
    '''
    hit_end_times = session.trials.stop_time[session.trials.hit].values
    hit_response_time = session.trials.response_time[session.trials.hit].values
    inside_grace_window = np.any((hit_response_time < time_point ) & (hit_end_times > time_point))
    
    auto_reward_time = session.trials.change_time[(session.trials.auto_rewarded) & (~session.trials.aborted)] + .5
    auto_end_time = session.trials.stop_time[(session.trials.auto_rewarded) & (~session.trials.aborted)]
    inside_auto_window = np.any((auto_reward_time < time_point) & (auto_end_time > time_point))
    return inside_grace_window | inside_auto_window

def format_all_sessions(all_flash_df, task_zero=True):
    change_flashes = []
    lick_flashes = all_flash_df.lick_bool.values
    prev_image = all_flash_df.loc[0].image_name
    for index, row in all_flash_df.iterrows():
        # Parse licks
        start_time = row.start_time
        stop_time = row.start_time + 0.75
        # Parse change_flashes
        if index > 0:
            this_change_flash = not ((row.image_name == prev_image) | (row.omitted) | (prev_image =='omitted'))
        else:
            this_change_flash = False
        prev_image = row.image_name
        change_flashes.append(this_change_flash)

    # map boolean vectors to the format psytrack wants
    licks = np.array([2 if x else 1 for x in lick_flashes])   
    if task_zero:
        changes = np.array([1 if x else 0 for x in change_flashes])[:,np.newaxis]
    else:
        changes = np.array([1 if x else -1 for x in change_flashes])[:,np.newaxis]
   
    session_dex = np.unique(all_flash_df.session_index.values)
    dayLength = []
    for dex in session_dex:
        dayLength.append(np.sum(all_flash_df.session_index.values == dex))

    inputDict = {   'task': changes}
    psydata = { 'y': licks, 
                'inputs':inputDict,
                'dayLength':np.array(dayLength)  }
    return psydata



def format_session(session,remove_consumption=True):
    '''
        Formats the data into the requirements of Psytrack
        ARGS:
            data outputed from SDK
            remove_consumption, if True (Default), then removes flashes following rewards   
        Returns:
            data formated for psytrack. A dictionary with key/values:
            psydata['y'] = a vector of no-licks (1) and licks(2) for each flashes
            psydata['inputs'] = a dictionary with each key an input ('random','timing', 'task', etc)
                each value has a 2D array of shape (N,M), where N is number of flashes, and M is 1 unless you want to look at history/flash interaction terms
    '''     
    # # It should be something as simple as this
    # change_flashes = session.stimlus_presentations.change_image 
    # lick_flashes = len(session.stimulus_presentations.lick_times) > 0
    if len(session.licks) < 10:
        raise Exception('Less than 10 licks in this session')   
 
    change_flashes = []
    lick_flashes = []
    omitted_flashes = []
    timing_flashes4 = []
    timing_flashes5 = []
    false_alarms = []
    hits =[]
    misses = []
    correct_rejects = []
    aborts = []
    auto_rewards=[]
    start_times=[]
    all_licks = session.licks.time.values
    num_since_lick = 0
    last_num_since_lick =0
    for index, row in session.stimulus_presentations.iterrows():
        # Parse licks
        start_time = row.start_time
        stop_time = row.start_time + 0.75
        this_licks = np.sum((all_licks > start_time) & (all_licks < stop_time)) > 0
        # Parse timing drive
        last_num_since_lick = num_since_lick
        if this_licks:
            num_since_lick = 0
        else:
            num_since_lick +=1
        
        # Parse change_flashes
        if index > 0:
            prev_image = session.stimulus_presentations.image_name.loc[index -1]
            this_change_flash = not ((row.image_name == prev_image) | (row.omitted) | (prev_image =='omitted'))
        else:
            this_change_flash = False
        # Parse Trial Data 
        # Pack up results
        if (not check_grace_windows(session, start_time)) | (not remove_consumption) :
            trial = get_trial(session,start_time, stop_time)
            lick_flashes.append(this_licks)
            change_flashes.append(this_change_flash)
            omitted_flashes.append(row.omitted)
            aborts.append(trial['aborted']) 
            false_alarms.append(trial['false_alarm'])
            misses.append(trial['miss'])
            hits.append(trial['hit'])
            correct_rejects.append(trial['correct_reject'])
            auto_rewards.append(trial['auto_rewarded'])
            start_times.append(start_time)
            timing_flashes4.append(timing_curve4(last_num_since_lick))
            timing_flashes5.append(timing_curve5(last_num_since_lick))
    # map boolean vectors to the format psytrack wants
    licks = np.array([2 if x else 1 for x in lick_flashes])   
    changes0 = np.array([1 if x else 0 for x in change_flashes])[:,np.newaxis]
    changes1 = np.array([1 if x else -1 for x in change_flashes])[:,np.newaxis]
    changesCR= np.array([0 if x else -1 for x in change_flashes])[:,np.newaxis]
    omitted = np.array([1 if x else 0 for x in omitted_flashes])[:,np.newaxis]
    timing4 = np.array(timing_flashes4)[:,np.newaxis]
    timing5 = np.array(timing_flashes5)[:,np.newaxis] 
    # Make Dictionary of inputs, and all data
    inputDict = {   'task0': changes0,
                    'task1': changes1,
                    'taskCR': changesCR,
                    'omissions' : omitted,
                    'timing4': timing4,
                    'timing5': timing5 }
    psydata = { 'y': licks, 
                'inputs':inputDict, 
                'false_alarms': false_alarms,
                'correct_reject': correct_rejects,
                'hits': hits,
                'misses':misses,
                'aborts':aborts,
                'auto_rewards':auto_rewards,
                'start_times':start_times }
    return psydata

def timing_curve4(num_flashes):
    '''
        Defines a timing function that maps the number of flashes from the last lick to the timing drive to lick on this flash
        num_flashes = 0 means I licked on this flash
        num_flashes = 1 means I licked on the last flash, but not this one. 
    '''
    if num_flashes < 0:
        raise Exception ('Timing cant be negative')
    elif num_flashes == 0:
        return 0
    elif num_flashes < 4:
        return -1
    elif num_flashes == 4:
        return -.5
    elif num_flashes == 5:
        return 0.5
    else:
        return 1

def timing_curve5(num_flashes):
    '''
        Defines a timing function that maps the number of flashes from the last lick to the timing drive to lick on this flash
        num_flashes = 0 means I licked on this flash
        num_flashes = 1 means I licked on the last flash, but not this one. 
    '''
    if num_flashes < 0:
        raise Exception ('Timing cant be negative')
    elif num_flashes == 0:
        return 0
    elif num_flashes < 5:
        return -1
    elif num_flashes == 5:
        return -.5
    elif num_flashes == 6:
        return 0.5
    else:
        return 1



def get_trial(session, start_time,stop_time):
    ''' 
        returns the behavioral state for a flash
    '''
    if start_time > stop_time:
        raise Exception('Start time cant be later than stop time') 
    trial = session.trials[(session.trials.start_time <= start_time) & (session.trials.stop_time >= stop_time)]
    if len(trial) == 0:
        trial = session.trials[(session.trials.start_time <= start_time) & (session.trials.stop_time+0.75 >= stop_time)]
        if len(trial) == 0:
            labels = {  'aborted':False,
                'hit': False,
                'miss': False,
                'false_alarm': False,
                'correct_reject': False,
                'auto_rewarded': False  }
            return labels
        else:
            trial = trial.iloc[0]
    else:
        trial = trial.iloc[0]

    labels = {  'aborted':trial.aborted,
                'hit': trial.hit,
                'miss': trial.miss,
                'false_alarm': trial.false_alarm,
                'correct_reject': trial.correct_reject & (not trial.aborted),
                'auto_rewarded': trial.auto_rewarded & (not trial.aborted)  }
    if trial.hit:
        labels['hit'] = (trial.change_time >= start_time) & (trial.change_time < stop_time )
    if trial.miss:
        labels['miss'] = (trial.change_time >= start_time) & (trial.change_time < stop_time )
    if trial.false_alarm:
        labels['false_alarm'] = (trial.change_time >= start_time) & (trial.change_time < stop_time )
    if (trial.correct_reject) &  (not trial.aborted):
        labels['correct_reject'] = (trial.change_time >= start_time) & (trial.change_time < stop_time )
    if trial.aborted:
        if len(trial.lick_times) >= 1:
            labels['aborted'] = (trial.lick_times[0] >= start_time ) & (trial.lick_times[0] < stop_time)
        else:
            labels['aborted'] = (trial.start_time >= start_time ) & (trial.start_time < stop_time)
    if trial.auto_rewarded & (not trial.aborted):
            labels['auto_rewarded'] = (trial.change_time >= start_time) & (trial.change_time < stop_time )
    return labels
    
def fit_weights(psydata, BIAS=True,TASK0=True, TASK1=False,TASKCR = False, OMISSIONS=False,TIMING4=False,TIMING5=False,fit_overnight=False):
    '''
        does weight and hyper-parameter optimization on the data in psydata
        Args: 
            psydata is a dictionary with key/values:
            psydata['y'] = a vector of no-licks (1) and licks(2) for each flashes
            psydata['inputs'] = a dictionary with each key an input ('random','timing', 'task', etc)
                each value has a 2D array of shape (N,M), where N is number of flashes, and M is 1 unless you want to look at history/flash interaction terms

        RETURNS:
        hyp
        evd
        wMode
        hess
    '''
    weights = {}
    if BIAS: weights['bias'] = 1
    if TASK0: weights['task0'] = 1
    if TASK1: weights['task1'] = 1
    if TASKCR: weights['taskCR'] = 1
    if OMISSIONS: weights['omissions'] = 1
    if TIMING4: weights['timing4'] = 1
    if TIMING5: weights['timing5'] = 1
    print(weights)

    K = np.sum([weights[i] for i in weights.keys()])
    hyper = {'sigInit': 2**4.,
            'sigma':[2**-4.]*K,
            'sigDay': 2**4}
    optList=['sigma']
    hyp,evd,wMode,hess =hyperOpt(psydata,hyper,weights, optList)
    credibleInt = getCredibleInterval(hess)
    return hyp, evd, wMode, hess, credibleInt, weights

def compute_ypred(psydata, wMode, weights):
    g = read_input(psydata, weights)
    gw = g*wMode.T
    total_gw = np.sum(g*wMode.T,axis=1)
    pR = 1/(1+np.exp(-total_gw))
    pR_each = 1/(1+np.exp(-gw))
    return pR, pR_each

def transform(series):
    '''
        passes the series through the logistic function
    '''
    return 1/(1+np.exp(-(series)))

def get_flash_index_session(session, time_point):
    '''
        Returns the flash index of a time point
    '''
    return np.where(session.stimulus_presentations.start_time.values < time_point)[0][-1]

def get_flash_index(psydata, time_point):
    '''
        Returns the flash index of a time point
    '''
    if time_point > psydata['start_times'][-1] + 0.75:
        return np.nan
    return np.where(np.array(psydata['start_times']) < time_point)[0][-1]


def moving_mean(values, window):
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, 'valid')
    return mm

def plot_weights(session, wMode,weights,psydata,errorbar=None, ypred=None,START=0, END=0,remove_consumption=True,validation=True,session_labels=None, seedW = None,ypred_each = None):
    K,N = wMode.shape    
    if START <0: START = 0
    if START > N: raise Exception(" START > N")
    if END <=0: END = N
    if END > N: END = N
    if START >= END: raise Exception("START >= END")

    weights_list = []
    for i in sorted(weights.keys()):
        weights_list += [i]*weights[i]
   
    my_colors=['blue','green','purple','red']  
    if 'dayLength' in psydata:
        dayLength = np.concatenate([[0],np.cumsum(psydata['dayLength'])])
    else:
        dayLength = []

    if (not (type(ypred) == type(None))) & validation:
        fig,ax = plt.subplots(nrows=4,ncols=1, figsize=(10,10))
        ax[3].plot(ypred, 'k',alpha=0.3,label='Full Model')
        if not( type(ypred_each) == type(None)):
            for i in np.arange(0, len(weights_list)):
                ax[3].plot(ypred_each[:,i], linestyle="-", lw=3, alpha = 0.3,color=my_colors[i],label=weights_list[i])        
        ax[3].plot(moving_mean(psydata['y']-1,25), 'b',alpha=0.5,label='data (n=25)')
        ax[3].set_ylim(0,1)
        ax[3].set_ylabel('Lick Prob',fontsize=12)
        ax[3].set_xlabel('Flash #',fontsize=12)
        ax[3].set_xlim(START,END)
        ax[3].legend()
        ax[3].tick_params(axis='both',labelsize=12)
    elif validation:
        fig,ax = plt.subplots(nrows=3,ncols=1, figsize=(10,8))
    else:
        fig,ax = plt.subplots(nrows=2,ncols=1, figsize=(10,6)  )
    for i in np.arange(0, len(weights_list)):
        ax[0].plot(wMode[i,:], linestyle="-", lw=3, color=my_colors[i],label=weights_list[i])        
        ax[0].fill_between(np.arange(len(wMode[i])), wMode[i,:]-2*errorbar[i], 
            wMode[i,:]+2*errorbar[i],facecolor=my_colors[i], alpha=0.2)    
        ax[1].plot(transform(wMode[i,:]), linestyle="-", lw=3, color=my_colors[i],label=weights_list[i])
        ax[1].fill_between(np.arange(len(wMode[i])), transform(wMode[i,:]-2*errorbar[i]), 
            transform(wMode[i,:]+2*errorbar[i]),facecolor=my_colors[i], alpha=0.2)                  
        if not (type(seedW) == type(None)):
            ax[0].plot(seedW[i,:], linestyle="--", lw=2, color=my_colors[i], label= "seed "+weights_list[i])
            ax[1].plot(transform(seedW[i,:]), linestyle="--", lw=2, color=my_colors[i], label= "seed "+weights_list[i])
    ax[0].plot([0,np.shape(wMode)[1]], [0,0], 'k--',alpha=0.2)
    ax[0].set_ylabel('Weight',fontsize=12)
    ax[0].set_xlabel('Flash #',fontsize=12)
    ax[0].set_xlim(START,END)
    ax[0].legend()
    ax[0].tick_params(axis='both',labelsize=12)
    for i in np.arange(0, len(dayLength)-1):
        ax[0].axvline(dayLength[i],color='k',alpha=0.2)
        if not type(session_labels) == type(None):
            ax[0].text(dayLength[i],ax[0].get_ylim()[1], session_labels[i][0:10],rotation=25)
    ax[1].set_ylim(0,1)
    ax[1].set_ylabel('Lick Prob',fontsize=12)
    ax[1].set_xlabel('Flash #',fontsize=12)
    ax[1].set_xlim(START,END)
    ax[1].legend(loc='upper right')
    ax[1].tick_params(axis='both',labelsize=12)
    for i in np.arange(0, len(dayLength)-1):
        ax[1].plot([dayLength[i], dayLength[i]],[0,1], 'k-',alpha=0.2)

    if validation:
        first_start = session.trials.loc[0].start_time
        jitter = 0.025
        #hits = 0
        #miss = 0
        #fa = 0
        #cr = 0
        #abort = 0
        #auto =0
        #for index, row in session.trials.iterrows(): 
        #    if row.hit:
        #        ax[2].plot(get_flash_index(psydata, row.change_time), 1+np.random.randn()*jitter, 'bo',alpha=0.2)
        #        hits +=1
        #    elif row.miss:
        #        ax[2].plot(get_flash_index(psydata, row.change_time), 1.5+np.random.randn()*jitter, 'ro',alpha = 0.2)   
        #        if not np.isnan(get_flash_index(psydata,row.change_time)):
        #            miss +=1
        #    elif row.false_alarm:
        #        ax[2].plot(get_flash_index(psydata, row.change_time), 2.5+np.random.randn()*jitter, 'ko',alpha = 0.2)
        #        fa +=1
        #    elif row.correct_reject & (not row.aborted):
        #        ax[2].plot(get_flash_index(psydata, row.change_time), 2+np.random.randn()*jitter, 'co',alpha = 0.2)   
        #        cr +=1
        #    elif row.aborted:
        #        if len(row.lick_times) >= 1:
        #            ax[2].plot(get_flash_index(psydata, row.lick_times[0]), 3+np.random.randn()*jitter, 'ko',alpha=0.2)  
        #        else:  
        #            ax[2].plot(get_flash_index(psydata, row.start_time), 3+np.random.randn()*jitter, 'ko',alpha=0.2)  
        #        abort +=1
        #    else:
        #        raise Exception('Trial had no classification')
        #    if row.auto_rewarded & (not row.aborted):
        #        ax[2].plot(get_flash_index(psydata, row.change_time), 3.5+np.random.randn()*jitter, 'go',alpha=0.2)    
        #        auto+=1 
     
        for i in np.arange(0, len(psydata['hits'])):
            if psydata['hits'][i]:
                ax[2].plot(i, 1+np.random.randn()*jitter, 'bo',alpha=0.2)
            elif psydata['misses'][i]:
                ax[2].plot(i, 1.5+np.random.randn()*jitter, 'ro',alpha = 0.2)   
            elif psydata['false_alarms'][i]:
                ax[2].plot(i, 2.5+np.random.randn()*jitter, 'ko',alpha = 0.2)
            elif psydata['correct_reject'][i] & (not psydata['aborts'][i]):
                ax[2].plot(i, 2+np.random.randn()*jitter, 'co',alpha = 0.2)   
            elif psydata['aborts'][i]:
                ax[2].plot(i, 3+np.random.randn()*jitter, 'ko',alpha=0.2)  
            if psydata['auto_rewards'][i] & (not psydata['aborts'][i]):
                ax[2].plot(i, 3.5+np.random.randn()*jitter, 'go',alpha=0.2)    
    
        ax[2].set_yticks([1,1.5,2,2.5,3,3.5])
        ax[2].set_yticklabels(['hits','miss','CR','FA','abort','auto'],{'fontsize':12})
        ax[2].set_xlim(START,END)
        ax[2].set_xlabel('Flash #',fontsize=12)
        ax[2].tick_params(axis='both',labelsize=12)

    plt.tight_layout()
    #print(str(hits) +" "+str(np.sum(psydata['hits'])))
    #print(str(miss)+" "+str(np.sum(psydata['misses'])))
    #print(str(fa)+" "+str(np.sum(psydata['false_alarms'])))
    #print(str(cr)+" "+str(np.sum(psydata['correct_reject'])))
    #print(str(abort)+" "+str(np.sum(psydata['aborts'])))
    #print(str(auto)+" "+ str(np.sum(psydata['auto_rewards'])))
    

def check_lick_alignment(session, psydata):
    plt.figure(figsize=(10,5))
    plt.plot(session.stimulus_presentations.start_time.values,psydata['y']-1, 'ko-')
    all_licks = session.licks
    for index, lick in all_licks.iterrows():
        plt.plot([lick.time, lick.time], [0.9, 1.1], 'r')
    plt.xlabel('time (s)')
    for index, row in session.trials.iterrows():
        if row.hit:
            plt.plot(row.change_time, 1.2, 'bo')
        elif row.miss:
            plt.plot(row.change_time, 1.25, 'gx')   
        elif row.false_alarm:
            plt.plot(row.change_time, 1.3, 'ro')
        elif row.correct_reject:
            plt.plot(row.change_time, 1.35, 'cx')   
        elif row.aborted:
            if len(row.lick_times) >= 1:
                plt.plot(row.lick_times[0], 1.4, 'kx')   
            else:  
                plt.plot(row.start_time, 1.4, 'kx')  
        else:
            raise Exception('Trial had no classification')
   


def generateSim_VB(K=4,
                N=64000,
                hyper={},
                boundary=4.0,
                iterations=20,
                seed=None,
                savePath=None):
    """Simulates weights, in addition to inputs and multiple realizations
    of responses. Simulation data is either saved to a file or returned
    directly.
    Args:
        K : int, number of weights to simulate
        N : int, number of trials to simulate
        hyper : dict, hyperparameters and initial values used to construct the
            prior. Default is none, can include sigma, sigInit, sigDay
        boundary : float, weights are reflected from this boundary
            during simulation, is a symmetric +/- boundary
        iterations : int, # of behavioral realizations to simulate,
            same input and weights can render different choice due
            to probabilistic model, iterations are saved in 'all_Y'
        seed : int, random seed to make random simulations reproducible
        savePath : str, if given creates a folder and saves simulation data
            in a file; else data is returned
    Returns:
        save_path | (if savePath) : str, the name of the folder+file where
            simulation data was saved in the local directory
        save_dict | (if no SavePath) : dict, contains all relevant info
            from the simulation 
    """

    # Reproducability
    np.random.seed(seed)

    # Supply default hyperparameters if necessary
    sigmaDefault = 2**np.random.choice([-4.0, -5.0, -6.0, -7.0, -8.0], size=K)
    if "sigma" not in hyper:
        sigma = sigmaDefault
    elif hyper["sigma"] is None:
        sigma = sigmaDefault
    elif np.isscalar(hyper["sigma"]):
        sigma = np.array([hyper["sigma"]] * K)
    elif ((type(hyper["sigma"]) in [np.ndarray, list]) and
          (len(hyper["sigma"]) != K)):
        sigma = hyper["sigma"]
    else:
        raise Exception(
            "hyper['sigma'] must be either a scalar or a list or array of len K"
        )

    sigInitDefault = np.array([4.0] * K)
    if "sigInit" not in hyper:
        sigInit = sigInitDefault
    elif hyper["sigInit"] is None:
        sigInit = sigInitDefault
    elif np.isscalar(hyper["sigInit"]):
        sigInit = np.array([hyper["sigInit"]] * K)
    elif (type(hyper["sigInit"]) in [np.ndarray, list]) and (len(hyper["sigInit"]) != K):
        sigInit = hyper["sigInit"]
    else:
        raise Exception("hyper['sigInit'] must be either a scalar or \
            a list or array of len K")

    # sigDay not yet supported!
    if "sigDay" in hyper and hyper["sigDay"] is not None:
        raise Exception("sigDay not yet supported, please omit from hyper")

    # -------------
    # Simulation
    # -------------

    # Simulate inputs
    X = np.random.normal(size=(N, K))
    X[:,0] = 1
    X[:,1] = np.abs(np.sin(np.arange(0,N,3.14/10)))[0:N]
    X[:,2] = 0   
    X[np.random.normal(size=(N,)) > 1,2] = 1

    # Simulate weights
    E = np.zeros((N, K))
    E[0] = np.random.normal(scale=sigInit, size=K)
    E[1:] = np.random.normal(scale=sigma, size=(N - 1, K))
    W = np.cumsum(E, axis=0)

    # Impose a ceiling and floor boundary on W
    for i in range(len(W.T)):
        cross = (W[:, i] < -boundary) | (W[:, i] > boundary)
        while cross.any():
            ind = np.where(cross)[0][0]
            if W[ind, i] < -boundary:
                W[ind:, i] = -2 * boundary - W[ind:, i]
            else:
                W[ind:, i] = 2 * boundary - W[ind:, i]
            cross = (W[:, i] < -boundary) | (W[:, i] > boundary)

    # Save data
    save_dict = {
        "sigInit": sigInit,
        "sigma": sigma,
        "seed": seed,
        "W": W,
        "X": X,
        "K": K,
        "N": N,
    }

    # Simulate behavioral realizations in advance
    pR = 1.0 / (1.0 + np.exp(-np.sum(X * W, axis=1)))

    all_simy = []
    for i in range(iterations):
        sim_y = (pR > np.random.rand(
            len(pR))).astype(int) + 1  # 1 for L, 2 for R
        all_simy += [sim_y]

    # Update saved data to include behavior
    save_dict.update({"all_Y": all_simy})

    # Save & return file path OR return simulation data
    if savePath is not None:
        # Creates unique file name from current datetime
        folder = datetime.now().strftime("%Y%m%d_%H%M%S") + savePath
        makedirs(folder)

        fullSavePath = folder + "/sim.npz"
        np.savez_compressed(fullSavePath, save_dict=save_dict)

        return fullSavePath

    else:
        return save_dict


def sample_model(psydata):
    bootdata = copy.copy(psydata)    
    if not ('ypred' in bootdata):
        raise Exception('You need to compute y-prediction first')
    temp = np.random.random(np.shape(bootdata['ypred'])) < bootdata['ypred']
    licks = np.array([2 if x else 1 for x in temp])   
    bootdata['y'] = licks
    return bootdata


def bootstrap_model(psydata, ypred, weights,seedW,plot_this=True):
    psydata['ypred'] =ypred
    bootdata = sample_model(psydata)
    bK = np.sum([weights[i] for i in weights.keys()])
    bhyper = {'sigInit': 2**4.,
        'sigma':[2**-4.]*bK,
        'sigDay': 2**4}
    boptList=['sigma']
    bhyp,bevd,bwMode,bhess =hyperOpt(bootdata,bhyper,weights, boptList)
    bcredibleInt = getCredibleInterval(bhess)
    if plot_this:
        plot_weights(None, bwMode, weights, bootdata, errorbar=bcredibleInt, validation=False,seedW =seedW )
    return (bootdata, bhyp, bevd, bwMode, bhess, bcredibleInt)

def bootstrap(numboots, psydata, ypred, weights, seedW, plot_each=False):
    boots = []
    for i in np.arange(0,numboots):
        print(i)
        boot = bootstrap_model(psydata, ypred, weights, seedW,plot_this=plot_each)
        boots.append(boot)
    return boots

def plot_bootstrap(boots, hyp, weights, seedW, credibleInt):
    plot_bootstrap_recovery_prior(boots,hyp, weights)
    plot_bootstrap_recovery_weights(boots,hyp, weights,seedW,credibleInt)


def plot_bootstrap_recovery_prior(boots,hyp,weights):
    fig,ax = plt.subplots(figsize=(3,4))
    my_colors=['blue','green','purple','red']  
    for i in np.arange(0, len(hyp['sigma'])):
        plt.plot(i,hyp['sigma'][i], 'o', color=my_colors[i])
    plt.yscale('log')
    plt.ylim(0.001, 20)
    ax.set_xticks([0,1,2])
    weights_list = []
    for i in sorted(weights.keys()):
        weights_list += [i]*weights[i]
    ax.set_xticklabels(weights_list)
    plt.ylabel('Smoothing Prior, $\sigma$')
    for boot in boots:
        plt.plot(boot[1]['sigma'], 'kx')
    plt.tight_layout()

def plot_bootstrap_recovery_weights(boots,hyp,weights,wMode,errorbar):
    fig,ax = plt.subplots( figsize=(10,3.5))
    K,N = wMode.shape
    plt.xlim(0,N)
    plt.xlabel('Flash #',fontsize=12)
    plt.ylabel('Weight',fontsize=12)
    ax.tick_params(axis='both',labelsize=12)

    my_colors=['blue','green','purple','red']  
    for i in np.arange(0, K):
        plt.plot(wMode[i,:], "-", lw=3, color=my_colors[i])
        ax.fill_between(np.arange(len(wMode[i])), wMode[i,:]-2*errorbar[i], 
            wMode[i,:]+2*errorbar[i],facecolor=my_colors[i], alpha=0.2)    

        for boot in boots:
            plt.plot(boot[3][i,:], '--', color=my_colors[i], alpha=0.5)
    plt.tight_layout()



def dropout_analysis(psydata, BIAS=True,TASK0=True, TASK1=False,TASKCR = False, OMISSIONS=False,TIMING4=True,TIMING5=False):
    models =[]
    labels=[]
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, TIMING4=TIMING4,TIMING5=TIMING5)
    models.append((hyp, evd, wMode, hess, credibleInt,weights))
    labels.append('Full')
    if BIAS:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=False, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, TIMING4=TIMING4,TIMING5=TIMING5)    
        models.append((hyp, evd, wMode, hess, credibleInt,weights))
        labels.append('Bias')
    if TASK0:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=False,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, TIMING4=TIMING4,TIMING5=TIMING5)    
        models.append((hyp, evd, wMode, hess, credibleInt,weights))
        labels.append('Task0')
    if TASK1:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=False, TASKCR=TASKCR, OMISSIONS=OMISSIONS, TIMING4=TIMING4,TIMING5=TIMING5)    
        models.append((hyp, evd, wMode, hess, credibleInt,weights))
        labels.append('Task1')
    if TASKCR:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=False, OMISSIONS=OMISSIONS, TIMING4=TIMING4,TIMING5=TIMING5)    
        models.append((hyp, evd, wMode, hess, credibleInt,weights))
        labels.append('TaskCR')
    if (TASK0 & TASK1) | (TASK0 & TASKCR) | (TASK1 & TASKCR):
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=False,TASK1=False, TASKCR=False, OMISSIONS=OMISSIONS, TIMING4=TIMING4,TIMING5=TIMING5)    
        models.append((hyp, evd, wMode, hess, credibleInt,weights))
        labels.append('All Task')
    if OMISSIONS:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=False, TIMING4=TIMING4,TIMING5=TIMING5)    
        models.append((hyp, evd, wMode, hess, credibleInt,weights))
        labels.append('Omissions')
    if TIMING4:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, TIMING4=False,TIMING5=TIMING5)    
        models.append((hyp, evd, wMode, hess, credibleInt,weights))
        labels.append('Timing4')
    if TIMING5:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, TIMING4=TIMING4,TIMING5=False)    
        models.append((hyp, evd, wMode, hess, credibleInt,weights))
        labels.append('Timing5')
    if TIMING4 & TIMING5:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, TIMING4=False,TIMING5=False)    
        models.append((hyp, evd, wMode, hess, credibleInt,weights))
        labels.append('All timing')

    return models,labels

def plot_dropout(models, labels):
    plt.figure(figsize=(10,3.5))
    ax = plt.gca()
    for i in np.arange(0,len(models)):
        plt.plot(i, (1-models[i][1]/models[0][1])*100, 'ko')
    #plt.xlim(0,N)
    plt.xlabel('Model Component',fontsize=12)
    plt.ylabel('% change in evidence',fontsize=12)
    ax.tick_params(axis='both',labelsize=12)
    ax.set_xticks(np.arange(0,len(models)))
    ax.set_xticklabels(labels)
    plt.tight_layout()
    ax.axhline(0,color='k',alpha=0.2)
    plt.ylim([-20,5])

def plot_summaries(psydata):
    fig,ax = plt.subplots(nrows=8,ncols=1, figsize=(10,10),frameon=False)
    ax[0].plot(moving_mean(psydata['hits'],80),'b')
    ax[0].set_ylim(0,.15); ax[0].set_ylabel('hits')
    ax[1].plot(moving_mean(psydata['misses'],80),'r')
    ax[1].set_ylim(0,.15); ax[1].set_ylabel('misses')
    ax[2].plot(moving_mean(psydata['false_alarms'],80),'g')
    ax[2].set_ylim(0,.15); ax[2].set_ylabel('false_alarms')
    ax[3].plot(moving_mean(psydata['correct_reject'],80),'c')
    ax[3].set_ylim(0,.15); ax[3].set_ylabel('correct_reject')
    ax[4].plot(moving_mean(psydata['aborts'],80),'b')
    ax[4].set_ylim(0,.4); ax[4].set_ylabel('aborts')
    total_rate = moving_mean(psydata['hits'],80)+ moving_mean(psydata['misses'],80)+moving_mean(psydata['false_alarms'],80)+ moving_mean(psydata['correct_reject'],80)
    ax[5].plot(total_rate,'k')
    ax[5].set_ylim(0,.15); ax[5].set_ylabel('trial-rate')
    #ax[5].plot(total_rate,'b')
    ax[6].set_ylim(0,.15); ax[6].set_ylabel('d\' trials')
    ax[7].set_ylim(0,.15); ax[7].set_ylabel('d\' flashes')   
    for i in np.arange(0,len(ax)):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].yaxis.set_ticks_position('left')
        ax[i].xaxis.set_ticks_position('bottom')
        ax[i].set_xticklabels([])

    
