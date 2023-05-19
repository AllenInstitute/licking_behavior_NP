import json
import numpy as np
import pandas as pd
from pathlib import Path

from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorNeuropixelsProjectCache

'''
This is a set of general purpose functions for interacting with the SDK
Alex Piet, alexpiet@gmail.com
11/5/2019
updated 01/22/2020
updated 04/07/2020
updated 03/01/2021
updated 02/11/2022
ported to NP data 05/2023
'''

BEHAVIOR_DIR = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/NP/behavior/'


def get_directory(version,verbose=False,subdirectory=None,group=None):
    root_directory  = BEHAVIOR_DIR 
    if subdirectory =='fits':
        subdir = 'session_fits/'
    elif subdirectory == "strategy_df":
        subdir = 'session_strategy_df/'
    elif subdirectory == "licks_df":
        subdir = 'session_licks_df/'
    elif subdirectory == 'interval_df':
        subdir = 'session_interval_df/'
    elif subdirectory == 'summary':
        subdir = 'summary_data/'
    elif subdirectory == 'figures':
        subdir = 'figures_summary/'
    elif subdirectory == 'session_figures':
        subdir = 'figures_sessions/'
    elif subdirectory == 'training_figures':
        subdir = 'figures_training/'
    elif subdirectory is None:
        subdir = ''
    else:
        raise Exception('Unkown subdirectory')
    if (group is not None) and (group != ""):
        subdir += group+'/'

    directory = root_directory+'psy_fits_v'+str(version)+'/'+subdir
    return directory

def get_np_manifest():
    '''
        Returns a dataframe of the NP sessions
    '''
    cache_dir = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/NP/data/'
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=Path(cache_dir))
    np_table = cache.get_ecephys_session_table(filter_abnormalities=False)
    np_table = np_table.sort_index()
    return np_table
 

def load_version_parameters(VERSION):
    json_path = BEHAVIOR_DIR+'psy_fits_v'+str(VERSION)+'/summary_data/behavior_model_params.json'
    with open(json_path,'r') as json_file:
        format_options = json.load(json_file)
    return format_options

def get_data(esid):
    '''
        Loads data from SDK interface
        ARGS: ecephys_session_id to load
    '''

    # Get SDK session object
    print('Loading SDK object')
    cache_dir = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/NP/data/'
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=Path(cache_dir))
    session = cache.get_ecephys_session(ecephys_session_id=esid)

    # Remove Passive session:
    print('removing passive session stimuli')
    session.stimulus_presentations_np = session.stimulus_presentations.query('active')
    session.stimulus_presentations_np['omitted'] \
        = session.stimulus_presentations_np['omitted'].astype(bool)

    # Remove receptive field columns
    drop_cols = ['color','contrast','orientation','position_x','position_y',\
        'spatial_frequency','temporal_frequency']
    session.stimulus_presentations_np.drop(columns=drop_cols,inplace=True)

    print('Adding stimulus annotations')
    # Get extended stimulus presentations
    add_licks_each_flash(session) 
    add_rewards_each_flash(session)
    #add_time_from_last_change(session.stimulus_presentations_np) 
    #add_time_from_last_lick(session.stimulus_presentations_np, session.licks)
    #add_time_from_last_reward(session.stimulus_presentations_np, session.rewards)
    return session

def add_licks_each_flash(session):
    licks_each_flash = add_licks_each_flash_inner(session.stimulus_presentations_np,
                                            session.licks)
    session.stimulus_presentations_np['licks'] = licks_each_flash
    session.stimulus_presentations_np['licked'] = [True if len(licks) > 0 else False \
        for licks in session.stimulus_presentations_np.licks.values]   

def add_licks_each_flash_inner(stimulus_presentations_df, licks_df,
                     range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the timestamps of licks that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        licks_df (pd.DataFrame): lick dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        licks_each_flash (pd.Series): lick times that fell within the window 
    '''

    lick_times = licks_df['timestamps'].values
    stimulus_presentations_df['next_start'] = stimulus_presentations_df['start_time'].shift(-1)
    stimulus_presentations_df.at[stimulus_presentations_df.index[-1], 'next_start'] = \
        stimulus_presentations_df.iloc[-1]['start_time'] + .75
    licks_each_flash = stimulus_presentations_df.apply(
        lambda row: lick_times[
            ((
                lick_times > row["start_time"]
            ) & (
                lick_times <= row["next_start"]
            ))
        ],
        axis=1,
    )
    stimulus_presentations_df.drop(columns=['next_start'], inplace=True)
    return licks_each_flash

def add_rewards_each_flash(session):
    '''
    Append a column to stimulus_presentations which contains the timestamps of rewards that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        rewards (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        nothing. session.stimulus_presentations is modified in place with 'rewards' column added
    '''

    rewards_each_flash = add_rewards_each_flash_inner(session.stimulus_presentations_np,
                                                session.rewards)
    session.stimulus_presentations_np['rewards'] = rewards_each_flash
    
def add_rewards_each_flash_inner(stimulus_presentations_df, rewards_df,
                       range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the timestamps of rewards that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        rewards_df (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        rewards_each_flash (pd.Series): reward times that fell within the window
    '''

    reward_times = rewards_df['timestamps'].values
    stimulus_presentations_df['next_start'] = stimulus_presentations_df['start_time'].shift(-1)
    stimulus_presentations_df.at[stimulus_presentations_df.index[-1], 'next_start'] = \
        stimulus_presentations_df.iloc[-1]['start_time'] + .75
    rewards_each_flash = stimulus_presentations_df.apply(
        lambda row: reward_times[
            ((
                reward_times > row["start_time"]
            ) & (
                reward_times <= row["next_start"]
            ))
        ],
        axis=1,
    )
    stimulus_presentations_df.drop(columns=['next_start'], inplace=True)

    return rewards_each_flash


def moving_mean(values, window,mode='valid'):
    '''
        Computes the moving mean of the series in values, with a square window 
        of width window
    '''
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, mode)
    return mm

def get_clean_rate(vector, length=4800):
    if len(vector) >= length:
        return vector[0:length]
    else:
        return np.concatenate([vector, [np.nan]*(length-len(vector))])

def get_clean_string(strings):
    '''
        Return a cleaned up list of weights suitable for plotting labels
    '''
    string_dict = {
        'bias':'licking bias',
        'omissions':'omission',
        'omissions0':'omission',
        'Omissions':'omission',
        'Omissions1':'post omission',
        'omissions1':'post omission',
        'task0':'visual',
        'Task0':'visual',
        'timing1D':'timing',
        'Full-Task0':'full model',
        'dropout_task0':'Visual Dropout',    
        'dropout_timing1D':'Timing Dropout', 
        'dropout_omissions':'Omission Dropout',
        'dropout_omissions1':'Post Omission Dropout',
        'Sst-IRES-Cre' :'Sst Inhibitory',
        'Vip-IRES-Cre' :'Vip Inhibitory',
        'Slc17a7-IRES2-Cre' :'Excitatory',
        'strategy_dropout_index': 'strategy index',
        'num_hits':'rewards/session',
        'num_miss':'misses/session',
        'num_image_false_alarm':'false alarms/session',
        'num_post_omission_licks':'post omission licks/session',
        'num_omission_licks':'omission licks/session',
        'post_reward':'previous bout rewarded',
        'not post_reward':'previous bout unrewarded',
        'timing1':'1',
        'timing2':'2',
        'timing3':'3',
        'timing4':'4',
        'timing5':'5',
        'timing6':'6',
        'timing7':'7',
        'timing8':'8',
        'timing9':'9',
        'timing10':'10',    
        'not visual_strategy_session':'timing sessions',
        'visual_strategy_session':'visual sessions',
        'visual_only_dropout_index':'visual index',
        'timing_only_dropout_index':'timing index',
        'lick_hit_fraction_rate':'lick hit fraction',
        'session_roc':'dynamic model (AUC)',
        'miss':'misses',
        }

    clean_strings = []
    for w in strings:
        if w in string_dict.keys():
            clean_strings.append(string_dict[w])
        else:
            clean_strings.append(str(w).replace('_',' '))
    return clean_strings

def get_clean_session_names(strings):
    string_dict = {
        1:'F1',
        2:'F2',
        3:'F3',
        4:'N1',
        5:'N2',
        6:'N3',
        '1':'F1',
        '2':'F2',
        '3':'F3',
        '4':'N1',
        '5':'N2',
        '6':'N3',
        'Familiar':'familiar',
        'Novel 1':'novel',  
        'Novel >1':'novel+'}

    clean_strings = []
    for w in strings:
        if w in string_dict.keys():
            clean_strings.append(string_dict[w])
        else:
            clean_strings.append(str(w).replace('_',' '))
    return clean_strings


def get_strategy_list(version):
    '''
        Returns a sorted list of the strategies in model <version>

        Raises an exception if the model version is not recognized. 
    '''
    if version in [100]:
        strategies=['bias','omissions','omissions1','task0','timing1D']
    else:
        raise Exception('Unknown model version')
    return strategies


def get_engagement_threshold():
    '''
        Definition for engagement in units of rewards/sec
    '''
    return 1/120

    
def get_bout_threshold():
    '''
        The minimum time between licks to segment licks into bouts
        700ms, or .7s
    '''
    return .7



