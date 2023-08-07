import h5py 
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal.windows import exponential
from scipy.ndimage.filters import convolve1d
import matplotlib.pyplot as plt

import licking_behavior_NP.psy_output_tools as po

def get_tensor():
    filename = '/allen/programs/mindscope/workgroups/behavioral-dynamics/vbnAllUnitSpikeTensor.hdf5'
    active_tensor = h5py.File(filename)
    return active_tensor

def plot(response_type):
    fig, ax = plt.subplots()
    areas_of_interest = ['LGd', 'VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', \
        'VISam', 'LP', 'MRN', 'CA3']
    for area in areas_of_interest:
        #responses = np.concatenate([np.mean(resp, axis=1) for resp in response_type[area]])
        responses = np.concatenate(response_type[area])
        responses = [baseline_subtract(r) for r in responses]
        responses = [exponential_convolve(r, 3, True) for r in responses]

        #responses = [r/r.max() for r in responses]
        print(area, len(responses))
        mean_response = np.nanmean(responses, axis=0)
        #mean_response = mean_response/mean_response.max()
        ax.plot(mean_response)

    ax.legend(areas_of_interest)
    ax.set_xlim([0, 100])
    ax.set_ylim([-0.002, 0.015])   

def plot_by_areas(visual_hit_responses, timing_hit_responses,areas_of_interest=['VISp']):
    fig, ax = plt.subplots()
    #areas_of_interest = ['LGd', 'VISp', 'VISl', 'VISrl', 'LP', 'VISal', 'VISpm', \
    #    'VISam', 'MRN', 'CA3']
    colors = ['pink','purple','blue','lightblue','green','yellow','orange','red','black','cyan']
    for index, area in enumerate(areas_of_interest):
        responses = np.concatenate(visual_hit_responses[area])
        #responses = [baseline_subtract(r) for r in responses]
        responses = [exponential_convolve(r, 3, True) for r in responses]
        mean_response = np.nanmean(responses, axis=0)
        ax.plot(mean_response,'-',color=colors[index]) 
  
        responses = np.concatenate(timing_hit_responses[area])
        #responses = [baseline_subtract(r) for r in responses]
        responses = [exponential_convolve(r, 3, True) for r in responses]
        mean_response = np.nanmean(responses, axis=0)
        ax.plot(mean_response,'--',color=colors[index])   

def f():

    # Get data
    stim_table_file='/allen/programs/mindscope/workgroups/behavioral-dynamics/master_stim_table.csv' 
    unit_table_file='/allen/programs/mindscope/workgroups/behavioral-dynamics/master_unit_table.csv'
    units = pd.read_csv(unit_table_file)
    stim_table = pd.read_csv(stim_table_file)
    stim_table = stim_table.drop(columns='Unnamed: 0') #drop redundant column
    active_tensor = get_tensor()

    # Filter units
    good_unit_filter = [(units['isi_violations']<0.5) &
                        (units['presence_ratio']>0.9) &
                        (units['amplitude_cutoff']<0.1) &
                        (units['quality']=='good') &
                        (units['no_anomalies'])][0]
    good_units = units[good_unit_filter]
    good_unit_ids = good_units['unit_id'].values

    # Define areas
    areas_of_interest = ['LGd', 'VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', \
        'VISam', 'LP', 'MRN', 'CA3']

    # get strategy information
    summary_df = po.get_np_summary_table(100).query('experience_level == "Familiar"')
    visual_hit_responses = {area:[] for area in areas_of_interest}
    timing_hit_responses = {area:[] for area in areas_of_interest}   
    visual_ids = summary_df.query('visual_strategy_session').index.values
    timing_ids = summary_df.query('not visual_strategy_session').index.values

    # Process visuals sessions
    for session_ind, session_id in enumerate(tqdm(list(active_tensor.keys()))):
        session_stim_table = stim_table[stim_table['session_id']==int(session_id)].reset_index()
        session_tensor = active_tensor[str(session_id)]
        session_units = get_tensor_unit_table(units, session_tensor['unitIds'][()])
        hit_flashes = session_stim_table[(session_stim_table['is_change'])&\
            (session_stim_table['miss'])]
        good_unit_indices = session_units[session_units['unit_id'].isin(good_unit_ids)].index.values
        
        for area_of_interest in areas_of_interest:
            area_unit_indices = get_unit_indices_by_area(session_units, \
                session_tensor['unitIds'][()], area_of_interest)
            unit_indices = np.intersect1d(good_unit_indices, area_unit_indices)
 
            if len(unit_indices)>1:
                area_tensor = get_tensor_for_unit_selection(unit_indices, session_tensor['spikes'])
                selected = area_tensor[:, hit_flashes.index.values, :]
                if int(session_id) in visual_ids:
                    visual_hit_responses[area_of_interest].append(np.mean(selected, axis=1))        
                elif int(session_id) in timing_ids:
                    timing_hit_responses[area_of_interest].append(np.mean(selected, axis=1))        

    return visual_hit_responses, timing_hit_responses

def exponential_convolve(response_vector, tau=1, symmetrical=False):
    
    center = 0 if not symmetrical else None
    exp_filter = exponential(10*tau, center=center, tau=tau, sym=symmetrical)
    exp_filter = exp_filter/exp_filter.sum()
    filtered = convolve1d(response_vector,exp_filter[::-1])
    
    return filtered


def baseline_subtract(response_vector, baseline_window=slice(0,20)):
    
    return response_vector - np.mean(response_vector[baseline_window])
    


def get_tensor_unit_table(unit_table, tensor_unit_ids):
    '''Returns unit table with units ordered as they are in the tensor
    
    INPUTS:
        unit_table: unit dataframe with unit metadata
        tensor_unit_ids: the unit ids stored in the session tensor (ie session_tensor['unitIds'][()])
    
    OUTPUTS:
        tensor_unit_table: unit table filtered for the units in the tensor and reordered for convenient indexing
    '''
    
    units = unit_table.set_index('unit_id').loc[tensor_unit_ids].reset_index()

    return units
    

def get_unit_ids_by_area(unit_table, areaname):
    '''Get the ids for units in a particular area
    
    INPUTS:
        unit_table: unit dataframe
        areaname: name of area as it appears in the units table 'structure_with_layer' column
        
    OUTPUTS:
        list of unit ids for units in the area of interest
    '''
    
    unit_ids = unit_table[unit_table['structure_with_layer']==areaname]['unit_id'].values
    
    return unit_ids
    

def get_unit_indices_by_area(unit_table, tensor_unit_ids, area_of_interest, method='equal'):
    '''
    Get the indices for the unit dimension of the tensor for only those units in a given area
    
    INPUTS:
        unit_table: unit dataframe for session
        tensor_unit_ids: the unit ids stored in the session tensor (ie session_tensor['unitIds'][()])
        areaname: the area of interest for which you would like to filter units
        method: 
            if 'equal' only grab the units with an exact match to the areaname
            if 'contains' grab all units that contain the areaname in the string. This can be useful to, for example,
                grab all of the units in visual cortex regardless of area or layer (areaname would be 'VIS')
    
    OUTPUT
        the indices of the tensor for the units of interest
    '''
    
    units = get_tensor_unit_table(unit_table, tensor_unit_ids)
    if method == 'equal':
        unit_indices = units[units['structure_acronym']==area_of_interest].index.values
    
    elif method == 'contains':
        unit_indices = units[units['structure_with_layer'].str.contains(area_of_interest)].index.values
    
    return unit_indices
    

def get_tensor_for_unit_selection(unit_indices, spikes):
    '''
    Subselect a portion of the tensor for a particular set of units. You might try to do this
    with something like spikes[unit_indices] but this ends up being very slow. When the H5 file is saved,
    the data is chunked by units, so reading it out one unit at a time is much faster
    
    INPUTS:
        unit_indices: the indices of the array along the unit dimension that you'd like to extract
        spikes: the spikes tensor (ie tensor['spikes'] from the h5 file)
        
    OUTPUT:
        the subselected spikes tensor for only the units of interest
    '''
    
    s = np.zeros((len(unit_indices), spikes.shape[1], spikes.shape[2]), dtype=bool)
    for i,u in enumerate(unit_indices):
        s[i]=spikes[u]
    
    return s


