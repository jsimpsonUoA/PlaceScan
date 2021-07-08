'''
Python functions to organise scan data
from PlaceScan classes, including combining and repositioning
scans.

Jonathan Simpson, jsim921@aucklanduni.ac.nz
Masters project, PAL Lab UoA, April 2018
'''

import os
import shutil
import json
import numpy as np
import copy



def combine_scans(scans, append_index=0, reposition=False, save_dir='/tmp/tmp-PlaceScan/'):
    '''
    Combine two or more PlaceSace objects into a single scan.
    The index at which the scans are combined can be specified.

    Arguments:
        --scans: A list of PlaceScan objects to combine. The priority
                 which each scan is given in the combination is
                 indicated by the order of the scan objects in the list.
        --append_index: The index of the lower priority scan to start
                 appending from
        --reposition: If true, the positions of each trace will be a 
                 continuation of the highest priority scan. Otherwise,
                 the original trace positions are unchanged.
        --save_dir: Directory to save the new scan to.

    Returns:
        --combined_scan: The PlaceScan object which is the combination
    '''
    
    try:
        sampling_rates = all(scan.sampling_rate == scans[0].sampling_rate for scan in scans)
        nptss = all(scan.npts == scans[0].npts for scan in scans)
        time_delays = all(scan.time_delay == scans[0].time_delay for scan in scans)
        header_values = all(set(scan.npy.dtype.names) == set(scans[0].npy.dtype.names) for scan in scans)
    
        if not (sampling_rates and nptss and time_delays and header_values):
            raise AttributeError
    
        combined_scan = copy.deepcopy(scans[0])
        x0 = combined_scan.x_positions[0]
        try:
            x_delta = abs(combined_scan.x_positions[1]-x0)
        except:
            x_delta = 1.0
        
        combined_scan_data = np.concatenate(tuple([scans[0].npy]+[scan.npy[append_index:] for scan in scans[1:]])) 
        combined_scan.npy = combined_scan_data     
    
        if reposition:
            size = combined_scan_data.shape[0]
            x_pos = np.arange(x0, size*x_delta, x_delta)
            try:
                combined_scan.npy[combined_scan.stage] = x_pos
            except:
                pass
        
        #Save the npy and json as a new scan, even save_dir is not specified
        combined_scan.write(save_dir, update_npy=False)
        
        #Open the saved scan
        from placescan.main import PlaceScan
        combined_scan = PlaceScan(save_dir,  trace_field=combined_scan.trace_field)     
    
    
        if save_dir[:4] == '/tmp':
            shutil.rmtree(save_dir)

    except AttributeError:
        raise Exception("PlaceScan combine_scans: Scans cannot be combined because or varying acquistion parameters (e.g. sampling_rate, npts, time_delay) and/or different scan_data headers.")

    return combined_scan    


def reposition_traces(x_pos, x_delta=None, x0=None, steps=0, flip=False):
    '''
    Function to modify the x positions of traces in a PlaceScan.
    Performs translation, respicifying of the x interval, and flipping.
    Order: x_delta  --> x0 --> steps --> flip
    
    Arguments:
        --x_pos: An array of x positions
        --x_delta: If required, the new interval between x positions
        --steps: A positive or negative integer specifying how many places
                 each trace is to be moved along in the curret x_positions.
                 Traces pushed off the end loop to the start.
        --flip: True to reverse data order about the centre x_position
        --x0: If required, the new x0 position. 
        
    Returns:
        --x_pos: The new x positions
    '''

    if x_delta:
        x_pos = np.arange(0.0, x_delta*len(x_pos), x_delta) + x_pos[0]
        
    if x0 != None:
        x_pos = x_pos+x0-x_pos[0]

    if steps:
        x_pos = np.concatenate((x_pos[-steps:],x_pos[:-steps]))
        
    if flip:
        x_pos = x_pos[::-1]
        
    return x_pos


def get_muted_indices(intervals, x_pos, keep=False):
    '''
    Function which returns an array specifying which traces
    are to be muted. The array contains 1s and 0s and is the
    same length as x_pos, with 0s in the x position intervals
    specified in mute_intervals, and 1s elsewhere.
    
    Arguments:
        --mute_intervals: A list of tuples and/or floats specifying the 
                          **closed** intervals or points over which to mute traces
        --x_pos: The x_position array of the traces
        --keep: True to invert the muting. Instead of muting the intervals given,
                this mutes everything not in the interval.
        
    Returns:
        --index_arr: An array of boolean values
    '''

    truth_array = np.zeros((len(intervals),len(x_pos)))
    for i in range(len(intervals)):
        _int = intervals[i]
        if isinstance(_int, (int, float)):
            _int = x_pos[np.argmin(np.abs(x_pos-_int))]
            truth_val = np.where(x_pos==_int, np.zeros(len(x_pos)), 1)
        else:
            truth_val = np.where(x_pos<min(_int), np.zeros(len(x_pos))+1, 0) + \
                             np.where(x_pos>max(_int), np.zeros(len(x_pos))+1, 0)
        truth_array[i] = truth_val

    truth_array = truth_array.sum(axis=0)
    index_arr = np.where(truth_array<len(intervals), np.zeros(len(x_pos)), 1)
    
    if keep:
        index_arr = np.abs(index_arr - 1)
    
    return index_arr.astype(bool)


def expand_updates(scan, rep_interval=1/20):
    '''
    Function which takes a scan where each update has more than
    one trace, and expands these traces into individual 
    x_positions. For scans where the sample is moving during an update.
    It is assumed that the stage was up to speed when the first record
    was recorded.
    
    Arguments:
        --scan: The scan to expand the traces for
        --rep_interval: The time interval (in s) between each record 
                        acquisition
        
    Returns:
        scan: The scan with the expanded updates
    '''
    
    pos_0 = scan.x_positions[0]
    
    
    if scan.place_version < 0.8:
        stage_config = next(module['config'] for module in scan.config[scan.plugins_key] if 'Stage' in module["python_class_name"])  #Not called 'python_class_name' for <0.7
    else:
        stage_config = next(module['config'] for name,module in scan.config[scan.plugins_key].items() if 'ATS' in name)
    
    tot_num = scan.trace_data.shape[0]*scan.trace_data.shape[2]
    
    try:
        start = stage_config['start']
        vel = stage_config['velocity']
        end = rep_interval*vel*(tot_num-1)
    except KeyError:
        start, end = 0, tot_num-1
    
    scan.x_positions = np.linspace(start, end, tot_num)+pos_0
    
    data = scan.trace_data
    data = np.split(data, data.shape[0])
    data = np.concatenate(data,axis=2)
    data = np.transpose(data,axes=[2,1,0,3])
    
    scan.trace_data = data
    scan.updates = tot_num
    
    return scan
    

def trace_averaging(traces, num_either_side=1):
    '''
    Function to average traces in a scan by taking the 
    average of adjacent traces and assigning the result
    to the original data.
    '''
    
    new_traces = []
    for i in range(len(traces)):
        inds = np.arange(i-num_either_side,i+num_either_side+1)
        inds = inds[np.where(inds > -1)]
        inds = inds[np.where(inds < len(traces))]
        av_trace = traces[inds[0]]
        for ind in inds[1:]:
            av_trace += traces[ind]
        new_traces.append(av_trace / len(inds))
    
    return np.array(new_traces)

def correct_picks_for_diameters(picks, x_pos, correct_to_diam,
                                true_paths, true_paths_x=None):
    '''
    Function to correct arrival time picks 
    where the travel path for each of the
    picks is not the same. A list of travel
    path lengths and the postions they correspond
    to can be used to extrapolate travel path
    lengths to all positions, or a list of travel
    paths at all positions can be passed. A reference
    travel path length is required to correct the times
    to.

    Arguments:
        --picks: The arrival time picks to be corrected
        --x_pos: The x positions of the picks
        --correct_to_diam: The rerference diameter to correct the
            picks to.
        --true_paths: A list of the true path lengths
        --true_paths_x: A list of equal length to true_paths
            specifying the x positions of the true_paths (if
            not given, then true_paths must be the same length
            as x_pos)

    Returns:
        --new_picks: The corrected picks
    '''

    if true_paths_x != x_pos:
        true_paths = np.interp(x_pos, true_paths_x, true_paths)

    new_picks = []
    for i, pick in enumerate(picks):
        vel = true_paths[i] / pick
        new_pick = correct_to_diam / vel
        new_picks.append(new_pick)

    return new_picks












