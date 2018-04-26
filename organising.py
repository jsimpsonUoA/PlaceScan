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



def combine_scans(scans, append_index=0, reposition=False, new_obj=True, 
                  save_dir='/tmp/tmp-PlaceScan/'):
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
        --new_obj: True if a completely new PlaceScan is to be returned.
                   Otherwse, the combination is assigned to the highest
                   priority scan.
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
        x_delta = abs(combined_scan.x_positions[1]-x0)
        
        combined_scan_data = np.concatenate(tuple([scans[0].npy]+[scan.npy[append_index:] for scan in scans[1:]])) 
        combined_scan.npy = combined_scan_data     

        if reposition:
            size = combined_scan_data.shape[0]
            x_pos = np.arange(x0, size*x_delta, x_delta)
            combined_scan.npy[combined_scan.rot_stage] = x_pos
        
        #Save the npy and json as a new scan, even save_dir is not specified
        combined_scan.write(save_dir, update_npy=False)
        
        #Open the saved scan
        from PlaceScan import PlaceScan
        combined_scan = PlaceScan(save_dir, trace_field=combined_scan.trace_field)     
    

        if save_dir[:4] == '/tmp':
            shutil.rmtree(save_dir)

    except AttributeError:
        raise Exception("PlaceScan combine_scans: Scans cannot be combined because or varying acquistion parameters (e.g. sampling_rate, npts, time_delay) and/or different scan_data headers.")
         
    if new_obj:
        return combined_scan    
    else:
        scans[0] = combined_scan


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
        
    if x0:
        x_pos = x_pos+x0-x_pos[0]

    if steps:
        x_pos = np.concatenate((x_pos[-steps:],x_pos[:-steps]))
        
    if flip:
        x_pos = x_pos[::-1]
        
    return x_pos


def get_muted_indices(intervals, x_pos):
    '''
    Function which returns an array specifying which traces
    are to be muted. The array contains 1s and 0s and is the
    same length as x_pos, with 0s in the x position intervals
    specified in mute_intervals, and 1s elsewhere.
    
    Arguments:
        --mute_intervals: A list of tuples and/or floats specifying the 
                          **closed** intervals or points over which to mute traces
        --x_pos: The x_position array of the traces
        
    Returns:
        --index_arr: An array of boolean values
    '''

    truth_array = np.zeros((len(intervals),len(x_pos)))
    for i in range(len(intervals)):
        _int = intervals[i]
        if isinstance(_int, (int, float)):
            truth_val = np.where(x_pos==_int, np.zeros(len(x_pos)), 1)
        else:
            truth_val = np.where(x_pos<min(_int), np.zeros(len(x_pos))+1, 0) + \
                             np.where(x_pos>max(_int), np.zeros(len(x_pos))+1, 0)
        truth_array[i] = truth_val

    truth_array = truth_array.sum(axis=0)
    index_arr = np.where(truth_array<len(intervals), np.zeros(len(x_pos)), 1)
    
    return index_arr.astype(bool)

















