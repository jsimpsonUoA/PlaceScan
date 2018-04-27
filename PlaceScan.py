'''
This module is designed to enable interaction with typical laser ultrasound
scans performed in the Physical Acoustics Laboratory at the University of 
Auckland. The main class is PlaceScan, which is an object representing the
scan, including the data and the associated metadata. The user can interact
with the scan by plotting and analysing the data through various functions.

This module is designed to handle the data generated using the PLACE:
Python package for Laboratory Automation, Control, and Experimentation
See https://place.auckland.ac.nz for documentation.

Written by Jonathan Simpson, jsim921@aucklanduni.ac.nz
PAL Lab UoA, March 2018
'''

import os
import glob
import warnings
import numpy as np
import json
import pickle

from scipy.signal import detrend

import organising as org
import plotting as plot
import auto_picker as ap

class PlaceScan():
    
    def __init__(self, directory, scan_type='rotation', trace_field='ATS660-trace'):
        
        if not os.path.isdir(directory):
            raise IOError('Scan directory not found: "{}"'.format(directory))
        
        self.scan_dir, self.scan_type = directory, scan_type
        self.scan_name = directory[directory[:-1].rfind('/')+1:-1]
        
        with open(directory+'config.json', 'r') as file:
            self.config = json.load(file)
        self.npy = np.load(glob.glob(directory+'*.npy')[0])
        
        self.metadata = self.config['metadata']
        self.updates = self.config['updates']
        
        if trace_field:

            self.trace_field = trace_field
            self.trace_data = self.npy[self.trace_field].copy()
            self._true_amps()
            if len(self.trace_data.shape) < 4:
                s = self.trace_data.shape
                self.trace_data = self.trace_data.reshape(list(s[:-1])+[1]*(4-len(s))+[s[-1]])

            sampling_key = next((key for key in self.metadata.keys() if 'sampling_rate' in key),
                               next(key for key in self.metadata.keys() if 'sample_rate' in key))
            self.sampling_rate = self.metadata[sampling_key]
            self.delta = 1.0/self.sampling_rate
            self.npts = self.trace_data.shape[-1]
            self.endtime = (self.npts - 1) * self.delta
            self.time_delay = next((val for (key,val) in self.metadata.items() if 'time_delay' in key), 0.0)
            self.times = np.arange(0.0, self.endtime+self.delta, self.delta)*1e6 - self.time_delay
            
            if scan_type == 'rotation':
                try:
                    self.x_positions = self.npy['ArduinoStage-position']
                    self.stage = 'ArduinoStage-position'
                except KeyError:
                    self.x_positions = self.npy['RotStage-position']
                    self.stage = 'RotStage-position'
            elif scan_type == 'linear':
                try:
                    self.x_positions = self.npy['LongStage-position']
                    self.stage = 'LongStage-position'
                except KeyError:
                    self.x_positions = self.npy['ShortStage-position']
                    self.stage = 'ShortStage-position'
            elif scan_type == 'single':
                self.x_positions = np.arange(0.,self.npy.shape[0])
    
    
    def write(self, save_dir, update_npy=True):
        '''
        Save the scan and config data.
        
        Arguments:
            --save_dir: The scan directory to save to
            --update_npy: True if the current x_positions are to be used in
                          the npy array. Any modifications to trace_data are
                          not saved, except for reordering, muting, etc.
            
        Returns:
            --None
        '''
    
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if update_npy and len(self.x_positions) == self.npy.shape[0]:
            self.npy[self.stage] = self.x_positions
                    
        with open(save_dir+'config.json', 'w') as file:
            json.dump(self.config, file)
        np.save(save_dir+'scan_data.npy', self.npy)    
    
    
    #####################   Data Organising methods ##########################
    
    def combine(self, scans, **kwargs):
        '''
        Combine the PlaceScan with one or more other scans (current scan
        takes highest priority).
        
        Arguments:
            --scans: A list of PlaceScan objects to combine with the current scan
            --**kwargs: The keyword arguments for the combination
            
        Returns:
            -None            
        '''
        
        comb_scan = org.combine_scans([self]+scans, **kwargs)
        return comb_scan
    
    def reposition(self, **kwargs):
        '''
        Options to change the positioning of the traces.
        
        Arguments:
            --**kwargs: The keyword arguments for reposition_traces
            
        Returns:
            --None
        '''
        
        self.x_positions = org.reposition_traces(self.x_positions, **kwargs)
        
        sorting_indices = self.x_positions.argsort()
        self.trace_data = self.trace_data[sorting_indices]
        self.npy[self.trace_field] = self.npy[self.trace_field][sorting_indices]
        self.x_positions.sort()
        
        
    def mute(self, mute_traces):
        '''
        Mute the traces with x positions given in in the list mute_traces.
        The x positions can be given as tuple intervals, or as
        individual positions. Traces are removed from x_positions and trace_data
        
        Arguments:
            --mute_traces: A list of tuples and/or numbers specifying traces
                          to mute
                         
        Returns:
            --None
        '''
        
        mute_indices = org.get_muted_indices(mute_traces, self.x_positions)
        
        self.trace_data = self.trace_data[mute_indices]
        self.x_positions = self.x_positions[mute_indices]
        
        
    #######################   Plotting methods ###############################
        
    def wiggle_plot(self, normed=True, bandpass=None, save_dir=None, 
                    save_ext='', tmax=None, plot_picks=False, **kwargs):
        '''
        Plot a wiggle plot of the trace data from the scan
        
        Arguments:
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --save_dir: The figure directory to save to. If this is specified,
                        the figures will be saved in the scan directory, and a
                        link will be made in the specified directory.
            --save_ext: The extension of the figure filename to describe the plot.
            --tmax: The maximum time to plot for
            --plot_picks: True if wave arrivals are to be plotted from file
            --kwargs: The keyword arguments for the wiggle plot
            
        Returns:
            --fig: The matplotlib figure with the wiggle plot
        '''
        
        local_vars = locals(); del local_vars['self']   
        plot_data, plot_times = self._get_plot_data(tmax, normed, bandpass)
        
        picks_dir = None
        if plot_picks:
            picks_dir = self.scan_dir+'p_wave_picks.csv'
            
        save_dirs = self._get_master_dirs(save_dir, 'wig'+save_ext)
        
        fig = plot.wiggle_plot(plot_data, plot_times, self.x_positions, tmax=tmax, 
                               save_dir=save_dirs, plot_picks_dir=picks_dir, **kwargs)
        
        self._create_fig_links(save_dirs, local_vars, 'wig'+save_ext)
        
        return fig
        

    def variable_density(self, normed=False, bandpass=None, save_dir=None,
                           save_ext='', plot_picks=False, tmax=None, **kwargs):
        '''
        Plot a variable density wiggle plot of the trace data from the scan
        
        Arguments:
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --save_dir: The figure directory to save to. If this is specified,
                        the figures will be saved in the scan directory, and a
                        link will be made in the specified directory.
            --save_ext: The extension of the figure filename to describe the plot.
            --plot_picks: True if wave arrivals are to be plotted from file
            --tmax: The maximum time to plot for
            --kwargs: The keyword arguments for the plot
            
        Returns:
            --fig: The matplotlib figure with the wiggle plot
        '''
        
        local_vars = locals(); del local_vars['self']
        plot_data, plot_times = self._get_plot_data(tmax, normed, bandpass)
            
        picks_dir = None
        if plot_picks:
            picks_dir = self.scan_dir+'p_wave_picks.csv'
            
        save_dirs = self._get_master_dirs(save_dir, 'vd'+save_ext)

        fig = plot.variable_density(plot_data, plot_times, self.x_positions, tmax=tmax,
                                        save_dir=save_dirs, plot_picks_dir=picks_dir, **kwargs)
        
        self._create_fig_links(save_dirs, local_vars, 'vd'+save_ext)
        
        return fig
    

    def trace_plot(self, normed=False, bandpass=None, trace_int=None, 
                   averaging=None, save_dir=None, save_ext='',
                    plot_picks=False, tmax=None, **kwargs):
        '''
        Function to plot traces as a time series on the same axis

        Arguments:
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --trace_int: Plot every nth trace where trace_int=n
            --averaging: Plot averages of every n traces where averaging=n
            --save_dir: The figure directory to save to. If this is specified,
                        the figures will be saved in the scan directory, and a
                        link will be made in the specified directory.
            --save_ext: The extension of the figure filename to describe the plot.
            --plot_picks: True if wave arrivals are to be plotted from file
            --tmax: The maximum time to plot for
            --kwargs: The keyword arguments for the trace plot

        Returns:
            --fig: The matplotlib figure with the plot
        '''
        
        local_vars = locals(); del local_vars['self']
        plot_data, plot_times = self._get_plot_data(tmax, normed, bandpass)
        
        if trace_int:
            plot_data = plot_data[::trace_int]
        elif averaging:
            x = max(len(plot_data) // averaging, 1)
            y = min(averaging, len(plot_data))
            z = len(plot_data[0])
            plot_data = plot_data.flatten()[:x*y*z].reshape((x,y,z))
            plot_data = np.sum(plot_data, axis=1) / y

        picks_dir = None
        if plot_picks:
            picks_dir = self.scan_dir+'p_wave_picks.csv'
            
        save_dirs = self._get_master_dirs(save_dir, save_ext)

        fig = plot.all_traces(plot_data, plot_times, ylabel=self.amp_label, tmax=tmax,
                                  save_dir=save_dirs, plot_picks_dir=picks_dir, **kwargs)
        
        self._create_fig_links(save_dirs, local_vars, save_ext)
        
        return fig
        
    
    def arrival_picks_plot(self, picks_dirs=None, save_dir=None, save_ext='', **kwargs):
        '''
        Plot a variable density wiggle plot of the trace data from the scan
        
        Arguments:
            --picks_dirs: The scan directories to plot the picks for
            --save_dir: The figure directory to save to. If this is specified,
                        the figures will be saved in the scan directory, and a
                        link will be made in the specified directory.
            --save_ext: The extension of the figure filename to describe the plot.
            --kwargs: The keyword arguments for the plot
            
        Returns:
            --fig: The matplotlib figure with the wiggle plot
        '''
        
        local_vars = locals(); del local_vars['self']
        
        if picks_dirs:
            picks_dirs = [picks_dir+'p_wave_picks.csv' for picks_dir in picks_dirs]   
        else:
            picks_dirs = self.scan_dir+'p_wave_picks.csv'
        save_dirs = self._get_master_dirs(save_dir, 'ar'+save_ext, common_sample=True)

        fig = plot.arrival_times_plot(picks_dirs, save_dir=save_dirs, **kwargs)
        
        self._create_fig_links(save_dirs, local_vars, 'ar'+save_ext, common_sample=True)
        
        return fig
        
        
        
    ########################  Picking methods  ###############################

    def pick_arrivals(self, bandpass=None, **kwargs):
        '''
        Function to easily pick wave arrivals on a wiggle plot of a PlaceScan
        Press 'q' to record the position of the cursor, 'w' to skip the trace,
        and 'Delete' to undo the last pick.
        
        Arguments:
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --kwargs: The keyword arguments for the auto picking
            
        Returns:
            None
        '''

        plot_data = self.trace_data[:,1,0].copy()
        
        if bandpass:
            plot_data = bandpass_filter(plot_data, bandpass[0], bandpass[1], self.sampling_rate)        
        plot_data = normalize(plot_data)
        
        ap.auto_picker(plot_data, self.times, self.x_positions, **kwargs)

  
    ########################  Private methods  ###############################

    def _true_amps(self):
        '''
        Function to transform the data to absolute amplitudes
        and generate the amplitude axis label. Note: this
        function is highly sensitive to the format of the config.json
        file generated during a PLACE scan, and assumes the use
        of an ATS* AlazarTech scope card and Polytec Vibrometer module.

        Arguments:
            --None
 
        Return:
            --None
        '''
        
        # Detrending
        self.trace_data = detrend(self.trace_data)
        
        try:
            scope_bits = self.metadata['bits_per_sample']
            scope_config = next(module['config'] for module in self.config['modules'] if 'ATS' in module['class_name'])
        
            # Assume that the trace input on the scope was the second channel
            scope_dynmc_range = ''.join(list(filter(str.isdigit, scope_config['analog_inputs'][1]['input_range'])))
            scope_dynmc_range = int(scope_dynmc_range) * 2
            if scope_dynmc_range > 50:   #Change from mV to V. Conditional is dependent on scope options.
                scope_dynmc_range = scope_dynmc_range / 1000.0     
    
            # Change the units to volts
            self.trace_data = self.trace_data / (2 ** scope_bits) * scope_dynmc_range
        except:
            print('PlaceScan _true_amps: Unable to calibrate oscilloscope amplitudes')        
        
        try:
            # Calibrate to the vibrometer units
            vib_calib = next(val for (key,val) in self.metadata.items() if 'calibration' in key and isinstance(val, type(1.0)))
            vib_calib_units = next(val for (key,val) in self.metadata.items() if 'calibration_units' in key)[:-3]    
        
            self.trace_data = self.trace_data * vib_calib
           
            self.amp_label = "Amplitude ({})".format(vib_calib_units)
        except:
            self.amp_label = "Amplitude (V)"
            print('PlaceScan _true_amps: Unable to calibrate vibrometer amplitudes') 
            

   
    def _get_master_dirs(self, save_dir, key, common_sample=False):
        '''
        Function to get the filename of figures to save them in the 
        figure directory.
        
        Arguments:
            --save_dir: The save_dir arguments specified in the plotting
                        method
            --key: The key for the type of plot
            --common_sample: If the figure is not scan-specific, set to True
            
        Returns:
            --master_save_dirs: The directories to save the figures to
        
        '''
    
        if save_dir:
            if not common_sample:
                fig_name = self.scan_name+'_'+key
            else:
                fig_name = self.scan_name[:self.scan_name.find('_')]+'_'+key
            master_save_dirs = [save_dir+fig_name+'.png', save_dir+fig_name+'.pdf']
            return master_save_dirs
        else:
            return
    
    
    def _create_fig_links(self, save_dirs, local_vars, key, common_sample=False):
        '''
        Function to establish links between the saved figures in the figure 
        directory and the scan directory self.scan_dir. The arguments and keyword
        arguments for the function call are also pickled in the scan_dir.
        
        Arguments:
            --save_dirs: The directories to which the figures have already
                         been saved
            --common_sample: If the figure is not scan-specific, set to True
            
        Returns:
            --None       
        '''
        
        if save_dirs:
            
            prefix_dir = self.scan_dir
            if common_sample:
                prefix_dir = prefix_dir[:prefix_dir[:-1].rfind('/')+1]
            
            for _dir in save_dirs:
                link = prefix_dir+_dir[_dir.rfind('/')+1:]
                if not os.path.exists(link):
                    os.symlink(_dir, link)
            
            with open(prefix_dir+'/'+self.scan_name+'_'+key+'.p', 'wb') as file:
                pickle.dump(local_vars, file, -1)
    
    
    def _get_plot_data(self, tmax, normed=False, bandpass=None):
        '''
        Function to slice the data for plotting
        
        Arguments:
            --tmax: The maximum time to plot for
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            
        Returns:
            --data: The plotting values
            --times: The plotting times
        '''
    
        if tmax:
            plot_times = self.times[np.where(self.times <= tmax)].copy()
            plot_data = self.trace_data[:,1,0][:,:len(plot_times)].copy()
        else:
            plot_times = self.times.copy()
            plot_data = self.trace_data[:,1,0].copy()            

        if bandpass:
            plot_data = bandpass_filter(plot_data, bandpass[0], bandpass[1], self.sampling_rate)
        if normed:
            plot_data = normalize(plot_data)
            
        return plot_data, plot_times
    
    
    
def normalize(data):
    '''
    Function to normalise the data
    
    Arguments:
        --None
        
    Returns:
        --data: The normalized data
    '''
    
    return np.array([array/np.amax(array) for array in data])
        
        
def bandpass_filter(data, min_freq, max_freq, sampling_rate):
    '''
    Apply a bandpass filter to data. Borrowed from obspy.signal.filter
    
    Arguments:
        --data: the data to be filtered
        --min_freq: The lower corner frequency of the bandpass filter
        --max_freq: The upper corner frequency of the bandpass filter
        --sampling_rate: The sampling rate of the data
        
    Returns:
        --data: The filtered data
    '''
    
    from scipy.signal import iirfilter, zpk2sos, sosfiltfilt
    
    fe = 0.5 * sampling_rate
    low = min_freq / fe
    high = max_freq / fe
    
    # Raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). No filter applied.").format(max_freq, fe)
        warnings.warn(msg)
        return data
    
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
        
    z, p, k = iirfilter(4, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    firstpass = sosfiltfilt(sos, data)
    
    return firstpass#sosfilt(sos, firstpass[::-1])[::-1]
        



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        









        
    
