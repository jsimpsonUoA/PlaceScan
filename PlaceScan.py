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
import picking as pk
#import dtw_main as dtw
import analysis

class PlaceScan():
    
    def __init__(self, directory, scan_type='rotation', trace_field='ATS660-trace',
                 apply_formatting=True, divide_energy=False):
        
        if not os.path.isdir(directory):
            raise IOError('Scan directory not found: "{}"'.format(directory))
        
        self.scan_dir, self.scan_type = directory, scan_type
        self.scan_name = directory[directory[:-1].rfind('/')+1:-1]
        self.sample = self.scan_name[:self.scan_name.find('_')]
        
        with open(directory+'config.json', 'r') as file:
            self.config = json.load(file)
        self.npy = np.load(glob.glob(directory+'*.npy')[0])

        self.metadata = self.config['metadata']
        self.updates = self.config['updates']
        self.timestamps = np.asarray(self.npy['PLACE-time'],dtype=float)
        
        self.place_version = float(self.metadata['PLACE_version'][:self.metadata['PLACE_version'].rfind('.')])
    
        if self.place_version < 0.7:
            self.plugins_key = 'modules'  
        else:
            self.plugins_key = 'plugins'                 
            
        if trace_field:

            self.trace_field = trace_field
            self.trace_data = self.npy[self.trace_field].copy().astype(np.float64)
            
            if len(self.trace_data.shape) < 4:
                s = self.trace_data.shape
                self.trace_data = self.trace_data.reshape(list(s[:-1])+[1]*(4-len(s))+[s[-1]])
            
            sampling_key = next((key for key in self.metadata.keys() if 'sampling_rate' in key), None)
            if not sampling_key:
                sampling_key = next(key for key in self.metadata.keys() if 'sample_rate' in key)
            self.sampling_rate = self.metadata[sampling_key]
            self.delta = 1.0/self.sampling_rate
            self.npts = self.trace_data.shape[-1]
            self.endtime = (self.npts - 1) * self.delta
            self.time_delay = next((val for (key,val) in self.metadata.items() if 'time_delay' in key), 0.0)

            '''
            if ('preamp' in self.config['comments'].lower()) or ('pre-amp' in self.config['comments'].lower()):
                print('Correcting times for Pre-amp')
                self.time_delay += 0.7                #Be careful with this!
            '''
                
            self.times = np.arange(0.0, self.endtime+self.delta, self.delta)*1e6 - self.time_delay
            self._true_amps()
            self._get_energy_from_comments(divide_energy)
            
            if scan_type == 'rotation':
                try:
                    self.x_positions = self.npy['ArduinoStage-position']
                    self.stage = 'ArduinoStage-position'
                except ValueError:
                    self.x_positions = self.npy['RotStage-position']
                    self.stage = 'RotStage-position'
                self.data_index = 1
            elif scan_type == 'linear':
                try:
                    self.x_positions = self.npy['LongStage-position']
                    self.stage = 'LongStage-position'
                except ValueError:
                    self.x_positions = self.npy['ShortStage-position']
                    self.stage = 'ShortStage-position'
            elif scan_type == 'single':
                self.x_positions = np.arange(0.,self.npy.shape[0])
                self.data_index = 1   #this was a quick fix, but may be useful permanently. Not always 0 though if scan_type=='single'
   
        #Open the formatting dictionary for the scan and apply formatting.
        if os.path.isfile(directory+'formatting.json'):
            with open(directory+'formatting.json', 'r') as file:
                self.formatting = json.load(file)
            if apply_formatting:
                self.apply_formatting()
        else:
            self.formatting = {}
    
    
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
        
        self.update_formatting('reposition', kwargs)
        
    def mute(self, mute_traces, keep=False, zero=True):
        '''
        Mute the traces with x positions given in in the list mute_traces.
        The x positions can be given as tuple intervals, or as
        individual positions. Traces are removed from x_positions and trace_data
        
        Arguments:
            --mute_traces: A list of tuples and/or numbers specifying traces
                          to mute
            --keep: True to invert the muting. Instead of muting the intervals given,
                    this mutes everything not in the interval.
            --zero: Makes each muted trace a trace of zeros. If False, there will
                     not be any data at that x_position
                         
        Returns:
            --None
        '''
        
        mute_indices = org.get_muted_indices(mute_traces, self.x_positions, keep=keep)
        self.muted_traces = np.where(mute_indices==False)

        if not zero:
            self.trace_data = self.trace_data[tuple(mute_indices)]
            self.x_positions = self.x_positions[tuple(mute_indices)]
        else:
            for ind in self.muted_traces:
                self.trace_data[ind] = self.trace_data[ind]*0.0
        
        self.update_formatting('mute', {'args':[mute_traces],'keep':keep,'zero':zero})
        
        
    def expand(self, **kwargs):
        '''
        Function to take make every record saved in an update
        into an individual update with a unique x_position. 
        This is designed for a scan where the sample is moving during
        the update.
        
        Arguments:
            --kwargs: The keyword arguments
                    
        Returns:
            --None
        '''
        
        org.expand_updates(self, **kwargs)
        
        self.update_formatting('expand', kwargs)
        
        
    def mute_by_signal(self, threshold, analog=False, max_analog_sig=1.033,
                       sig_plot=False):
        '''
        Mute traces in a scan based on the signal level of the vibrometer.
        The signal level may be stored in an individual column of the npy
        file (e.g. 'Polytec-signal'), which will be sorting by a digital
        value. Alternatively, the signal level may be stored as analog input
        on a third oscilliscope channel in the npy array. In this case, the 
        first analog signal value of the update is used as the criteria.
        
        Arguments:
            --threshold: The threshold for muting the traces.
            --analog: True if the signal levels are contained in a third
                      oscilliscope channel
            --max_analog_sig: The voltage which corresponds to 100% signal
            --sig_plot: True to plot the signal levels
            
       Returns:
           None
        '''
        
        if not analog:
            
            signal_levels = self.npy['Polytec-signal'] / 512 * 100.0
        
        else:
            
            dymn_range = self._get_scope_dynm_range(2) / 2.0
            new_shape = (self.trace_data.shape[0]*self.trace_data.shape[2], self.trace_data.shape[-1])
            
            signals = self.trace_data[:,2,:,:].reshape(new_shape)
            signal_levels = np.array([signal[0] for signal in signals])     
            signal_levels = 100.0*((signal_levels-dymn_range)/max_analog_sig)
        
        if sig_plot:
            plot.simple_plot_points(range(0,len(signal_levels)), signal_levels,
                                    xlab='Update Number', ylab='Signal Level (%)')    
        
        mute_indices = np.where(signal_levels<threshold,0,1).astype(bool)
        self.muted_traces = mute_indices

        if self.trace_data.shape[0] == len(signal_levels):
            self.trace_data = self.trace_data[tuple(mute_indices)]
        else:   #For single update
            mute_indices.reshape((self.trace_data.shape[0],self.trace_data.shape[2]))
            print('Work in progress:',mute_indices.shape, self.trace_data.shape)
        self.x_positions = self.x_positions[tuple(mute_indices)]    
        
        print('Mute by signal: Keeping {} out of {} traces.'.format(len(self.trace_data),self.updates))
        
        self.update_formatting('mute_by_signal', {'args':[threshold],'analog':analog,
                                       'max_analog_sig':max_analog_sig,'sg_plot':sig_plot})
        
        
    def change_polarity(self, positions=None):
        '''
        Change the polarity of the scan data by multiplying by -1.
        If no positions are specified, the entire scan is
        altered.
        
        Arguments:
            --positions: A list of tuples of x_positon ranges and/or individual
                         x_positions specifying traces to reverse polarity for
            
       Returns:
           None       
        '''
        
        if not positions:
            self.trace_data = self.trace_data * -1.
        else:
            flip_indices = org.get_muted_indices(positions, self.x_positions, keep=True)
            for i in range(len(flip_indices)):
                if flip_indices[i]:
                    self.trace_data[i] = self.trace_data[i] * -1.
                    
        self.update_formatting('change_polarity', {'positions':positions})
                                   

    def average_traces(self, number_of_traces=1):
        '''
        Function to average the traces in a scan by taking
        the summed average of traces including and either side
        of each trace. The number of traces either side can be
        specified

        Arguments:
        number_of_traces: The number of traces each side of a trace
                  to be used in the averaging. Ususally 1.

        Returns:
            None
        '''

        data = self.trace_data[:,1,:].copy()  
        s = data.shape
        data = data.reshape(s[0]*s[1],s[2])

        data = org.trace_averaging(data, num_either_side=number_of_traces)
        
        data.reshape(s)
        if len(data.shape) < len(s):
            new_data = []
            for i, row in enumerate(data):
                new_data.append(np.array([row]))
            data = np.array(new_data)
        self.trace_data[:,1] = data


    def apply_formatting(self):
        '''
        Function which formats a scan with the organising methods
        using the parameters stored in self.formatting. This can
        be performed by specifying apply_formatting=True when
        initialising a PlaceScan object for convenient organising.
        
        Arguments:
            N/A
            
        Returns:
            --None
        '''
        
        for function,kwargs in self.formatting.items():
            if 'args' in kwargs:
                args = kwargs.pop('args')
                if args:
                    eval(function)(self,*args,**kwargs)
                else:
                    eval(function)(self,**kwargs)
            
        
    def update_formatting(self, key, kwargs):
        '''
        Function which updates and saves the fomratting dictionary for
        the scan each time an organising function is used.
        
        Arguments:
            --key: The name of the organising function
            --kwargs: The args (as 'args':[list_of_args]) and kwargs
                      in a dictionary
                      
        Returns:
            None
        '''
        
        self.formatting['PlaceScan.'+key] = kwargs
        
        with open(self.scan_dir+'formatting.json', 'w') as f:
            json.dump(self.formatting, f)
        
        
        
    #######################   Plotting methods ###############################
        
    def wiggle_plot(self, normed=True, bandpass=None, dc_corr_seconds=None,
                    save_dir=None, save_ext='', tmax=None, tmin=0.0, 
                    plot_picks=False, decimate=False, **kwargs):
        '''
        Plot a wiggle plot of the trace data from the scan
        
        Arguments:
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --dc_corr_seconds: The number of seconds at the beginning of the trace
                        to calcualte a DC shift correction from.
            --save_dir: The figure directory to save to. If this is specified,
                        the figures will be saved in the scan directory, and a
                        link will be made in the specified directory.
            --save_ext: The extension of the figure filename to describe the plot.
            --tmax: The maximum time to plot for
            --tmin: The minimum time to plot for
            --plot_picks: True if wave arrivals are to be plotted from file
            --decimate: true, or integer factor to downsample data by
            --kwargs: The keyword arguments for the wiggle plot
            
        Returns:
            --fig: The matplotlib figure with the wiggle plot
        '''
        
        local_vars = locals(); del local_vars['self']   
        plot_data, plot_times = self._get_plot_data(tmax, tmin, normed, bandpass, dc_corr_seconds, decimate)
        
        picks_dir = self._get_picks_dir(plot_picks)
        
        save_dirs = self._get_master_dirs(save_dir, 'wig'+save_ext)
        
        fig = plot.wiggle_plot(plot_data, plot_times, self.x_positions, tmax=tmax, 
                               save_dir=save_dirs, plot_picks_dir=picks_dir, **kwargs)
        
        self._create_fig_links(save_dirs, local_vars, 'wig'+save_ext)
        
        return fig
        

    def variable_density(self, normed=False, bandpass=None, dc_corr_seconds=None, 
                           save_dir=None, save_ext='', plot_picks=False, 
                           tmax=None, **kwargs):
        '''                
        Plot a variable density wiggle plot of the trace data from the scan
        
        Arguments:
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --dc_corr_seconds: The number of seconds at the beginning of the trace
                        to calcualte a DC shift correction from.
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
        plot_data, plot_times = self._get_plot_data(tmax, None, normed, bandpass, dc_corr_seconds)
            
        picks_dir = self._get_picks_dir(plot_picks)
            
        save_dirs = self._get_master_dirs(save_dir, 'vd'+save_ext)

        fig = plot.variable_density(plot_data, plot_times, self.x_positions, tmax=tmax,
                                        save_dir=save_dirs, plot_picks_dir=picks_dir, **kwargs)
        
        self._create_fig_links(save_dirs, local_vars, 'vd'+save_ext)
        
        return fig
    

    def trace_plot(self, normed=False, bandpass=None, dc_corr_seconds=None, 
                    differentiate=False, trace_int=None, averaging=None, 
                    position=None, trace_index=None, save_dir=None, save_ext='', 
                    plot_picks=False, tmin=0.0, tmax=None, window_around_p=None, 
                    picks_offset=None, **kwargs):
        '''
        Function to plot traces as a time series on the same axis

        Arguments:
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --dc_corr_seconds: The number of seconds at the beginning of the trace
                to calcualte a DC shift correction from.
            --differentiate: An integer specifying how many times the signal should
                be differentiated. 
            --trace_int: Plot every nth trace where trace_int=n
            --averaging: Plot averages of every n traces where averaging=n
            --position: The x_position of the trace for comparison
            --trace_index: If position is not provided, the index of the trace 
                in trace_data to plot for.
            --save_dir: The figure directory to save to. If this is specified,
                the figures will be saved in the scan directory, and a
                link will be made in the specified directory.
            --save_ext: The extension of the figure filename to describe the plot.
            --plot_picks: True if wave arrivals are to be plotted from file
            --tmin: The minimum time to plot for.
            --tmax: The maximum time to plot for
            --kwargs: The keyword arguments for the trace plot

        Returns:
            --fig: The matplotlib figure with the plot
        '''
        
        local_vars = locals(); del local_vars['self']
        plot_data, plot_times = self._get_plot_data(tmax, tmin, normed, bandpass, dc_corr_seconds, differentiate=differentiate)

        if trace_int != None:
            plot_data = plot_data[::trace_int]
        elif averaging != None:
            x = max(len(plot_data) // averaging, 1)
            y = min(averaging, len(plot_data))
            z = len(plot_data[0])
            plot_data = plot_data.flatten()[:x*y*z].reshape((x,y,z))
            plot_data = np.sum(plot_data, axis=1) / y
        elif position != None:
            plot_data = [plot_data[np.argmin(np.abs(self.x_positions-position))]]
        elif trace_index != None:
            plot_data = [plot_data[trace_index]]
            position = self.x_positions[trace_index]
        
        if window_around_p:
            plot_data, plot_times = self._get_time_window_around_pick(plot_data[0], plot_times, 
                        self.scan_dir+'p_wave_picks.csv', picks_offset, position, window_around_p)
            plot_data = [plot_data]
            
        picks_dir = None
        if plot_picks:
            picks_dir = self.scan_dir+'p_wave_picks.csv'
            
        save_dirs = self._get_master_dirs(save_dir, save_ext)
        ylab = self.amp_label
        if normed:
            ylab = 'Amplitude (a.u.)'

        fig = plot.all_traces(plot_data, plot_times, ylab=ylab, tmax=tmax,
                                  save_dir=save_dirs, plot_picks_dir=picks_dir, 
                                  position=position, picks_offset=picks_offset, **kwargs)
        
        self._create_fig_links(save_dirs, local_vars, save_ext)

        return fig
        
    
    def arrival_picks_plot(self, picks_dirs=None, scans=None, save_dir=None, save_ext='',
                           pick_type='p', **kwargs):
        '''
        Plot the arrival picks of a scan on a Cartesian or polar plot.
        
        Arguments:
            --picks_dirs: The scan directories to plot the picks for
            --scans: A list of PlaceScan objects to plot the picks for.
            --save_dir: The figure directory to save to. If this is specified,
                        the figures will be saved in the scan directory, and a
                        link will be made in the specified directory.
            --save_ext: The extension of the figure filename to describe the plot.
            --pick_type: The type/name of teh arrival to plot
            --kwargs: The keyword arguments for the plot
            
        Returns:
            --fig: The matplotlib figure with the wiggle plot
        '''
        
        local_vars = locals(); del local_vars['self']; del local_vars['scans']
        
        if scans:
            picks_dirs = [scan.scan_dir+scan.scan_name+'_{}-picks.csv'.format(pick_type) for scan in scans]   
        else:
            picks_dirs = self.scan_dir+self.scan_name+'_{}-picks.csv'.format(pick_type)
        save_dirs = self._get_master_dirs(save_dir, 'ar'+save_ext, common_sample=True)

        fig = plot.arrival_times_plot(picks_dirs, save_dir=save_dirs, **kwargs)
        
        self._create_fig_links(save_dirs, local_vars, 'ar'+save_ext, common_sample=True)
        
        return fig
        
    
    def wigner_spectrogram(self, normed=False, bandpass=None, dc_corr_seconds=None,
                        trace_index=0, position=None, average=False, save_dir=None,
                         save_ext='', tmax=None, max_freq=None, **kwargs):
        '''
        Function to plot a Wigner-Ville multitaper spectrogram
        of the scan time series data.
        
        Arguments:
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --dc_corr_seconds: The number of seconds at the beginning of the trace
                        to calcualte a DC shift correction from.
            --trace_index: If average==False, the index of the trace in trace_data
                         to plot for
            --position: The x_position of the trace for comparison
            --average: Plot the spectrogram for the full-stack average of all
                       traces in tace_data
            --save_dir: The figure directory to save to. If this is specified,
                        the figures will be saved in the scan directory, and a
                        link will be made in the specified directory.
            --save_ext: The extension of the figure filename to describe the plot.
            --tmax: The maximum time to plot for
            --max_freq: The maximum frequency to calculate the spectrogram for.
            --kwargs: The keyword arguments for the Wigner-Ville spectrogram and
                      plotting
            
        Returns:
            --fig: The figure containing the spectrogram plot
            --wv: The values for the Wigner-Ville spectrogram
        '''
        
        local_vars = locals(); del local_vars['self']
        plot_data, plot_times = self._get_plot_data(tmax, 0.0, normed, bandpass, dc_corr_seconds)

        if average:
            plot_data = np.sum(plot_data, axis=0) / len(plot_data)
        else:
            if position != None:
                plot_data = plot_data[np.argmin(np.abs(self.x_positions-position))]
            elif trace_index != None:
                plot_data = plot_data[trace_index]

            
        if not max_freq and bandpass:
            max_freq = bandpass[-1]
            
        save_dirs = self._get_master_dirs(save_dir, save_ext)
        
        fig, wv = plot.wv_spect(plot_data, plot_times, trace_ylab=self.amp_label, 
                                    sampling_rate=self.sampling_rate, 
                                    max_freq=max_freq, save_dir=save_dirs, **kwargs)
        
        self._create_fig_links(save_dirs, local_vars, save_ext)

        return fig, wv
    
    
    def multitaper_spectrum(self, normed=False, bandpass=None, dc_corr_seconds=None, 
                   position=None, trace_index=None, save_dir=None, save_ext='', 
                   plot_picks=False, tmin=0.0, tmax=None, fig=None,
                   common_sample=False, window_around_p=None, picks_offset=None,
                    **kwargs):
        '''
        Calculate and plot the power spectral density for the given position
        or trace index. A multitaper spectral estimate is calculated.
        '''
        
        local_vars = locals(); del local_vars['self']
        plot_data, plot_times = self._get_plot_data(tmax, tmin, normed, bandpass=None, dc_corr_seconds=dc_corr_seconds)
        
        if position != None:
            plot_data = plot_data[np.argmin(np.abs(self.x_positions-position))]
        elif trace_index != None:
            plot_data = plot_data[trace_index]
            position = self.x_positions[trace_index]
        else:
            plot_data = np.sum(plot_data, axis=0) / plot_data.shape[0]  #average all the data

        if window_around_p:
            plot_data, plot_times = self._get_time_window_around_pick(plot_data, plot_times, 
                        self.scan_dir+'p_wave_picks.csv', picks_offset, position, window_around_p)
        
        if bandpass:
            plot_data = bandpass_filter(plot_data, bandpass[0], bandpass[1], self.sampling_rate)
        
        picks_dir = None
        if plot_picks:
            picks_dir = self.scan_dir+'p_wave_picks.csv'
            
        save_dirs = self._get_master_dirs(save_dir, 'spect'+save_ext, common_sample=common_sample)

        fig = plot.multitaper_spect(plot_data, plot_times, self.sampling_rate, 
                                    ylab=self.amp_label, tmax=tmax, save_dir=save_dirs,
                                    plot_picks_dir=picks_dir, fig=fig, **kwargs)
       
        self._create_fig_links(save_dirs, local_vars, 'spect'+save_ext, common_sample=common_sample)
        
        return fig        
        
    
    def animated_comparison(self, plot_type='wiggle', scans=None, save_dir=None, 
                            save_ext='',  file_type='.gif', **kwargs):
        '''
        Create an animated GIF of plots from different scans for easy
        comparison.
        
        Arguments:
            --plot_type: The type of plot to compare. One of 'wiggle', 
                    'variable_density', 'wigner', 'trace_plot', or 'trace_comparison'.
            --scans: A list of PlaceScan objects to plot the comparsion for
            --save_dir: The figure directory to save to. If this is specified,
                        the figures will be saved in the scan directory, and a
                        link will be made in the specified directory.
            --save_ext: The extension of the figure filename to describe the plot.
            --kwargs: The keyword arguments for the animation
            
        Returns:
            --fig: The figure with the animation
        '''
        
        local_vars = locals().copy(); del local_vars['self']; del local_vars['scans']
        
        if not isinstance(scans, list):
            print('PlaceScan Animated Comparison: Please provide a list of PlaceScan\
                  objects to plot the comparison for.')
            return
        elif self not in scans:
            scans = [self]+scans

        if plot_type == 'wiggle':
            plotting_functions = [scan.wiggle_plot for scan in scans]
        elif plot_type == 'variable_density':
            plotting_functions = [scan.variable_density for scan in scans]
        elif plot_type == 'wigner':
            plotting_functions = [scan.wigner_spectrogram for scan in scans]
        elif plot_type == 'trace_comparison':
            plotting_functions = [scan.trace_comparison for scan in scans]
        elif plot_type == 'trace_plot':
            plotting_functions = [scan.trace_plot for scan in scans]
        else:
            print("PlaceScan Animated Comparison: Please provide a valid plot type:\
                  One of 'wiggle', 'variable_density', 'wigner', 'trace_plot', or 'trace_comparison'.")            
            
        save_dirs = self._get_master_dirs(save_dir, save_ext, common_sample=True,
                                          file_types=[file_type])
        if save_dirs:
            save_dirs = save_dirs[0]
        
        fig = plot.animate_plots(plotting_functions, save_dir=save_dirs, **kwargs)
        
        
        #[fdict.pop('scans') for fdict in local_vars['kwargs']['func_kwargs'] 
        #                               if 'func_kwargs' in local_vars['kwargs']]
        self._create_fig_links(save_dirs, local_vars, save_ext, common_sample=True)

        return fig
        

    def trace_comparison(self, normed=False, bandpass=None, dc_corr_seconds=None,
                        scans=None, position=None, trace_index=None, save_dir=None,
                        average=False, save_ext='', tmin=0.0, tmax=None, 
                        plot_picks=False, labels=None, show=True, fig=None,
                        inset_params=None, **kwargs):
        '''
        Create an animated GIF of plots from different scans for easy
        comparison.
        
        Arguments:
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --dc_corr_seconds: The number of seconds at the beginning of the trace
                        to calcualte a DC shift correction from.
            --scans: A list of PlaceScan objects to plot the comparsion for
            --position: The x_position of the trace for comparison, or a list of positions
                         for each scan
            --trace_index: If position is not provided, the index of the trace 
                         in trace_data to plot for, or a list of trace indices.
            --save_dir: The figure directory to save to. If this is specified,
                        the figures will be saved in the scan directory, and a
                        link will be made in the specified directory.
            --average: Plot the average of all traces for each scan.
            --save_ext: The extension of the figure filename to describe the plot.
            --tmin: The minimum time to plot for
            --tmax: The maximum time to plot for
            --plot_picks: True if wave arrivals are to be plotted from file
            --labels: The labels for each trace.
            --show: True to show the plot
            --fig: A figure to plot onto
            --inset_params: A dictioanry containing parameters for a zoomed inset
            --kwargs: The keyword arguments for the Wigner-Ville spectrogram and
                      plotting
            
        Returns:
            --fig: The figure with the animation
        '''
        
        local_vars = locals().copy(); del local_vars['self']; del local_vars['scans']
        
        num = len(scans)
        if not isinstance(scans, list):
            print('PlaceScan Trace Comparison: Please provide a list of PlaceScan\
                  objects to plot the comparison for.')
            return
        elif self not in scans:
            scans = [self]+scans
        
        averaging = [None]*len(scans)
        if position != None:
            if isinstance(position, list):
                trace_inds = [np.argmin(np.abs(scans[i].x_positions-position[i])) for i in range(len(scans))]
            else:
                trace_inds = [np.argmin(np.abs(scan.x_positions-position)) for scan in scans]
            print("PlaceScan Trace Comparison: Plotting comparsion at x-position {}."
                    .format(round(scans[0].x_positions[trace_inds[0]],5)))
        elif trace_index != None:
            if isinstance(trace_index, list):
                trace_inds = trace_index
            else:
                trace_inds = [trace_index]*num
            print("PlaceScan Trace Comparison: Plotting comparsion at x-position {}."
                    .format(round(scans[0].x_positions[trace_inds[0]],5)))
        elif average:
            averaging = [len(scan.trace_data) for scan in scans]
            trace_inds = [None]*len(scans)
        else:
            print("PlaceScan Trace Comparison: Please provide a valid position\
                  or trace index to plot.")    
            return
        
        save_dirs = self._get_master_dirs(save_dir, save_ext, common_sample=True)
        
        legend = True
        if labels and len(labels) < num:
            print("PlaceScan Trace Comparison: Please provide a label for each trace.")
            return
        elif not labels:
            labels = [None]*num
            legend = False
            
        colors = ['c','g','r','y','b','m']    
        linestyles = ['-', '--', '-.', ':']
        for i in range(len(scans)-1):            
            fig = scans[i].trace_plot(normed=normed, bandpass=bandpass, 
                           dc_corr_seconds=dc_corr_seconds, trace_index=trace_inds[i], 
                           save_dir=None, plot_picks=plot_picks, tmin=tmin, tmax=tmax,
                           label=labels[i], fig=fig, show=False, linestyle=linestyles[i%4], 
                           legend=False, averaging=averaging[i],**kwargs)  
            
        fig = scans[-1].trace_plot(normed=normed, bandpass=bandpass, 
                       dc_corr_seconds=dc_corr_seconds, trace_index=trace_inds[-1], 
                       save_dir=None, plot_picks=plot_picks, tmin=tmin, tmax=tmax,
                       label=labels[i+1], fig=fig, show=show, linestyle=linestyles[(len(scans)-1)%4], 
                       legend=legend, averaging=averaging[i], inset_params=inset_params, **kwargs)       
        
        if save_dirs:
            print('saving')
            if isinstance(save_dirs, list):
                for _dir in save_dirs:
                    fig.savefig(_dir, bbox_inches='tight')
            else:
                fig.savefig(save_dirs, bbox_inches='tight')    
        
        self._create_fig_links(save_dirs, local_vars, save_ext, common_sample=True)

        return fig        
    
    
    def cross_correlation(self, scans=None, bandpass=None, dc_corr_seconds=None, 
                   position=None, trace_index=None, save_dir=None, save_ext='', 
                   tmin=0.0, tmax=None, fig=None, common_sample=False, **kwargs):
        '''
        Calculate and plot the cross correlation function for two traces. Either 
        two traces within a scan can be used, or a lsit of scans can be given to
        cross-correlate the traces at a given position. If no position or trace
        index is given, the average of all traces in the scan is used.
        
        Arguments:
            --scans: None if using two traces from the same scan, or a list of
                     scans for comparing the traces between scans.
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --dc_corr_seconds: The number of seconds at the beginning of the trace
                        to calcualte a DC shift correction from.
            --position: The x_position of the traces for correlation, or a list of positions
                        within a scan to correlate.
            --trace_index: The trace index of the traces for correlation, or a list of indices
                        within a scan to correlate.
            --save_dir: The figure directory to save to. If this is specified,
                        the figures will be saved in the scan directory, and a
                        link will be made in the specified directory.
            --save_ext: The extension of the figure filename to describe the plot.
            --tmin: The minimum time to perform the correlation for
            --tmax: The maximum time to perform the correlation for
            --fig: A figure to plot onto
            --kwargs: The keyword arguments for the plotting
            
        Returns:
            --fig: The figure with the corss-correlation plot
            --max_lag: The lag time where the maximum correlation coefficient occurs.
        '''
        
        local_vars = locals(); del local_vars['self']
        
        plot_data, plot_times = self._get_plot_data(tmax, tmin, False, bandpass=bandpass, dc_corr_seconds=dc_corr_seconds)
        if scans == None:
            try:
                if position != None:
                    plot_data = np.array([plot_data[np.argmin(np.abs(self.x_positions-pos))] for pos in position])
                else:
                    plot_data = np.array([plot_data[ind] for ind in trace_index])
            except TypeError:
                print('PlaceScan cross_correlation: Please provide a list of two or more\
                      numbers for position or trace_index')
        else:
            common_sample = True
            if self not in scans:
                scans = [self]+scans
            plot_data = [scan._get_plot_data(tmax, tmin, False, bandpass=bandpass, dc_corr_seconds=dc_corr_seconds)[0] for scan in scans]
            if position != None:
                plot_data = np.array([data[np.argmin(np.abs(scan.x_positions-position))] for scan, data in zip(scans, plot_data)])
            elif trace_index != None:
                plot_data = np.array([data[trace_index] for data in plot_data])
            else:
                plot_data = np.array([np.sum(data, axis=0) / data.shape[0] for data in plot_data])  #average all the data            
              
        save_dirs = self._get_master_dirs(save_dir, 'corr'+save_ext, common_sample=common_sample)

        fig, max_lag = plot.cross_correlation(*plot_data[:2], self.sampling_rate, ylab='Correlation Coefficient',
                                     save_dir=save_dirs, fig=fig, **kwargs)
       
        self._create_fig_links(save_dirs, local_vars, 'corr'+save_ext, common_sample=common_sample)
        
        return fig, max_lag         
        
        
    ########################  Picking methods  ###############################

    def manual_pick(self, bandpass=None, **kwargs):
        '''
        Function to easily manually pick wave arrivals on a wiggle plot of a PlaceScan
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
        
        pk.manual_picker(plot_data, self.times, self.x_positions, **kwargs)

  
    def spectrogram_auto_pick(self, normed=False, bandpass=None, trace_index=None, 
                   average=False, save_dir=None, save_ext='',
                    tmax=None, max_freq=None, show=False, search_window=None, 
                    max_pick=True, threshold_pick=False, threshold=0.5, 
                    picks_save_dir=None, **kwargs):
        '''
        Function to automatically pick arrivals from a Wigner-Ville spectrogram.
        If average or trace_index is not specified, all traces are picked.
        
        Arguments:
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --trace_index: If average==False, the index of the trace in trace_data
                         to plot for
            --average: Plot the spectrogram for the full-stack average of all
                       traces in tace_data
            --save_dir: The figure directory to save to. If this is specified,
                        the figures will be saved in the scan directory, and a
                        link will be made in the specified directory.
            --save_ext: The extension of the figure filename to describe the plot.
            --tmax: The maximum time to plot for
            --max_freq: The maximum frequency to calculate the spectrogram for.
            --show: Whether or not to show the picking plot and spectrogram.
            --search_window: A [start, end] list of times within the spectrogram
                     to search for arrivals in. If not specified, the entire 
                     sepctrogram will be searched.
            --max_pick: Pick the wave arrival using the time where the absolute
                        maximum of the average spectral power occurs.
            --threshold_pick: Pick the wave arrival by selecting the first maximum
                        after average spectral power becomes greater than threshold.
            --threshold: The spectral power threshold for threshold_pick 
            --picks_save_dir: The directory and filename to save the picks to.
            --kwargs: The keyword arguments for the Wigner-Ville spectrogram and
                      plotting
            
        Returns:
            --picks: A tuple of arrival time picks
        '''
        
        local_vars = locals(); del local_vars['self']
            
        save_dirs = self._get_master_dirs(save_dir, save_ext)   
        
        if trace_index == None:
            trace_indices = range(len(self.trace_data))
        else:
            trace_indices = [trace_index]
            
        picks = dict.fromkeys(self.x_positions[tuple(trace_indices)], -1)
        for index in trace_indices:
            fig, wv = self.wigner_spectrogram(normed=normed, bandpass=bandpass, 
                        trace_index=index, average=average, save_dir=None, 
                        save_ext='',tmax=tmax, max_freq=max_freq, show=False,
                        number_of_plots=3, **kwargs)        
            
            pick, fig = pk.spectrogram_auto_pick(wv, 0.0, self.sampling_rate, 
                                     show=show, search_window=search_window, fig=fig, 
                                     save_dir=save_dirs, max_pick=max_pick,
                                     threshold_pick=threshold_pick, threshold=threshold,
                                     **kwargs)
            picks[self.x_positions[index]] = pick*1e6
        
        if picks_save_dir:
            pk.save_data(picks_save_dir, picks=picks)
        
        self._create_fig_links(save_dirs, local_vars, save_ext) 
        
      
    def dynamic_time_warping_pick(self, mode='position', scans=None, number_of_iterations=1,
                                start_position=None, start_index=None, bandpass=None,
                                dc_corr_seconds=None, tmin=0.0, tmax=None, save_picks=True,
                                reverse=False, pick_type='p', arrival_time_corr=0.0, 
                                early_err_corr=0.0, late_err_corr=0.0, **kwargs):
        '''
        Function to pick the arrival of wave energy in trace data using the
        dynamic time warping method. A list of more than one scan is provided,
        as the routine needs to compare two similar traces.
        
        Arguments:
            --mode: The mdoe of comparison for the dtw. Either 'position' or
                'scan'. 'position' compares the same position/trace index
                for two scans, while 'scan' compares adjacent traces in self
            --scans: A list of PlaceScan objects, if mode is 'position'
            --position: The position in the scans to pick the arrival for
            --trace_index: The trace_index of the scan data to pick the arrival for,
                if position is not specified.
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --dc_corr_seconds: The number of seconds at the beginning of the trace
                to calcualte a DC shift correction from.        
            --tmin: The minimum time to plot for
            --tmax: The maximum time to plot for    
            --save_picks: True to save the picks to the scan directory
            --reverse: For mode == 'scan', iterate through the scan from
                highest x_position to lowest.
            --pick_type: The type of arrival pick
            --arrival_time_corr: An arrival time correction to apply to
                each arrival time in picks. The correction is **added**, so negative
                numbers can be specified (in microseconds).
            --early_err_corr: A correction to apply to the early errors
            --late_err_corr: A correction to apply to the late errors
            --kwargs: The keyword arguments for the dtw picking
            
        Returns:
            --None
        '''

        local_vars = locals(); del local_vars['self']; del local_vars['scans']
        
        save_dirs = None
        if save_picks:
            if mode == 'scan':
                save_dirs = self._get_master_dirs(self.scan_dir, pick_type+'-'+'picks', file_types=['.csv'])
            elif mode == 'position':
                save_dirs = [scan._get_master_dirs(scan.scan_dir, pick_type+'-'+'picks', file_types=['.csv'])[0] for scan in scans]
                
        if mode == 'scan':
            scans = [self] + [self] * (number_of_iterations -1)

        #Probably broken
        if start_position != None:
            trace_inds = np.array([np.argmin(np.abs(scan.x_positions-start_position))for scan in scans])
        elif start_index != None:
            trace_inds = np.array([start_index]*len(scans))
        else:
            print('PlaceScan Dynamic Time Warping: Please provide a trace x_position\
                  or index for picking')  
            return
            
        if mode == 'scan':  #Increase trace inds by 1 each index, to compare adjacent traces in a scan
            trace_inds = trace_inds+np.arange(len(trace_inds))
            trace_inds = trace_inds[np.where(trace_inds<len(trace_inds))]
        if reverse:
            trace_inds = trace_inds[::-1]
        
        data, times = zip(*[scan._get_plot_data(tmax, tmin, False, bandpass, dc_corr_seconds) for scan in scans])
        traces = [data[i][trace_inds[i]] for i in range(len(trace_inds))]
        vels, picks = dtw.pick_vp_dtw(traces, times, self.sampling_rate, **kwargs)
        
        if save_dirs:
            if mode == 'scan':
                pk.save_data(save_dirs[0], picks=dict(zip(self.x_positions[tuple(trace_inds)],picks)), update_picks=False, arrival_time_correction=arrival_time_corr,
                    early_err_corr=early_err_corr, late_err_corr=late_err_corr)
                self._create_fig_links(save_dirs, local_vars, pick_type+'-'+'picks', make_links=False)
            elif mode == 'position':
                for i in range(len(save_dirs)):
                    pk.save_data(save_dirs[i], picks={scans[i].x_positions[trace_inds[i]]:picks[i]}, update_picks=False, arrival_time_correction=arrival_time_corr,
                    early_err_corr=early_err_corr, late_err_corr=late_err_corr)
                    scans[i]._create_fig_links(save_dirs[i], local_vars, pick_type+'-'+'picks', make_links=False)
        
        return vels, picks

    def dtw_multiscan_picking(self, scans=None, control_scan_index=0, start_position=None, 
                            start_index=0, bandpass=None, dc_corr_seconds=None, tmin=0.0,
                            tmax=None, save_picks=True, pick_type='p', picks_offset=0.0,
                            max_adjacent_jump=None, order_criteria=None, 
                            arrival_time_corr=0.0, early_err_corr=0.0, late_err_corr=0.0, **kwargs):
        '''
        Function to pick the wave arrivals for all positions
        for a set of scans, using dynamic time warping. This
        works by using the (saved) picks from all positions
        for one scan, and then determining the corresponding 
        time on the waveforms for those positions in all the
        other scans via dtw. The result is wave arrival picks
        for all positions in all scans.

        Arguments:
            --scans: A list of PlaceScan objects, if mode is 'position'
            --control_scan_index: The index of the scan in scans which
                already has picks and acts as the control pick for each position.
            --start_position: The initial position in the scans to pick the arrival for
            --start_index: The initial trace_index of the scan data to pick the arrival for,
                if start_position is not specified.
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --dc_corr_seconds: The number of seconds at the beginning of the trace
                to calcualte a DC shift correction from.        
            --tmin: The minimum time to plot for
            --tmax: The maximum time to plot for    
            --save_picks: True to save the picks to the scan directory
            --pick_type: The type of arrival pick
            --picks_offset: An x_position offset for the control scan picks
            --max_adjacent_jump: The maximum jump in time between the picks of
                two adjacent traces in a scan. Different from max_jump for
                dtw, which is the maximum jump in arrival time between 
                adjacent pressures.
            --order_criteria: An extra picking constraint which will not permit
                a pick to be either sooner (set to 'sooner') or later (set
                to' later') than the original scan's pick at that location.
            --arrival_time_corr: An arrival time correction to apply to
                each arrival time in picks. The correction is **added**, so negative
                numbers can be specified (in microseconds).
            --early_err_corr: A correction to apply to the early errors
            --late_err_corr: A correction to apply to the late errors
            --kwargs: The keyword arguments for the dtw picking
            
        Returns:
            --None
        '''

        local_vars = locals(); del local_vars['self']; del local_vars['scans']

        cs = scans[control_scan_index]   #control_scan
        del scans[control_scan_index]
        _, cs_picks = plot.picks_from_csv(cs.scan_dir+'/'+cs.scan_name+'_'+pick_type+'-'+'picks.csv')
        cs_p_picks = cs_picks[1]
        cs_data, cs_times = cs._get_plot_data(tmax, tmin, False, bandpass, dc_corr_seconds)
        
        if save_picks:
            save_dirs = [scan._get_master_dirs(scan.scan_dir, pick_type+'-'+'picks', file_types=['.csv'])[0] for scan in scans]
        
        if start_position != None:
            start_indices = [np.argmin(np.abs(scan.x_positions-start_position)) for scan in scans]
        elif start_index != None:
            start_indices = [start_index]*len(scans)

        data, times = zip(*[scan._get_plot_data(tmax, tmin, False, bandpass, dc_corr_seconds) for scan in scans])
        
        all_picks, prevpick_position = [], -1
        for i in range(len(cs.x_positions)-start_index):
            print('DTW Multiscan position: {}'.format(round(cs.x_positions[start_index+i],2)))
            trace_list, times_list = [], []
            position_picks = [-1]*len(scans)
            for scan_ind, scan in enumerate(scans):
                si = start_indices[scan_ind]+i
                if si < len(scan.x_positions):
                    try:
                        if i not in scan.muted_traces:
                            trace_list.append((data[scan_ind][si],scan_ind))
                            times_list.append(times[scan_ind])
                        else:
                            print('arf')
                    except:
                        trace_list.append((data[scan_ind][si],scan_ind))
                        times_list.append(times[scan_ind])
            
            template_pick, query_pick = cs_p_picks[start_index+i], cs_p_picks[start_index+i]
            manual, manual_windowing, ind = False, False, 0
            if template_pick == -1:
                manual, manual_windowing = True, True
            else:
                trace_list, times_list = [(cs_data[start_index+i],0)] + trace_list, [cs_times] + times_list
                
            while ind < len(trace_list)-1:   
                if len(all_picks) > 0:
                    prevpick_position = all_picks[-1][ind]
                template_pick, query_pick, template_vel, query_vel, successful =\
                    dtw.dtw_main([trace_list[ind][0],trace_list[ind+1][0]], [times_list[ind],times_list[ind+1]],
                    cs.sampling_rate, manual=manual, prev_temp_pick=template_pick, prev_query_pick=query_pick,
                    manual_windowing=manual_windowing, manual_guidance=True, extra_prev_query_pick=prevpick_position, **kwargs)

                if not manual and len(all_picks) > 1:
                    if  max_adjacent_jump and abs(all_picks[-1][trace_list[ind+1][1]]-query_pick) > max_adjacent_jump:
                        print('Arrival time difference too great between adjacent positions. Going to manual. At scan {}'.format(scans[ind].scan_name))
                        successful = False
                    elif order_criteria == 'sooner' and  query_pick < .95*cs_p_picks[start_index+i]:
                        print('DTW arrival time greater than initial scan. Going to manual. At scan {}'.format(scans[ind].scan_name))
                        successful = False
                    elif order_criteria == 'later' and  query_pick > 1.05*cs_p_picks[start_index+i]:
                        print('DTW arrival time less than initial scan. Going to manual. At scan {}'.format(scans[ind].scan_name))
                        successful = False
                        
                if successful:
                    position_picks[trace_list[ind+1][1]] = query_pick  #Might want to sort out avergaing here, but this should work.
                    manual, manual_windowing = False, False
                    ind += 1
                else:
                    manual = True

            all_picks.append(position_picks)
             
        all_picks = list(zip(*all_picks))
        if save_picks:
            for i, scan in enumerate(scans):
                x_positions = scan.x_positions[start_indices[i]:]
                picks = all_picks[i][:len(x_positions)]
                pk.save_data(save_dirs[i], picks=dict(zip(x_positions,picks)), update_picks=False, arrival_time_correction=arrival_time_corr,
                    early_err_corr=early_err_corr, late_err_corr=late_err_corr)
                scan._create_fig_links(save_dirs[i], local_vars, pick_type+'-'+'picks', make_links=False)
        

    def aic_picking(self, normed=False, bandpass=None, dc_corr_seconds=None,
                    tmax=None, tmin=0.0, save_picks=True, bounds=None,
                    arrival_time_corr=0.0, early_err_corr=0.0, late_err_corr=0.0,
                    pick_type='p',**kwargs):
        '''
        Function to automatically pick wave arrivals
        using the Akaike Information Criterion. This
        method is generally reliable and accurate, and
        is the simplest of the automatic picking algorithms
        in PlaceScan. Each trace is picked independently,
        and manual correction can easily be applied when
        desired.

        Arguments:
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --dc_corr_seconds: The number of seconds at the beginning of the trace
                        to calcualte a DC shift correction from.
            --tmax: The maximum time to plot for
            --tmin: The minimum time to plot for
            --save_picks: True to save the picks to file.
            --bounds: A two-element tuple containing lower and
                upper bounds to constrain the automatic picking. The upper and
                lower bounds can either be numbers, or a list bounds for each
                trace.
            --arrival_time_corr: An arrival time correction to apply to
                each arrival time in picks. The correction is **added**, so negative
                numbers can be specified (in microseconds).
            --early_err_corr: A correction to apply to the early errors
            --late_err_corr: A correction to apply to the late errors
            --pick_type: The type of arrival pick
            --kwargs: The keyword arguments for the ACI picker

        Returns:
            --picks: A list of the picks
        '''

        local_vars = locals(); del local_vars['self']

        if save_picks:
            save_dirs = self._get_master_dirs(self.scan_dir, pick_type+'-picks', file_types=['.csv'])

        data, times = self._get_plot_data(tmax=tmax, tmin=tmin, normed=normed,
                    bandpass=bandpass,dc_corr_seconds=dc_corr_seconds)
        picker = pk.AICPicker(data, times, self.x_positions,bounds=bounds,**kwargs)
        picks = picker.picks 

        if save_picks:
            pk.save_data(save_dirs[0], picks=dict(zip(self.x_positions,picks)), update_picks=False, 
                    arrival_time_correction=arrival_time_corr, early_err_corr=early_err_corr, late_err_corr=late_err_corr)
            self._create_fig_links(save_dirs, local_vars, pick_type+'-picks', make_links=False)
        
        return picks

    def max_amp(self):
        '''
        Return the maximum absolute amplitude in trace_data
        '''
        return np.amax(np.abs(self.trace_data[:,1,0]))
    

    ########################  Analysis methods  ##############################


    def rock_physics_calculator(self, labels, data_filename, scans=None, 
                            save_ext='', pick_type='p',**kwargs):
        '''
        Function to do rock physics calculations on PlaceScan
        objects. The scans must all be on the same sample. A
        RPCalculator object is retruned, which contains all 
        relevant rock physics data as attributes. All this data
        is automatically saved in the rp_data.p pickle file for
        the sample.

        Arguments:
            --labels: A list of labels to identify the scan or scans
            --data_filename: The csv filename (path) that contains data
                for the sample like mass, height, etc.
            --scans: A lsit of PlaceScan objects to calcualte the anisotropies for
            --save_ext: An extension for the filename
            --pick_type: The type/name of the pick to use for P-waves
            --**kwargs: The keyword arguments for the anisotropy calcualtion

        Returns:
            --anisotropies: The calcaulated anisotropies
        '''

        local_vars = locals(); del local_vars['self']; del local_vars['scans']
        
        if scans:
            picks_dirs = [scan.scan_dir+scan.scan_name+'_{}-picks.csv'.format(pick_type) for scan in scans] 
            save_dir =  self.scan_dir[:self.scan_dir[:-1].rfind('/')+1]
            id_ = self.sample
        else:
            picks_dirs = self.scan_dir+self.scan_name+'_{}-picks.csv'.format(pick_type)
            save_dir = self.scan_dir
            id_ = self.scan_name
        save_dirs = self._get_master_dirs(save_dir, 'rock_physics'+save_ext, file_types=['.p'], common_sample=True) 

        sample_rp = analysis.RPCalculator(self.sample, picks_dirs, save_dir+id_+'_rp_data.p', labels, data_filename, **kwargs)

        self._create_fig_links(save_dirs, local_vars, 'rock_physics'+save_ext, common_sample=True, make_links=False)

        return sample_rp


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
        
        try:
            scope_bits = self.metadata['bits_per_sample']
            if self.place_version < 0.8:
                scope_config = next(module['config'] for module in self.config[self.plugins_key] if 'ATS' in module["python_class_name"])  #Not called 'python_class_name' for <0.7
            else:
                scope_config = next(module['config'] for name,module in self.config[self.plugins_key].items() if 'ATS' in name)
            
            # Assume that the trace input on the scope was the second channel
            for i in range(self.trace_data.shape[1]-1):
                scope_dynmc_range = self._get_scope_dynm_range(i+1)
                scope_impedance = ''.join(list(filter(str.isdigit, scope_config['analog_inputs'][i+1]['input_impedance'])))
                
                # Change the units to volts. Maybe divide by 2 if input impedance is 50 ohms (not active yet).
                self.trace_data[:,i+1,:,:] = self.trace_data[:,i+1,:,:] / (2.0 ** scope_bits) * scope_dynmc_range
                if scope_impedance == '50':
                    self.trace_data[:,i+1,:,:] = self.trace_data[:,i+1,:,:] #Not doing anything here yet.
        except:
            print('PlaceScan _true_amps: Unable to calibrate oscilloscope amplitudes')        
        
        try:
            # Calibrate to the vibrometer units
            vib_calib = next(val for (key,val) in self.metadata.items() if 'calibration' in key and isinstance(val, type(1.0)))
            vib_calib_units = next(val for (key,val) in self.metadata.items() if 'calibration_units' in key)[:-3]    
        
            self.trace_data[:,1,:,:] = self.trace_data[:,1,:,:] * vib_calib
           
            self.amp_label = "Amplitude ({})".format(vib_calib_units)
        except:
            self.amp_label = "Amplitude (V)"
            print('PlaceScan _true_amps: Unable to calibrate vibrometer amplitudes') 
       
        
    def _get_scope_dynm_range(self, ind):
        '''
        Internal function to get the dynamic range of an
        oscilliscope channel.
        
        Arguments:
            --ind: The index of the channel in the npy array structure
        
        Returns:
            --dynm_range: The dynamic range of the scope channel
        '''
        
        if self.place_version < 0.8:
            scope_config = next(module['config'] for module in self.config[self.plugins_key] if 'ATS' in module["python_class_name"])  #Not called 'python_class_name' for <0.7
        else:
            scope_config = next(module['config'] for name,module in self.config[self.plugins_key].items() if 'ATS' in name)
        
        scope_dynmc_range = ''.join(list(filter(str.isdigit, scope_config['analog_inputs'][ind]['input_range'])))
        scope_dynmc_range = float(scope_dynmc_range) * 2.0
        if scope_dynmc_range > 50:   #Change from mV to V. Conditional is dependent on scope options.
            scope_dynmc_range = scope_dynmc_range / 1000.0 
            
        return scope_dynmc_range
    
   
    def _get_master_dirs(self, save_dir, key, common_sample=False,
                         file_types=['.png', '.pdf']):
        '''
        Function to get the filename of figures to save them in the 
        figure directory.
        
        Arguments:
            --save_dir: The save_dir arguments specified in the plotting
                        method
            --key: The key for the type of plot
            --common_sample: If the figure is not scan-specific, set to True
            --file_types: A list of strings indicating the file types to save to.
            
        Returns:
            --master_save_dirs: The directories to save the figures to
        
        '''
    
        if save_dir:
            if not common_sample:
                fig_name = self.scan_name+'_'+key
            else:
                fig_name = self.scan_name[:self.scan_name.find('_')]+'_'+key
            master_save_dirs = [save_dir+fig_name+file_type for file_type in file_types]
            return master_save_dirs
        else:
            return
    
    
    def _create_fig_links(self, save_dirs, local_vars, key, make_links=True,
                          common_sample=False):
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
            
            if not isinstance(save_dirs, list):
                save_dirs = [save_dirs]
            
            prefix_dir = self.scan_dir
            if common_sample:
                prefix_dir = prefix_dir[:prefix_dir[:-1].rfind('/')+1]
            
            if make_links:
                for _dir in save_dirs:
                    link = prefix_dir+_dir[_dir.rfind('/')+1:]
                    if not os.path.exists(link):
                        os.symlink(_dir, link)
            
            dict_dir = prefix_dir+save_dirs[0][save_dirs[0].rfind('/')+1:save_dirs[0].rfind('.')]+'.p'
            with open(dict_dir, 'wb') as file:
                pickle.dump(local_vars, file, -1)
    
    
    def _get_picks_dir(self, plot_picks):
        '''
        Function to get the directory of the arrival time picks
        for a scan.
        '''
        
        picks_dir = None
        if plot_picks:
            first_choice = self.scan_dir+self.scan_name+'_p-picks.csv'
            if os.path.exists(first_choice):
                picks_dir = first_choice
            else:
                picks_dir = self.scan_dir+'p_wave_picks.csv'
            if os.path.exists(self.scan_dir+'r_wave_picks.csv'):
                picks_dir = picks_dir#[picks_dir, self.scan_dir+'r_wave_picks.csv']
                
        return picks_dir
        
    def _get_plot_data(self, tmax=None, tmin=None, normed=False, bandpass=None,
                       dc_corr_seconds=None, decimate=False, differentiate=False):
        '''
        Function to slice the data for plotting
        
        Arguments:
            --tmax: The maximum time to plot for
            --tmin: The minimum time to plot for
            --normed: Ture if the data is to be normalised
            --bandpass: A tuple of (min_freq, max_freq) for a bandpass filter
            --dc_corr_seconds: The number of seconds at the beginning of each
                trace to calcualte the mean from, and then subtract from the
                entire trace. The results is that the first parts of the 
                trace will be centred on 0.
            --decimate: Reduce the sampling rate of the data for plotting. Recommended
                if the pltos are being saved and the original sampling rate is high.
                If True, the data is down sampled to a frequency twice the high
                corner of the bandpass. If no bandpass is applied, decimate needs
                to be an integer factor to downsample by
            --differentiate: An integer specifying how many times the signal should
                be differentiated. 
            
        Returns:
            --data: The plotting values
            --times: The plotting times
        '''

        plot_times = self.times.copy()
        plot_data = self.trace_data[:,self.data_index,:].copy()  
        s = plot_data.shape
        plot_data = plot_data.reshape(s[0]*s[1],s[2])
        
        # Detrending
        plot_data = detrend(plot_data)  
        
        if differentiate:
            from scipy.integrate import cumtrapz
            plot_data = cumtrapz(plot_data,x=plot_times,axis=1)
            plot_times = plot_times[:-1]
            plot_data = cumtrapz(plot_data,x=plot_times,axis=1)
            plot_times = plot_times[:-1]

        if bandpass:
            plot_data = bandpass_filter(plot_data, bandpass[0], bandpass[1], self.sampling_rate)

        if tmax:
            ind = np.where(plot_times <= tmax)[0][-1]
            plot_times = plot_times[:ind+1]
            plot_data = plot_data[:,:ind+1]
        if tmin != None:
            ind = np.where(plot_times >= tmin)[0][0]
            plot_times = plot_times[ind:]
            plot_data = plot_data[:,ind:] 



        if normed:
            plot_data = normalize(plot_data, normed)

        # Detrending
        plot_data = detrend(plot_data)  
        
        if dc_corr_seconds:
            plot_data = np.array([array - np.mean(array[:int(self.sampling_rate * dc_corr_seconds)])
                                        for array in plot_data])
    
        if decimate:
            if bandpass:
                factor = int(self.sampling_rate // bandpass[1] / 2)
            else:
                factor = int(decimate)
            plot_data = plot_data[:,::factor]
            plot_times = plot_times[::factor]

        return plot_data, plot_times
    
    def _get_time_window_around_pick(self, plot_data, plot_times, picks_dir, 
                                     picks_offset, position, window, taper=True):
        '''
        Function to return the data windowed about the picked arrival for
        that trace.
        '''
     
        headers, picks_data = plot.picks_from_csv(picks_dir)
    
        x_pos, start_ind, end_ind = None, 0, -1
        if headers:
            x_pos = picks_data[0] + picks_offset
            pick_index = np.argmin(np.abs(x_pos-position))
            pick = picks_data[1][pick_index]
            start_time = pick - abs(window[0])
            end_time = pick + abs(window[1])
            start_ind = np.argmin(np.abs(plot_times-start_time))
            end_ind = np.argmin(np.abs(plot_times-end_time))
            
        plot_data, plot_times = detrend(plot_data[start_ind:end_ind+1]), plot_times[start_ind:end_ind+1]    
        if taper:
            taper = np.hanning(len(plot_data))
            plot_data = np.multiply(plot_data,taper)
            
        return plot_data, plot_times
            
    
    def _get_energy_from_comments(self, calib_amps=False):
        '''
        Function to get the source laser energy from the comments
        of the config, provided the units are specified in mJ/pulse.

        Arguments:
            --calib_amp: True to divide the trace_data amplitudes by the energy
        '''

        com = self.config['comments']
        mJ_pos = com.rfind('mJ')
        com = com[:mJ_pos].strip()
        ind= -1
        while True:
            if com[ind] not in '0123456789.':
                start_pos = ind+1
                break
            elif abs(ind) == len(com):
                break
            ind += -1

        try:
            energy = float(com[start_pos:])
            if calib_amps:
                self.trace_data[:,1,:,:] = self.trace_data[:,1,:,:] / energy
            return energy
        except ValueError:
            print('PlaceScan: Could not source energy. Using energy of 1 mJ/pulse')
            return 1.0


def normalize(data, mode):
    '''
    Function to normalise the data
    
    Arguments:
        --data: The data to be normalised
        --mode: The mode of nomralisation. True for trace-by-trace
                normalisation, or 'scan' for normalisation relative
                to the maximum value in data
        
    Returns:
        --data: The normalized data
    '''
    
    if mode == True:
        return np.array([array/np.amax(np.abs(array)) for array in data])
    if mode == 'scan':
        return data/np.amax(np.abs(data))
    if isinstance(mode, (int,float)):
        return data/mode
        
        
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
        
        
def spectrum_comparison(scans, trace_indices=None, positions=None, labels=None,
                        save_dir=None, show=True, **kwargs):
    '''
    Function to plot multiple power spectra on the same plot.
    
    Arguments:
        --scans: A list of PlaceScan objects to plot the spectra for
        --trace_indices: A list of trace indices to calculate the spectra for.
                Each scan in scans has one trace index in trace_indices
        --positions: A list of x_positions to calculate the spectra for.
                Each scan in scans has one position in positions
        --labels: The labels for each of the spectra
        --save_dir: The directory to save the figure to
        --show: True to display the plot
        --**kwargs: The keyword arguments for calculating the spectra and plotting
        
    Returns:
        --fig: The figure with the spectra plotted.
    '''
    
    
    if positions:
        trace_indices = [None]*len(scans)
    elif trace_indices:
        positions = [None]*len(scans)
    else:
        positions = [None]*len(scans)
        trace_indices = [None]*len(scans)
        
        
    if not labels:
        labels = [None]*len(scans)
        
    fig = scans[0].multitaper_spectrum(position=positions[0], trace_index=trace_indices[0],
               save_dir=None, label=labels[0], show=False, plot_trace=False,
               common_sample=True, **kwargs)            
    
    s_dir = None
    for i in range(1, len(scans)):
        if i == len(scans)-1:
            s_dir=save_dir
        fig = scans[i].multitaper_spectrum(position=positions[i], trace_index=trace_indices[i], 
                 save_dir=s_dir, label=labels[i], fig=fig, show=show and i==len(scans)-1, 
                 plot_trace=False, common_sample=True, **kwargs)    

    return fig

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        









        
    
