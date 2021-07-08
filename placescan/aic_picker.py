'''
This script performs first break arrival time picking
of waveforms using a simple implementation of the 
Akaike Information Criterion. Pre-processing such
as bandpass filtering may applied in another script
before importing the AICPicker class. 

The code makes its AIC pick and then provides a plot
to manually correct the picks. Generally, there needs
to be good S/N to get reliable picks. The code is designed
to pick multplie waveforms at a time (e.g. a laser ultrasonic
scan).

Author: Jonathan Simpson, Physical Acoustics Lab,
        University of Auckland
Email:  jsim921@aucklanduni.ac.nz
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import argrelextrema, detrend



class AICPicker():
    def __init__(self, data, times, x_pos, amp_factor=10.,bounds=None,
                plot_bounds=True, preferred_bound=None, plot_example=False,
                title=''):
        '''
        AICPicker is a Python class to pick the wave arrival times based on the
        Akaike Information Criterion. See [paper] for info. All that is required
        is the time series data, and the corresponding times. 

        The picking is performed simplisitcally, and the plot is displayed
        afterwards for manual checking and picking

        Arguments:
            --data: A 1-D array of time series data (i.e. the y data),
                or a 2D array of multiple time series arrays.
            --times: The times for the data (i.e. the x-data). 1D.
                Same length as the number of time series points.
            --x_pos: The x positions, or a list of identifiers, for the
                traces in data.
            --amp_factor: The amplification factor of the manaul picking
                wiggle plot
            --bounds: Optional bounds to constrain the picking, given as
                a two-element tuple with the lower and upper bounds. These
                upper and lower bounds can either be numbers, or a list of
                bounds for each trace.
            --preferred_bound: If a local minima cannot be found within the 
                specified bounds, the pick will be placed at this bound.
                Can be one of 'lower', 'upper', or None, if keeping the pick
                outside the bounds is acceptable.
            --plot_bounds: True to plot the bounds on the manual picking plot
            --plot_example: False, or an x_pos to plot an example of the trace 
                with the AIC function and pick
            --title: A title for the plot
        '''

        self.data = np.atleast_2d(data)
        self.times = np.asarray(times)
        self.x_pos = x_pos
        self.amp_factor = amp_factor
        self.bounds = bounds
        self.plot_bounds = plot_bounds
        self.num_t = self.data.shape[0]
        self.picks = np.zeros(self.num_t) - 1.
        self.warn = False
        self.preferred_bound = preferred_bound
        self.title = title

        self.sort_bounds()
        self.get_picks()
        if plot_example:
            self.plot_aic_pick(plot_example)
        self.manual_picks()

    def sort_bounds(self):
        '''
        Function which sorts the bounds for the picking.
        At completion of this function, self.bounds is either
        None or a 
        '''

        try:
            if self.bounds:
                for i, bound in enumerate(self.bounds):
                    try:
                        if len(bound) == self.num_t:
                            self.bounds[i] = bound
                        else:
                            raise ValueError
                    except TypeError:
                        self.bounds[i] = [bound]*self.num_t
        except ValueError:
            print('AICPicker: Bounds not provided in the correct format.')
                

    def get_picks(self):
        '''
        Function which handles all the automatic picking 
        using aic
        '''

        for i in range(self.num_t):
            aic_func = self.aic(self.data[i])
            
            #If the trace was zero muted
            if np.mean(aic_func) == 0.:
                pick = -1.
            else:
                #If bounds are given, choose the smallest local minima in the
                #given range as the pick. This will always work for clean data,
                #provided the pick is actually in the bounds. This hopefully 
                #deals well with data that has lower S/N as well.
                if self.bounds:
                    low_i = np.argmin(np.abs(self.times-self.bounds[0][i]))
                    upp_i = np.argmin(np.abs(self.times-self.bounds[1][i]))
                    aic_func = aic_func[low_i:upp_i+1]
                    local_minima_inds = argrelextrema(aic_func, np.less)
                    local_minima = aic_func[local_minima_inds]
                    #print(local_minima_inds,np.argmin(local_minima))
                    try:
                        pick_index = local_minima_inds[0][np.argmin(local_minima)]+low_i
                        pick = self.times[pick_index]
                    except:
                        if not self.warn:
                            print('AICPciker Warning: Could not find pick(s) in given bounds.')
                            self.warn = True
                        if self.preferred_bound == 'lower':
                            pick = self.bounds[0][i]
                        elif self.preferred_bound == 'upper':
                            pick = self.bounds[1][i]
                        else:
                            pick_index = np.argmin(aic_func)
                            pick = self.times[pick_index]
                else:
                    pick_index = np.argmin(aic_func)
                    pick = self.times[pick_index]
            self.picks[i] = pick

        self.correct_blanks()


    def aic(self, signal):
        '''
        Function which calculates the aic function. All the
        automatic picking is performed here with no parameters
        to set.
        '''
        length = len(signal)
        output = [0]
        with np.errstate(divide='ignore'):
            for k in range(1, length - 1):
                val = (k * np.log(np.var(signal[0:k])) +
                    (length-k-1) * np.log(np.var(signal[k+1:length])))
                if val == -np.inf:
                    val = 0
                output.append(val)
        output.append(0)
        return np.array(output)

    def correct_blanks(self):
        '''
        Function which looks through the picks and
        detects where a pick has not been made (i.e the pick
        is -1.). The -1. is substitued for a pick that is
        on the line joining the first two proper picks either
        side of the blank pick. For where the blank pick is at
        the start or end of the data, the first or last pick is
        substituted. Assumes evenly spaced and continuous x_positions.
        '''

        for i in range(len(self.picks)):
            pick = self.picks[i]
            if pick == -1.:
                i1, i2 = i, i
                while i1 > -1:
                    if self.picks[i1] != -1.:
                        break
                    i1 = i1-1
                while i2 < len(self.picks):
                    if self.picks[i2] != -1.:
                        break
                    i2 = i2+1
                if i1 == -1:
                    new_pick = self.picks[i2]
                elif i2 == len(self.picks):
                    new_pick = self.picks[i1]
                else:
                    num_consec_blanks = i2-i1
                    prev_pick, next_pick = self.picks[i1], self.picks[i2]
                    new_pick = prev_pick + ((i-i1) / num_consec_blanks * (next_pick-prev_pick))

                self.picks[i] = new_pick

    def manual_picks(self):
        '''
        Function to handle the manual inspection and picking
        after the automatic picking
        '''
        self.wiggle_plot()


    def wiggle_plot(self):
        '''
        A function to create an interactive wiggle plot for manual
        inspection of the picks
        '''

        fig,ax = plt.subplots(1,figsize=(10,10))

        pick_y = []
        for i in range(self.num_t):
            data = self.data[i]*self.amp_factor+self.x_pos[i]
            line, = plt.plot(self.times, data, 'k', linewidth=0.5, marker='')
            ax.fill_between(self.times, data, self.x_pos[i], where=data>self.x_pos[i], color='black')
            new_points = np.column_stack((self.times, data))
            if i >0:
                plotted_points = np.concatenate((plotted_points,new_points),axis=0)
            else:
                plotted_points = new_points
            pick_y.append(data[np.argmin(np.abs(self.times-self.picks[i]))])

        self.picks_line = plt.plot(self.picks, pick_y, 'b-', linewidth=1.,label='Current Picks')

        if self.plot_bounds:
            plt.plot(self.bounds[0],self.x_pos,'-r',linewidth=.5,label='Lower Bound')
            plt.plot(self.bounds[1],self.x_pos,'-g',linewidth=.5,label='Upper Bound')

        plt.ylabel('Position')
        plt.xlabel('Time (us)')
        plt.title(self.title)

        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.mpl_connect('key_press_event', lambda event: self.update_fig(event, fig,
                                             plotted_points, pick_y))    

        plt.title('Use cursor to hover over desired arrival time pick on a trace and\npress any key to change the pick to mouse location. Close the plot when complete')
        plt.legend(loc='upper right')
        plt.show()

    def update_fig(self, event, fig, plotted_points, pick_y):
        '''
        Function to update the figure.
        '''

        x, y = event.xdata, event.ydata
        dist = np.sum((plotted_points - np.array([x,y]))**2, axis=1)
        closest_ind = np.argmin(dist)
        closest_point = plotted_points[closest_ind]
        trace_ind = closest_ind//len(self.times)

        self.picks[trace_ind] = closest_point[0]
        pick_y[trace_ind] = closest_point[1]
        self.picks_line[0].remove()
        self.picks_line = plt.plot(self.picks, pick_y, 'b-', linewidth=1.,label='Current Picks')
        fig.canvas.draw()

    def plot_aic_pick(self, position):
        '''
        Function to plot a trace with its AIC pick
        '''

        fig, ax = plt.subplots(nrows=2, sharex=True,figsize=(9,5))
        
        index = np.argmin(np.abs(self.x_pos-position))
        trace = self.data[index]
        ax[0].plot(self.times,trace, linestyle='-',color='#2272b5',marker='')
        aic = self.aic(trace)
        ax[1].plot(self.times[5:-5], aic[5:-5], linestyle='-',color='#2272b5',marker='')
        pick_time = self.times[np.argmin(aic)]
        ax[0].axvline(pick_time, linestyle='--', color='r')
        ax[1].axvline(pick_time, linestyle='--', color='r')
        
        ax[0].set_yticks([]); ax[1].set_yticks([])
        ax[0].set_ylabel('$u_i(t)$')
        ax[1].set_xlabel('Time ($\mu$s)'), ax[1].set_ylabel('AIC')
        plt.show()


def _process_kwargs(kwargs):
    keys = []
    values = []

    for item in kwargs:
        if item.find('=') != -1:
            key_val = item.split('=')
            keys.append(key_val[0])
            values.append(key_val[1])

    for i in range(len(keys)):
        if keys[i] == 'amp_factor':
            values[i] = float(values[i])
        elif keys[i] == 'bounds':
            bounds = values[i].split(',')
            bounds[0] = bounds[0][1:]
            bounds[-1] = bounds[-1][:-1]
            bounds = list([float(num) for num in bounds])
            values[i] = bounds
        elif keys[i] == 'plot_bounds':
            values[i] = bool(values[i])
        elif keys[i] == 'preferred_bound':
            values[i] = str(values[i])
        elif keys[i] == 'plot_example':
            if values[i] == 'True':
                values[i] = 1.
            else:
                values[i] = float(values[i])
        elif keys[i] == 'title':
            values[i] = str(values[i])
        else:
            raise

    kwargs = dict(zip(keys, values))

    return kwargs


if __name__ == '__main__':

    args = sys.argv
    raise_exception = False
    extracting_args = True

    try:
        filename = args[1]
        output_name = args[2]
        trace_field = args[3]
        data_index = int(args[4])
        sampling_rate = float(args[5])

        if len(args) > 6:
            kwargs = args[6:]
            kwargs = _process_kwargs(kwargs)
        else:
            kwargs = {}

        raise_exception = True
        extracting_args = False
        npy = np.load(filename)
        trace_data = npy[trace_field][:,data_index,0]
        trace_data = detrend(trace_data / np.amax(trace_data))
        times = np.arange(trace_data.shape[-1]) / sampling_rate * 1e6
        
        try:
            x_pos = npy['RotStage-position']
        except:
            x_pos = np.arange(len(trace_data))

        picker = AICPicker(trace_data, times, x_pos, **kwargs)

        with open(output_name,'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Position','P-wave Pick (us)'])
            for i in range(len(picker.picks)):
                writer.writerow([x_pos[i],picker.picks[i]])            

    except:
        if not raise_exception:
            print('\nAIC Picker: Some arguments may not be present or are invalid.\n')
            
            print('\nThis script performs AIC first break picking on data acquired with the PLACE acquisition software.\n\
Please pass the name of a valid PLACE .npy file, the name of an output csv file for the picks, the\n\
field in which the data is saved (e.g. "ATS660-trace"), the array index where the data is saved,\n\
and the sampling rate of the data (in Hz). Keyword arguments may be specified after this.\n\n')

            print('Usage:   python aic_picker.py [input_filename] [output_filename] [trace_field] [data_index] [sampling_rate] [kwargs]\n\n\
Example: python aic_picker.py data.npy scan_picks.csv ATS660-trace 0 bounds=[0,10] title="AIC Picker"\n\n')

        else:
            raise