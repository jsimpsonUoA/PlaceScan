'''
Python module with functions to pick arrival times of wave
phases on time series waveform data. Functions provide both manual
and automatic picking methods.

Jonathan Simpson, jsim921@aucklanduni.ac.nz
Masters project, PAL Lab UoA, April 2018
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
from plotting import format_fig
import inspect

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def manual_picker(values, times, x_positions, save_file=None, amp_factor=50, tmax=1e20):
    '''
    Function to create a plot which makes it easy to manually
    pick arrival times.
    
    Arguments:
        --values: The amplitude values of the traces (list of traces)
        --times: The times for the traces (single list, same for all)
        --x_positions: The trace positions
        --save_file: The filename to save the picks to
        --amp_factor: Factor to multiply trace values by when plotting
        --tmax: The maximum time to plot for
        
    Returns:
        None
    '''

    global pick_times, late_err, early_err, current_trace, pick_markers
    
    fig, ax = plt.subplots()
    current_trace = x_positions[0]
    pick_times = dict.fromkeys(x_positions, -1)
    late_err = dict.fromkeys(x_positions, -1)
    early_err = dict.fromkeys(x_positions, -1)
    
    plotted_traces = []
    for i in range(len(values)):
        color = 'k'
        if x_positions[i] == current_trace:
            color = 'r'
        data = values[i]*amp_factor+x_positions[i]
        line, = plt.plot(times, data, color, linewidth=0.5)
        #ax.fill_betweenx(times, data, x_positions[i], where=data>x_positions[i], color='black')
        plotted_traces.append(line)     

    maxy = x_positions[-1]+(x_positions[-1]+x_positions[0])*0.01
    miny = x_positions[0]-(x_positions[-1]+x_positions[0])*0.01  
    fig, ax = format_fig(fig, ax, '', 'Time ($\mu$s)', "Position ($^\circ$)", (0.0, min(tmax, times[-1])),
                    (miny, maxy), show=False)
    pick_markers = [] 
    
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event', lambda event: key_press(event, fig,
                                   plotted_traces, x_positions, save_file))         
    
    plt.show()
    
    
def key_press(event, fig, plotted_traces, distances, filename):
    '''
    Function which handles the arrival time picking of
    traces. Press 'q' to record the current y-position of
    the mosue on the plot, 'w' to skip the current trace,
    or Delete to go back a trace.
    '''   
    
    global pick_times, late_err, early_err, current_trace
    
    close_fig = False
    time = event.xdata
    
    if time != None:
        if event.key == 'q':
            pick_times[current_trace] = time
        elif event.key == 'e':
            early_err[current_trace] = time
        elif event.key == 'r':
            late_err[current_trace] = time
        elif event.key == 'w':
            close_fig = update_figure(fig, plotted_traces, distances)
        elif event.key == 'delete':
            close_fig = update_figure(fig, plotted_traces, distances, forward=False)  #Position before delete is important
            if current_trace in pick_times:
                pick_times[current_trace] = -1
                late_err[current_trace] = -1
                early_err[current_trace] = -1
            
        if (pick_times[current_trace] != -1) and\
               (late_err[current_trace] != -1) and\
                    (early_err[current_trace] != -1):
            close_fig = update_figure(fig, plotted_traces, distances)                
            
        if close_fig:
            plt.close()
            
            if filename:
                save_data(filename)


def update_figure(fig, plotted_traces, distances, forward=True):
    '''
    Function to update the figure and change the colour of the 
    current trace
    '''
    
    
    global current_trace, pick_times, late_err, early_err, pick_markers
    
    current_index = list(distances).index(current_trace)
    
    if forward and current_index == len(distances)-1:
        return True
    elif not forward and current_index == 0:
        return False
    elif forward:
        current_trace = distances[current_index+1]
        plotted_traces[current_index].set_color('k')
        plotted_traces[current_index+1].set_color('r')
    elif not forward:
        current_trace = distances[current_index-1]
        plotted_traces[current_index].set_color('k')
        plotted_traces[current_index-1].set_color('r')
        
    update_markers(fig, plotted_traces)
        
    fig.canvas.draw()
    return False      
        

def update_markers(fig, plotted_traces):
    '''
    Function to update the pick markers on the plot
    '''
    
    global pick_times, late_err, early_err, pick_markers

    colors = ['r','g','b']

    [marker[0].remove() for marker in pick_markers]
    pick_markers = []
    
    dicts = [pick_times, late_err, early_err]
    for i0 in [0,1,2]:
        
        #Get the markers for plotting arrival time picks
        vals, keys = list(dicts[i0].values()), list(dicts[i0].keys())
        vals = [x for _,x in sorted(zip(keys,vals))]; keys.sort()
        
        #Make pick time marker sit on trace
        for i in range(len(keys)):
            if vals[i] != -1:
                data = np.transpose(plotted_traces[i].get_xydata())
                pick_time_index = np.abs(data[0]-vals[i]).argmin()
                keys[i] = data[1][pick_time_index]
                
        #Don't plot the marker if no pick is made
        keys = [keys[i] for i in  range(len(keys)) if vals[i]!=-1]
        vals = [val for val in vals if val!=-1]
        
        pick_markers.append(plt.plot(vals, keys, marker='x', markersize=6.0, 
                                color=colors[i0], linestyle='',markeredgewidth=0.5))


def save_data(filename, picks=None, early_err_=None, late_err_=None, update_picks=True,
            arrival_time_correction=0.0, early_err_corr=0.0, late_err_corr=0.0):
    '''
    Function to save the picks data

    Arguments:
        --filename: The name of the file to save the picks to
        --picks: The arrival time picks, as a dictionary where the
            keys are the positions of the picks and the values are
            the picks (in units of microseconds)
        --early_err: If picks is given, then this can be a dict of
            early errors. Can be None
        --late_err: If picks is given, then this can be a dict of
            late errors. Can be None
        --update_picks: True if a preexisting picks file is to be updated
            with the new picks, rather than completely overwriting the file
        --arrival_time_correction: An arrival time correction to apply to
            each arrival time in picks. The correction is **added**, so negative
            numbers can be specified (in microseconds).
        --early_err_corr: A correction to apply to the early errors
        --late_err_corr: A correction to apply to the late errors
        
    Returns:
        --None
    '''
     
    global pick_times, late_err, early_err
     
    if picks:
        keys = picks.keys()
        pick_times = picks
        if not late_err_:
            if float(late_err_corr) == 0.:
                late_err =  dict.fromkeys(keys, -1)
            else:
                late_err =  pick_times.copy()
        else:
            late_err = late_err_
        if not early_err_:
            if float(early_err_corr) == 0.:
                early_err =  dict.fromkeys(keys, -1)
            else:
                early_err =  pick_times.copy()
        else:
            early_err = early_err_
    
    picks_dicts = [pick_times, late_err, early_err]
    if update_picks:
        prev_picks_dicts = picks_data_from_csv(filename)
        for i in range(3):
            prev_picks_dicts[i].update(picks_dicts[i])
        pick_times, late_err, early_err = tuple(prev_picks_dicts)

    for i, times_dict in enumerate(picks_dicts):
        for key, val in times_dict.items():
            times_dict[key] = val + [arrival_time_correction, late_err_corr, early_err_corr][i]  
    items = [list(pick_times.items()), list(early_err.items()), list(late_err.items())]
    [_list.sort(key=lambda x: x[0]) for _list in items]
    
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['Position','P-wave Pick (us)', 'Early Error (us)', 'Late Error (us)'])
        for i in range(len(items[0])):
            csvwriter.writerow([items[0][i][0],items[0][i][1],items[1][i][1],items[2][i][1]])
            
            
def picks_data_from_csv(filename):
    '''
    Function to retrieve wave arrival picks from a csv file.
    
    Arguments:
        --directory: The filename of the picks csv
        
    Returns:
        --headers: The column headings in the picks file
        --columns: The columns in the csv file
    '''
    
    try:
        with open(filename, 'r') as file:
            
            reader = list(csv.reader(file))
            data = np.zeros((len(reader)-1,len(reader[0])))
            
            dicts = [{}, {}, {}]
            for i in range(len(reader[1:])):
                data[i] = np.array([float(item) for item in reader[i+1]])
            
            data = data.transpose()
            for i in range(len(reader[0])-1):
                dicts[i] = dict(zip(data[0], data[i+1]))
                
        return dicts
    except:
        return [{},{},{}]
           
                    
            
def spectrogram_auto_pick(spect, start, sampling_rate, search_window=None,
                          fig=None, show=False, save_dir=None, max_pick=True,
                          threshold_pick=False, threshold=0.5, **kwargs):
    '''
    Function to pick wave arrivals from the spectrogram of a time series.
    The frequency values are summed over time to produce average spectral
    power as a function of time. Broadband high-power peaks are seleected
    within a specified time window as wave arrivals. If the maximum is found
    to be the edge of the serach interval, then the search interval is expanded
    to look for a new maximum.
    
    Arguments:
        --spect: The spectrogram (as a numpy 2-D array with time in x
                 and frequency in y).
        --time_window: The start time of the spectrogram
        --sampling_rate: The sampling rate of the data
        --search_window: A range of times within the spectrogram to search
                         for wave arrivals.
        --fig: The figure on which the spectrogram is plotted. The average
               power and wave pick(s) will be plotted on the last axis
        --show: True if the picking figure/spectrogram is to be displayed.
        --save_dir: The figure directory(s) to save to. 
        --max_pick: Pick the wave arrival using the time where the absolute
                    maximum of the average spectral power occurs.
        --threshold_pick: Pick the wave arrival by selecting the first maximum
                    after av_pow becomes greater than threshold.
        --threshold: The spectral power threshold for threshold_pick 
        --**kwargs: Keyword arguments for the plotting
                         
    Returns:
        --pick: The time of the wave arrival pick, if found.
        --fig: The figure showing the spectrogram and picking process
    '''    
    
    av_pow = np.sum(spect,axis=0) / spect.shape[0]
    
    end = start + 1 / sampling_rate * spect.shape[1]
    if search_window:
        first_time, last_time = search_window[0], search_window[1]
    else:
        first_time, last_time = start, end
    
    first_index = int((first_time - start) * sampling_rate)
    last_index = int((last_time - start) * sampling_rate)
    
    if max_pick:
        looping = True
        search_inc = 0.2e-6   #Number of seconds to shorten window if max not found in interval
        while looping:
            data = av_pow[first_index:last_index]
            if len(data) == 0 or last_index<1:
                looping = False
                max_ind, pick = None, None        
            else:
                max_ind = np.argmax(data)
            if search_window:
                if max_ind == len(data)-1:
                    last_index = last_index - int((search_inc * sampling_rate))
                    last_time = last_time - search_inc
                elif max_ind == 0:
                    first_index = first_index + int((search_inc * sampling_rate))
                    first_time = first_time + search_inc
                else:
                    looping = False
            else:
                looping = False
    elif threshold_pick:
        peak_picking_threshold = 3  #Number of data points after a maximum in av_pow to qualify the max as a pick
        max_ind = None
        data = av_pow[first_index:last_index]
        for i1 in range(len(data)):
            if data[i1] > threshold:
                slopes = np.diff(data[i1:])
                prev_neg = 0
                for i2 in range(len(slopes)):
                    if slopes[i2] <= 0.:
                        prev_neg += 1
                    elif prev_neg and slopes[i2] > 0.:
                        prev_neg = 0
                    if prev_neg > peak_picking_threshold:
                        max_ind = i1+i2-prev_neg
                        break
                break
        
    if max_ind != None:
        max_ind = int((start + first_time) * sampling_rate + max_ind)
        pick = max_ind / sampling_rate
    else:
        pick = -1.        
    
    print('Arrival Time Pick: {}'.format(pick))
        
    if fig:
        ax = fig.get_axes()[-2]
        ax.plot(np.arange(start,end,1/sampling_rate)*1e6, av_pow)
        ax.axvline(x=pick*1e6, linestyle='--', color='r', linewidth=1.0)

        format_kwargs = {key:val for key,val in kwargs.items() if 
                         key in list(inspect.getargspec(format_fig))[0]}
        format_fig(fig, ax, ylab='Average Power', xlab='Time ($\mu$s)',
                   save_dir=save_dir, **format_kwargs)
    
    return pick, fig


def correct_picks(filename, pick_corr=None, early_err_corr=0.0, late_err_corr=0.0):
    '''
    A function to apply a correction to the  picks in a
    picks csv file.
    '''
    
    data = picks_data_from_csv(filename)
    
    save_data(filename, picks=data[0], early_err_=None, late_err_=None, update_picks=False,
         arrival_time_correction=pick_corr, early_err_corr=early_err_corr, late_err_corr=late_err_corr)


def smooth_picks_by_av(filename, num_of_traces=3):
    '''
    A function to smooth the picks in a scan by taking the
    average of multiple picks. num_of_traces should be an odd
    number.
    '''

    data = picks_data_from_csv(filename)
    num_either_side = num_of_traces // 2
    items = data[0].items()
    items = sorted(items, key=lambda item:item[0])
    pos = [item[0] for item in items]
    picks = [item[1] for item in items]

    new_picks = []
    for i in range(len(picks)):
        inds = np.arange(i-num_either_side,i+num_either_side+1)
        inds = inds[np.where(inds > -1)]
        inds = inds[np.where(inds < len(picks))]
        av_pick = picks[inds[0]]
        for ind in inds[1:]:
            av_pick += picks[ind]
        new_picks.append(av_pick / len(inds))

    save_data(filename, picks=dict(zip(pos,new_picks)), update_picks=True)
    































