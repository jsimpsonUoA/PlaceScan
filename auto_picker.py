'''
Python code to facilitate picking of wave arrivals on wiggle plots.
Press 'q' to record the position of the cursor, 'w' to skip the trace,
and 'Delete' to undo the last pick

Jonathan Simpson, jsim921@aucklanduni.ac.nz
Masters project, PAL Lab UoA, April 2018
Main code originally written January 2018 for UniService Work
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
from plotting import format_fig

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def auto_picker(values, times, x_positions, save_file=None, amp_factor=50, tmax=1e20):
    '''
    Function to create a plot which makes it easy to 
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


def save_data(filename):
    '''
    Function to save the picks data
    '''
     
    global pick_times, late_err, early_err
     
    items = [list(pick_times.items()), list(early_err.items()), list(late_err.items())]
    [_list.sort(key=lambda x: x[0]) for _list in items]
    
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['Position','P-wave Pick (us)', 'Early Error (us)', 'Late Error (us)'])
        for i in range(len(items[0])):
            csvwriter.writerow([items[0][i][0],items[0][i][1],items[1][i][1],items[2][i][1]])
