'''
Python functions to support picking of wave arrivals
using the dynamic time warping (DTW) method with PlaceScan
objects. 

Jonathan Simpson, jsim921@aucklanduni.ac.nz
Masters project, PAL Lab UoA, April 2018
Originally developed by Evert Duran, edur409@aucklanduni.ac.nz
See https://github.com/paul-freeman/poropyck for the full
DTW poropyck python package.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from dtw import dtw


class Cursor(object):
    def __init__(self, ax, axx, axy):
        
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        
        self.lx_x = axy.axhline(color='k')  # the horiz line
        self.ly_y = axx.axvline(color='k')  # the vert line

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)
        self.lx_x.set_ydata(y)
        self.ly_y.set_xdata(x)
        
        plt.draw()

def pick_vp_dtw(amplitudes, times, sampling_rate, path_length=1.0, length_error=.01, manual=False,
                window_size=2., max_jump=1.5, min_time=5., alpha=0, plot_lags=False):
    
    global COORDS
    print('Alpha:', np.abs(alpha))
    print('Number of dtw runs:', len(amplitudes) //2)
    vels, picks = [], []
    template_arrival, query_arrival, template_vel, query_vel, successful = \
            dtw_main([amplitudes[0], amplitudes[1]], [times[0], times[1]], 
            sampling_rate, path_length=path_length, length_error=length_error, 
            manual=True, manual_windowing=True, alpha=np.abs(alpha), plot_lags=plot_lags)
    vels += [template_vel, query_vel]
    picks += [template_arrival, query_arrival]
    
    i, master_manual = 1, manual
    while i < len(amplitudes)-1:
        template_arrival, query_arrival, template_vel, query_vel, successful = \
                dtw_main([amplitudes[i], amplitudes[i+1]], [times[i], times[i+1]], 
                sampling_rate, path_length=path_length, length_error=length_error,
                manual=manual, window_size=window_size, prev_temp_pick=template_arrival, 
                prev_query_pick=query_arrival, max_jump=max_jump, alpha=np.abs(alpha), plot_lags=plot_lags)
                
        if successful:       
            #print('{}: Diff: {}s'.format(i, round(picks[i] - template_arrival,3)))
            vels[i] = (vels[i] + template_vel) / 2
            vels += [query_vel]
            picks[i] = (picks[i] + template_arrival) / 2   # Not averaging here says that the query arrival pick is correct
            picks += [query_arrival]
            i, manual = i+1, master_manual
            
        else:
            print(i)
            manual = True
            template_arrival, query_arrival = picks[i-1], picks[i]
            
    return vels, picks
        

def multiple_run_dtw(amplitudes, times, sampling_rate, path_length=1.0, length_error=.01, manual=False,
                window_size=2., max_jump=1.5, min_time=5.):
    '''
    Bit of an experiemnt. Delete if not needed.
    '''
    global COORDS

    vels, picks = [], []
    template_times, query_times, template_picks, query_picks, dtw_template, dtw_query = \
            dtw_main([amplitudes[0], amplitudes[1]], [times[0], times[1]], 
            sampling_rate, path_length=path_length, length_error=length_error, 
            manual=True, manual_windowing=True)
    
    picking_range = (get_index(dtw_template, min(template_picks), .9/sampling_rate), 
                     get_index(dtw_template, max(template_picks), .9/sampling_rate))
    initial_template_times = dtw_template[picking_range[0]:picking_range[1]]
    initial_query_times = dtw_query[picking_range[0]:picking_range[1]]
    
    all_vels, all_picks = [], []
    
    print('Number of runs: {}'.format(len(initial_template_times)))
    print(initial_template_times, initial_query_times)
    for run in range(len(initial_template_times)):
        print('Run {}'.format(run))
        vels, picks = [], []
        template_arrival, query_arrival, template_vel, query_vel, successful = \
                dtw_main([amplitudes[0], amplitudes[1]], [times[0], times[1]], 
                sampling_rate, path_length=path_length, length_error=length_error, 
                manual=False, manual_windowing=False, window_size=window_size,
                prev_temp_pick=initial_template_times[run], prev_query_pick=initial_query_times[run],
                max_jump=max_jump, manual_guidance=False)
        vels += [template_vel, query_vel]
        picks += [template_arrival, query_arrival]
            
        i = 1
        while i < len(amplitudes)-1:
            template_arrival, query_arrival, template_vel, query_vel, successful = \
                    dtw_main([amplitudes[i], amplitudes[i+1]], [times[i], times[i+1]], 
                    sampling_rate, path_length=path_length, length_error=length_error,
                    manual=False, window_size=window_size, prev_temp_pick=template_arrival, 
                    prev_query_pick=query_arrival, max_jump=max_jump, manual_guidance=False)
         
            vels[i] = (vels[i] + template_vel) / 2
            vels += [query_vel]
            picks[i] = template_arrival#picks[i] # (picks[i] + query_arrival) / 2   # Not averaging here says that the query arrival pick is correct
            picks += [query_arrival]
            i+=1
        
        all_vels.append(vels)
        all_picks.append(picks)

    return all_vels, all_picks

def dtw_main(amplitudes, times, sampling_rate, path_length=1.0, length_error=.01, manual=False,
             window_size=2., prev_temp_pick=0., prev_query_pick=0., max_jump=1.5,
             manual_windowing = False, manual_guidance=True, alpha=0, plot_lags=False,
             extra_prev_query_pick=-1.0):
    
    '''
    This is messy and I need to write a proper docstring

    extra_prev_query_pick is for the dtw plot to show where the 
    previous position's pick was in the same scan if doing
    a multiscan run.
    '''

    global COORDS
    
    COORDS = []
    
    # The user picks the window for the dtw picking
    if manual_windowing:
        template_trace, template_times = manual_window_pick(amplitudes[0], times[0])
        query_trace, query_times = manual_window_pick(amplitudes[1], times[1])
    else:
        template_window = (prev_query_pick-window_size, prev_query_pick+window_size)
        query_window_centre = prev_query_pick+(prev_query_pick-prev_temp_pick)
        query_window = (query_window_centre-window_size, query_window_centre+window_size)
        
        prev_pick_index =get_index(times[0], prev_query_pick, .9e6/sampling_rate)
        template_trace, template_times = get_windowed_data(amplitudes[0], times[0],
                             *template_window, norm_factor=max(amplitudes[0]))    #Normalising by maximum in entire trace atm
        query_trace, query_times = get_windowed_data(amplitudes[1], times[1],     
                             *query_window, norm_factor=max(amplitudes[1]))       #Normalising by maximum in entire trace atm
    
    template_trace_env, query_trace_env = np.abs(hilbert(template_trace)), np.abs(hilbert(query_trace))
    
    # Get the dtw time arrays for both the waveform and envelope of the windowed traces.
    dtw_template, dtw_query = do_dtw(template_trace, query_trace, template_times, query_times, plot=manual_windowing, alpha=alpha)
    dtw_template_env, dtw_query_env = do_dtw(template_trace_env, query_trace_env, template_times, query_times, plot=False, alpha=alpha)
    
    if manual:
        looping = True
        while looping:
            dtw_plot([template_trace, template_trace_env], [query_trace, query_trace_env], template_times,
                         query_times, [dtw_template, dtw_template_env], [dtw_query, dtw_query_env],
                         manual_picking=True, template_picks=[prev_query_pick,0.,0.],query_picks=[extra_prev_query_pick,0.0,0.0])
            
            #Extract the times from the DTW picking
            if len(COORDS) > 0:
                looping = False
                tp, qp = [], []
                template_picks = list([np.append(tp,COORDS[i][1]) for i in range(len(COORDS))])
                query_picks = list([np.append(qp,COORDS[i][0]) for i in range(len(COORDS))])

                if plot_lags:
                    fig = lag_time_plot(dtw_template, dtw_query)
                if len(COORDS) > 1:
                    return template_times, query_times, template_picks, query_picks, dtw_template, dtw_query                  
            print('Please pick some arrival times on the plot')

    else:
        template_pick_index = get_index(dtw_template, prev_query_pick, 1.2e6/sampling_rate)
        if not template_pick_index:
            print('PlaceScan DTW: Template pick time not found in dtw array.')
            template_pick_index = 0
            
        template_pick, query_pick = prev_query_pick, dtw_query[template_pick_index]
        prelim_pick = (template_pick, query_pick)
        one_to_one_points = list(zip(*smoothed_gradient_av(dtw_template, dtw_query)))
        smallest_dist = np.inf
        
        if manual_guidance:
            #  Check for large dispersion in the region where picking
            window_samples = int(max_jump /1e6 * sampling_rate/2.)
            poss_t_times = dtw_template[template_pick_index-window_samples:template_pick_index+window_samples]
            poss_q_times = dtw_query[template_pick_index-window_samples:template_pick_index+window_samples]
            grads = np.asarray(np.diff(poss_t_times)/np.diff(poss_q_times))
            max_consec_infs, consec_infs = 1, 1
            for i in range(1, len(grads)):
                if (grads[i]==0. or grads[i]==np.inf) and  (grads[i-1]==0. or grads[i-1]==np.inf):
                    consec_infs += 1
                else:
                    consec_infs = 1
                if consec_infs > max_consec_infs:
                    max_consec_infs = consec_infs
            
            if max_consec_infs > .75*sampling_rate:
                print('Picking area too dispersive. Going to manual.')
                return prev_temp_pick, prev_query_pick, None, None, False
        
        
        # Snap to nearest one-to-one candidate
        one_to_one_candidates = []
        #Refine the pick, if reasonable
        for i in range(len(one_to_one_points)):
            dist = np.linalg.norm(np.array(one_to_one_points[i])-np.array(prelim_pick))
            if dist < smallest_dist and dist < max_jump:
                smallest_dist = dist
                one_to_one_candidates.append(np.array(one_to_one_points[i]))
                template_pick, query_pick = one_to_one_points[i][1], one_to_one_points[i][0]  #template_pick could be pre_query_pick
                
        template_picks, query_picks = [template_pick], [query_pick]    
    
    #Calculate and plot the velocities and errors.
    template_vel = \
               calculate_velocities(template_picks, path_length, length_error)
    query_vel = \
               calculate_velocities(query_picks, path_length, length_error)
    
    '''
    #Plot the final summary of the dtw picking process.
    template_query_stats = [np.mean(template_picks),np.min(template_picks),np.max(template_picks)]
    query_picks_stats = [np.mean(query_picks),np.min(query_picks),np.max(query_picks)]
    dtw_plot([template_trace, template_trace_env], [query_trace, query_trace_env], template_times,
             query_times, [dtw_template, dtw_template_env], [dtw_query, dtw_query_env],
             manual_picking=False, template_picks=template_query_stats, query_picks=query_picks_stats)
    '''

    if not manual and manual_guidance and np.abs(np.mean(template_picks)-np.mean(query_picks)) > max_jump:
        print('Picking off course. Going to manual.')
        return prev_temp_pick, prev_query_pick, None, None, False        
    
    return np.mean(template_picks), np.mean(query_picks), template_vel, query_vel, True


def onpick(event):
    '''
    Subroutine to pick values from the active plot
    '''
    
    global COORDS
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    
    COORDS.append((xdata[ind], ydata[ind]))
    return COORDS

def get_index(times, search_time, threshold=0.01):
    '''
    Get the index of the given search_time in time
    '''
    
    min_ind = np.argmin(np.abs(times-search_time))
    if np.abs(times-search_time)[min_ind] < threshold:     
        return min_ind
    print('Could not find min index')

def manual_window_pick(amplitude, time):
    '''
    Function to plot the waveforms for manual selection of the
    picking windows. Returns the times and corresponding indices
    of the picked windows.
    '''
    
    global COORDS

    print('Choose the beginning and end that you want to compare')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click on points')
    COORDS = []
    ax.plot(time, amplitude, '-', picker=5), plt.xlabel('Time ($\mu$s)')
    plt.grid(color='0.5')
    cid = fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()
    
    start_time = np.min(COORDS[0][0])
    end_time = np.min(COORDS[1][0])
    
    return get_windowed_data(amplitude, time, start_time, end_time)
    
def get_windowed_data(amplitude, time, start_time, end_time, norm_factor=None):
    '''
    Return the trace amplitudes and times sliced by the window
    start and end times.
    '''
    start_ind = np.min(np.where(time>=start_time))
    end_ind = np.min(np.where(time>=end_time))
    
    windowed_trace = amplitude[start_ind:end_ind]
    if norm_factor:
        windowed_trace = windowed_trace/norm_factor
    else:
        windowed_trace = windowed_trace/np.max(np.abs(windowed_trace))
    
    windowed_time = time[start_ind:end_ind]
    
    return windowed_trace, windowed_time



def do_dtw(template, query, template_times, query_times, alpha=0, plot=True):
    '''
    Function to calculate the dynamic time warping mapping between
    two waveforms. The template and query amplitudes and times are
    passed as parameters. The function returns the dtw mapping as two
    time arrays. For each entry in these arrays, the template time is
    the same position on the waveform as the query time at the same index
    in the array.
    '''

    dist, query_inds, template_inds, costs = dtw(query, template, alpha=alpha)
    del costs
    
    # The lower the distance of alignment, the better the match
    #print('Distance of the DTW algorithm: {:.3f}'.format(dist))  

    dtw_template = np.array([template[int(template_inds[i]-1)] for i in range(len(template_inds))])
    dtw_query = np.array([query[int(query_inds[i]-1)] for i in range(len(query_inds))])
    dtw_template_times = np.array([template_times[int(template_inds[i]-1)] for i in range(len(template_inds))])
    dtw_query_times = np.array([query_times[int(query_inds[i]-1)] for i in range(len(query_inds))])

    if plot:
        plot_dtw_waveform_comp(template, query, template_times, query_times, 
                               dtw_template, dtw_query, dtw_template_times, dtw_query_times)

    return dtw_template_times, dtw_query_times


def plot_dtw_waveform_comp(template, query, template_times, query_times,
                           dtw_template, dtw_query, dtw_template_times,
                           dtw_query_times):
    '''
    Function to plot the two windowed waveforms, along with lines connecting
    those waveforms which visualise the dtw mapping.
    '''    
    
    plt.figure('DTW Points of Match', figsize=(9,5))
    
    #Plot the waveforms
    plt.plot(template_times, template, label='2 MPa', c='b')
    plt.plot(query_times, query, label='12 MPa', c='g')
    
    #Plot the matching points
    for i in np.arange(0,len(dtw_template)-1,1):
        x_points = [dtw_template_times[i],dtw_query_times[i]]
        y_points = [dtw_template[i],dtw_query[i]]
        if i % 2:
            plt.plot(x_points, y_points, 'r-', lw=0.5)
    
    plt.axis('tight')
    plt.legend()
    plt.xlabel('Time ($\mu$s)'), plt.yticks([])
    plt.savefig('dtw_example.pdf', bbox_inches='tight')

    plt.show()   
    

def dtw_plot(template_waveforms, query_waveforms, template_times,
             query_times, dtw_template_times, dtw_query_times,
             manual_picking=False, template_picks=None,query_picks=None):
    '''
    This function plots the dtw mapping in a template time vs query time
    plot for one or more dtw mappings (e.g. waveform and envelope). The 
    Template and waveform plots are shown alongside the mapping plot
    for reference. The mapping plot can be used to pick a wave arrival
    by sleceting a range of points around mappings that fit the physics
    of the experiment (e.g. pick where the mapping function has a gradient
    of 1, showing a constant time shift). Picking is allowed if manual_picking
    is True. template_picks and query_picks are three-element lists of
    [mean pick, min pick, max pick] to display if desired.
    
    Arguments:
        --template_waveforms: A list of amplitude arrays corresponding to the
                template waveforms
        --query_waveforms: A list of amplitude arrays corresponding to the
                template waveforms
        --template_times: An array of times for the template waveforms
        --query_times: An array of times for the query waveforms
        --dtw_template_times: A list of arrays which give the template time
                points for the dtw mapping(s).]
        --dtw_query_times: A list of arrays which give the query time
                points corresponding to the template time points for the dtw.
                
    Returns:
        None
    '''
    
    fig=plt.figure('DTW Mapping', figsize=(10, 10))
    
    left, width = 0.12, 0.60
    bottom, height = 0.08, 0.60
    bottom_h =  0.16 + width
    left_h = left + 0.27
    rect_plot = [left_h, bottom, width, height]
    rect_x = [left_h, bottom_h, width, 0.2]
    rect_y = [left, bottom, 0.2, height]

    axplot = plt.axes(rect_plot)
    axx = plt.axes(rect_x, sharex=axplot)
    axy = plt.axes(rect_y, sharey=axplot)
    
    colors = ['r','m','b','m']
    ls = ['-','--']
    start_time = min(template_times[0], query_times[0])   #max for tighter bounds
    end_time = max(template_times[-1], query_times[-1])   #min for tighter bounds
    num_of_waveforms = len(template_waveforms)

    # Plot the dtw mapping lines
    i=0
    for template, query in zip(dtw_template_times, dtw_query_times):
        picker = 5. if i==0 else None
        axplot.plot(query,template,colors[i],picker=picker,lw=2,ls=ls[i%2])
        i+=1
    axplot.axis([start_time,end_time,start_time,end_time]) #Give same time scale as template

    #Define and plot the 1:1 line of match
    x1=np.linspace(0,end_time,10)
    axplot.plot(x1,x1,'g',lw=2) #plot the 1:1 line of match

    axplot.plot(*smoothed_gradient_av(dtw_template_times[0], dtw_query_times[0]),'oy',lw=2)

    # Plot template waveforms
    for i in range(num_of_waveforms):
        axy.plot(template_waveforms[i],template_times,colors[i],lw=2,ls=ls[i%2])    
    #axy.axis([-1.1,1.1,start_time,end_time])
    axy.invert_xaxis()
    
    # Plot query waveforms
    for i in range(num_of_waveforms):
        axx.plot(query_times,query_waveforms[i],colors[i+num_of_waveforms],lw=2,ls=ls[i%2])
    #axx.axis([start_time,end_time,-1.3,1.3])

    title = 'DTW Picking Sumamry'
    if manual_picking:
        
        global COORDS
        #Pick the times from the graph
        title = 'Click on Arrival Time Points'
    
        #Global variable for storing the picked coordinates 
        COORDS=[]
        fig.canvas.mpl_connect('pick_event', onpick)
    
    if template_picks:
        for ax in [axplot, axy]:
            ax.axhspan(template_picks[-2],template_picks[-1], alpha=0.5, color='r')
            ax.axhline(template_picks[0],xmin=-1.3,xmax=1.3,linewidth=2, color='r')
    if query_picks:
        for ax in [axplot, axx]:
            ax.axvspan(query_picks[-2],query_picks[-1], alpha=0.5, color='b')
            ax.axvline(query_picks[0],ymin=-1.3,ymax=1.3,linewidth=2, color='b')
            
    cursor = Cursor(axplot, axx, axy)
    plt.connect('motion_notify_event', cursor.mouse_move)
    
    axplot.set_title(title)        
    plt.show(block=True)
    

def smoothed_gradient_av(x, y):
    
    grads = np.asarray(np.diff(y)/np.diff(x))  # Either 1., 0., or inf due to the nature of dtw
    good_indices = []
    win = 2
    
    for i in range(len(grads)-win):
        for i1 in range(win):
            if not 0.<grads[i+i1]<2.:
                break
            elif i1 == win-1:
                good_indices.append(i)
                    
    return y[good_indices], x[good_indices]

def calculate_velocities(picks, path_length, length_error):
    '''
    Function to calculate the velocity from arrival time
    picks.
    
    Arguments:
        --picks: A distribution (list) of arrival time picks
        --path_length: The propagtaion distance of the waveform.
        --length_error: The error in path_length
        
    Returns:
        --vel: The best estiamte of the velocity
    '''
    
    pick_length_mc = None
    if len(np.unique(picks)) > 1:
        vel = path_length * 1000. / np.mean(picks)
    else:
        vel = path_length * 1000 / picks[0]
        vel_err = vel * (length_error/path_length)

    return vel
    

def lag_time_plot(dtw_template, dtw_query, **kwargs):
    '''
    Function to plot the lag times obtained from the dtw mapping.
    The lag times are defined by the distance from each dtw mapping
    point to the line y=x. Since the dtw mapping is not a function f(x)
    in general (i.e. an x value can appear more than once), this lag time
    definition is more a representation of the time differences between the
    template and query. For example, if the dtw says the template leads the
    query by 2s at a time of t=1s, then this lead will appear as a sqrt(2)
    lead at a time of t=2s. A positive lag means the tempalte leads (query lags),
    and a negative lag means the query leads (template lags).
    
    Arguments:
        --dtw_template: The template array for the dtw mapping
        --dtw_query: The query array for the dtw mapping
        --**kwargs: The keyword arguments for plotting
        
    Returns:
        None
    '''
    
    diff = dtw_query - dtw_template    
    lag_times = dtw_template + (diff / 2.)
    lags = diff# * np.sqrt(2.) / 2  # Distance to the line y=x
    
    ind = np.where(lag_times>7.5)[0][0]
    print('Mean lag time from dtw: {} +- {}'.format(round(np.mean(lags[ind:-100]),2),
                                               round(np.std(lags[ind:]),2)))
    
    from plotting import simple_plot_points
    fig = simple_plot_points(lag_times, [0]*len(lag_times), linestyle='-', color='r', show=False)
    fig = simple_plot_points(lag_times, lags, show=True, linestyle='', marker='o',
                             fig=fig, color='b', ylab='Lag (s)', xlab='Time (s)',markersize=2.)
    
    return fig
    
    
    
    

global COORDS
COORDS = []
font = {'size' : 18}
plt.rc('font', **font)

if __name__ == '__main__':
    pick_vp_dtw()