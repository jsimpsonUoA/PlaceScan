'''
Python functions to facilitate plotting of scan data. These functions
are used in conjunction with PlaceScan, but some could be used stand-alone.

Jonathan Simpson, jsim921@aucklanduni.ac.nz
Masters project, PAL Lab UoA, 26/03/18
'''

import numpy as np
import warnings
from scipy.signal import detrend as dt_func
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap
from mtspec import wigner_ville_spectrum, mtspec
import csv
import inspect

global LATEX_FORMAT
LATEX_FORMAT = False

def format_fig(fig, ax, title=None, xlab=None, ylab=None, xlim=None, ylim=None, 
               save_dir=None, show=True, lab_font=20.0, lab_color='k', title_font=20.0, 
               tick_fontsize=14.0, grid=False, legend=False, legend_loc='lower left',
               legend_fontsize=12.0, polar=False, polar_min=0, colorbar=True, 
               color_key=None, color_label='', color_data=None, color_data_labels=None,
               colorbar_trim=[0,1], colorbar_round=1, colorbar_horiz=False, lines=None,
               color_log=False):
    '''
    General function to format and save a figure.

    Arguemnts:
        --fig: The figure instance
        --ax: The current axis
        --title: The title of the plot
        --xlab: The x label for the axis
        --ylab: The y label for the axis
        --xlim: The x limit tuple
        --ylim: The y limit tuple
        --save_dir: The directory (or list of dirs) to save the figure to.
        --show: Set to True to show the plot
        --lab_font: The axis label fontsize
        --lab_color: The axis labels color
        --title_font: The title font size
        --tick_fontsize: The axis tick fontsize
        --grid: Show a grid on the plot
        --polar: True if this is a polar plot
        --polar_min: The minimum theta axis label for a polar plot. Either 
                   0 or -180.
        --colorbar: True to plot a colorbar, if color_key is specified.
        --color_[key, label, data, data_labels, bar_trim, bar_round, bar_horiz, log]: 
            The arguments for creating a sequential color bar for the plotted data.
            See create_colorbar function for more details.
        --lines: The lines plotted on the axis

    Returns:
        --fig: The figure isntance
        --ax: The axis instance
    '''

    if title:
        fig.suptitle(title, fontsize=title_font)
    
    try:
        ax.yaxis.major.formatter.set_powerlimits((-5,5))
        ax.xaxis.major.formatter.set_powerlimits((-5,5))
    except:
        pass
    
    if xlab:
        ax.set_xlabel(xlab, fontsize=lab_font, color=lab_color)
    if ylab:
        ax.set_ylabel(ylab, fontsize=lab_font, color=lab_color)
        
    if xlim or polar_min:
        if polar:
            
            labels = np.linspace(0.,360., 8, endpoint=False)
            if polar_min == -180:
                labels = np.where(labels>xlim[1]+1.,labels-360.,labels)
            elif polar_min:
                labels = np.where(labels>polar_min+360.,labels-360.,labels)
            if not plt.rcParams['text.usetex']:
                ax.set_xticklabels(['{}°'.format(s) for s in labels.astype(int)])
            else:
                ax.set_xticklabels(['${}^\\circ$'.format(s) for s in labels.astype(int)])
        else:
            ax.set_xlim(xlim)
    #if ylim != None:
    try:
        ax.set_ylim(ylim) 
    except TypeError:
        pass

    if tick_fontsize:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

    if grid:
        if polar:
            ax.set_rlabel_position(340) 
            ax.yaxis.get_major_locator().base.set_params(nbins=6)
            ax.grid(which='major', linestyle=':')
        else:
            ax.grid()

    if legend:
        ax.legend(loc=legend_loc,fontsize=legend_fontsize)

    if polar:
        #ax.set_rlabel_position(275.0)  Commented on 10/12/19
        ax.set_rlabel_position(23.0)

    if color_key:
        fig, ax = create_colorbar(fig, ax, color_key, color_label,
             color_data, lab_font, tick_fontsize, color_data_labels,
             colorbar_trim, colorbar, colorbar_round, colorbar_horiz,
             lines, color_log)
        plt.sca(ax)

    if save_dir:
        print('saving')
        if isinstance(save_dir, list):
            for _dir in save_dir:
                fig.savefig(_dir, bbox_inches='tight')
        else:
            fig.savefig(save_dir, bbox_inches='tight',dpi=300)
    
    if show:
        plt.show()
        plt.close()

    return fig, ax

def create_colorbar(fig, ax, color_key, color_label, color_data,
                    lab_font, tick_fontsize, color_data_labels=None,
                    colorbar_trim=[.2,.9], colorbar=True, colorbar_round=1,
                    colorbar_horiz=False, lines=None,color_log=False):
    '''
    Function which is called within format_fig
    to color-code plotted lines by a 3rd variable.
    This function also handles the formatting of
    the colorbar

    Arguments:
        --fig: The matplotlib figure
        --ax: The matplotlib axis containing the lines
        --color_key: The color of the shaded colorbar
            Accepted values are 'grey', 'purple',
            'green', 'blue', orange', or 'red', or
            any matplotlib colorbar key.
        --color_label: A label for the colorbar axis
        --color_data: The 3rd varialbe to color code the
            plotted lines by. This can be a tuple of two
            numbers representing the lower and upper extremes,
            assuming the lines between are evenly spaced in
            this range. Alternatively, this can be a list
            of the same length as the number of plotted
            lines, containing numbers which will be used
            to assign colors to the lines. A list of numbers
            represented as strings is also acceptable.
        --lab_font: The axis label fontsize
        --tick_fontsize: The axis tick fontsize
        --color_data_labels: A  list containing tick labels for the
            colorbar, if something different from the automatic 
            ticklabels based on color_data is required. These are evenly
            spaced along the colorbar
        --colorbar_trim: A color map usually spans the range 0.0-->1.0
            If a smaller range is desired for better visibility etc.,
            then this can be given here as a two element list specifying
            the lower and upper bounds of the colors, e.g. [0.1,0.9]
        --colorbar: True to plot the colorbar, False to hide it
        --colorbar_round: The number of decimal places to round the 
                          colorbar axis labels to
        --colorbar_horiz: True to have a horizontal colorbar
        --lines: The specific lines to apply the color scale to.
        --color_log: True to make the color bar a logarithmic scale. False
            gives a linear scale.

    Returns:
        --fig: The figure with the colorbar plotted
    '''

    if not lines:
        lines = ax.lines

    #Find the upper and lower values for the colormap
    if isinstance(color_data, tuple):
        color_data = np.linspace(color_data[0], color_data[1], len(lines))
    else:
        if isinstance(color_data[0], str):
            color_data = np.array([float(num) for num in color_data])
    max_n, min_n = max(color_data), min(color_data)
    data_range = max_n - min_n
    interval = colorbar_trim[1] - colorbar_trim[0]
    data_midpoint = data_range / 2 + min_n
    interval_midpoint = interval / 2 + colorbar_trim[0]
    vmax = data_range / interval
    shift = (interval_midpoint * vmax) - data_midpoint
    vmax = vmax - shift
    vmin = 0.0 - shift
    #Create the colormap
    if color_key in ['grey', 'purple','green', 'blue', 'orange','red']:
        cmap = get_cmap(color_key.capitalize()+'s')
    else:
        cmap = get_cmap(color_key)
    if color_log:
        norm = mpl.colors.LogNorm(vmin=min_n, vmax=max_n)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    #Color-code the plot
    for i in range(len(color_data)):
        if i < len(lines):
            lines[i].set_color(cmap(int(255*norm(color_data[i]))))

    #Only add the colorbar if there is not already one on the plot.
    #Note: This conditional may not behave as expected if Axes objects
    #are manually added elsewhere.
    add_axes = np.sum([isinstance(i,mpl.axes.Axes) for i in fig.get_children()]) < 2
    if colorbar and add_axes:
        #Rearrange the axes
        ax_pos = list(ax.get_position().bounds)
        ax_dim_to_use = 2 + int(colorbar_horiz)
        ax_size =  ax_pos[ax_dim_to_use]         #The width or height
        ax_pos[ax_dim_to_use] = ax_size * 0.91   #Shrink width or height
        ax.set_position(ax_pos)                     

        #Setup and insert the colorbar axis
        cbar_width = ax_size * (0.03 * (1+colorbar_horiz))
        if colorbar_horiz:
            cbar_pos, orien = [ax_pos[0], ax_pos[1]+ax_size-cbar_width, ax_pos[2], cbar_width], 'horizontal' 
        else:
            cbar_pos, orien = [ax_pos[0]+ax_size-cbar_width, ax_pos[1], cbar_width, ax_pos[3]], 'vertical' 
        cbar_ax = plt.axes(cbar_pos)
    
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        cb = plt.colorbar(mappable=mappable,cax=cbar_ax,boundaries=np.arange(min_n,max_n,data_range/1000), orientation=orien)
        
        #Format the colorbar
        if not isinstance(color_data_labels, type(None)):
            ticks = mpl.ticker.FixedLocator(np.linspace(min_n,max_n-data_range*0.0015, len(color_data_labels)))
            cb.set_ticks(ticks) 
            cb.set_ticklabels(color_data_labels)
        else:
            ticks = mpl.ticker.FixedLocator(np.linspace(min_n,max_n-data_range*0.0015, 5))
            cb.set_ticks(ticks) 
            if colorbar_round:
                cb.set_ticklabels(np.round(np.linspace(min_n,max_n, 5),colorbar_round))
            else:
                cb.set_ticklabels(np.linspace(int(min_n),int(max_n), 5,dtype=int))   #Integers
        cb.ax.tick_params(labelsize=tick_fontsize)
        cb.set_label(color_label, fontsize=lab_font, rotation=int(not colorbar_horiz)*270.,verticalalignment='bottom',labelpad=colorbar_horiz*10) 
        if colorbar_horiz:
            cb.ax.xaxis.tick_top(), cb.ax.xaxis.set_label_position('top')

    return fig, ax


def get_plot_data(data, times, sampling_rate, tmax=None, tmin=None, normed=False, 
                    bandpass=None, dc_corr_seconds=None, decimate=False, differentiate=False,
                    taper=None, detrend=True):
        '''
        A function to apply signal processing and other
        routines to scan data before plotting. Assumes
        the data is four-dimensional.
        
        Arguments:
            --data: The data to process. This is a two or three-dimensional array.
                If three dimensions, the array is flattened into two.
            --times: An array of times with the same length as the last dimension of data.
            --sampling_rate: The sampling rate of the data in Hz
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
            --taper: A taper to apply to a section of the data. See the apply_taper
                function.
            --detrend: True to detrend the data
            
        Returns:
            --data: The plotting values
            --times: The plotting times
        '''

        plot_times = times.copy()
        plot_data = data.copy()  
        s = plot_data.shape
        if len(s) > 2:
            plot_data = plot_data.reshape(s[0]*s[1],s[2])
        
        if detrend:
            # Detrending
            try:
                plot_data = dt_func(plot_data)    
            except ValueError:
                print('Detrending error warning. Still proceeding')
        
        if differentiate:
            from scipy.integrate import cumtrapz
            plot_data = cumtrapz(plot_data,x=plot_times,axis=1)
            plot_times = plot_times[:-1]
            plot_data = cumtrapz(plot_data,x=plot_times,axis=1)
            plot_times = plot_times[:-1]
            
        if bandpass:
            plot_data = bandpass_filter(plot_data, bandpass[0], bandpass[1], sampling_rate)

        if detrend:
            # Detrending
            plot_data = dt_func(plot_data)  

        if tmax:
            ind = np.where(plot_times <= tmax)[0][-1]
            plot_times = plot_times[:ind+1]
            plot_data = plot_data[:,:ind+1]
        if tmin != None:
            ind = np.where(plot_times >= tmin)[0][0]
            plot_times = plot_times[ind:]
            plot_data = plot_data[:,ind:] 

        if detrend:
            # Detrending
            plot_data = dt_func(plot_data)  

        if normed:
            plot_data = normalize(plot_data, normed)
        
        if dc_corr_seconds:
            plot_data = np.array([array - np.mean(array[:int(sampling_rate * dc_corr_seconds)])
                                        for array in plot_data])
    
        if decimate:
            if bandpass:
                factor = int(sampling_rate // bandpass[1] / 2)
            else:
                factor = int(decimate)
            plot_data = plot_data[:,::factor]
            plot_times = plot_times[::factor]

        if taper:
            plot_data, plot_times = apply_taper(taper, plot_data, plot_times)

        return plot_data, plot_times


def wiggle_plot(values, times, x_positions, fig=None, figsize=(8,6),amp_factor=8.0,
                tmax=1e20, xlab=r'$\theta$ (degrees)', ylab=r'Time ($\mu$s)', 
                plot_picks_dir=None, pick_errors=None, picks_offset=0., xlim=None, 
                **kwargs):
    '''
    Function to plot a wiggle plot of a laser ultrasound scan
    
    Arguments:
        --values: A array/list of arrays containing the waveforms
        --times: An array of times against which the values are plotted
        --x_positions: The x-axis locations to plot each wiggle at
        --fig: The figure to plot the wiglle plot on
        --figsize: The figure size of the plot
        --amp_factor: Factor to multiply values by when plotting
        --tmax: The maximum time to plot for
        --xlab: The x label for the plot
        --ylab: The ylabel for the plot
        --plot_picks_dir: The directory where wave arrival picks are saved
        --pick_errors: Which wave arrival pick errors to plot.
                       One of 'both', 'early', 'late', or None.
        --picks_offset: The amount to offset the arrival time picks from the
                 saved positions
        --xlim: The limits for the x axis of the plot
        --**kwargs: Keyqord arguments for figure formatting/saving (see format_fig)
        
    Returns:
        --fig: The figure instance
    '''
    
    if not fig:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    else:
        ax = fig.gca()
    
    for i in range(len(values)):
        data = values[i]*amp_factor+x_positions[i]
        ax.plot(data, times, color='black', linewidth=1., marker='')
        ax.fill_betweenx(times, data, x_positions[i], where=data>x_positions[i],
                         color='black', rasterized=False)

    if plot_picks_dir:
        if isinstance(plot_picks_dir,list):
            [plot_picks(ax, _dir, pick_errors, picks_offset=picks_offset, marker='') for _dir in plot_picks_dir]
        else:
            plot_picks(ax, plot_picks_dir, pick_errors, picks_offset=picks_offset, marker='')
    
    if not xlim:
        maxx = x_positions[-1]+(x_positions[-1]+x_positions[0])*0.01
        minx = x_positions[0]-(x_positions[-1]+x_positions[0])*0.01
        xlim = (minx, maxx)

    fig, ax = format_fig(fig, ax, xlab=xlab, ylab=ylab, 
                         xlim=xlim, ylim=(min(times[-1], tmax), 0.0), **kwargs)

    return fig



def variable_density(values, times, x_positions, fig=None, figsize=(9,7), gain=0.0,
                tmax=1e20, xlab='Position (degrees)', plot_picks_dir=None, 
                pick_errors=None, picks_offset=0., xlim=None, colorbar=False, 
                color_map='seismic', data_limits=None, **kwargs):
    '''
    Function to plot a variable density wiggle plot of a laser ultrasound scan
    
    Arguments:
        --values: A array/list of arrays containing the waveforms
        --times: An array of times against which the values are plotted
        --x_positions: The x-axis locations to plot each wiggle at
        --fig: The figure to plot the wiglle plot on
        --figsize: The figure size of the plot
        --gain: The gain for the traces
        --tmax: The maximum time to plot for
        --xlab: The x label for the plot
        --plot_picks_dir: The directory where wave arrival picks are saved
        --pick_errors: Which wave arrival pick errors to plot.
                       One of 'both', 'early', 'late', or None.
        --picks_offset: The amount to offset the arrival time picks from the
                 saved positions
        --xlim: The x axis limits of the data
        --colorbar: True to show a color scale of the amplitudes
        --color_map: The matplotlib color map to use for the vd plot
        --data_limits: A custom 2-element list of [min_val,max_val] to 
            manually set the limits of the scale for plotting. If None,
            the minimum and maximum data values are used.
        --**kwargs: Keyqord arguments for figure formatting/saving (see format_fig)
        
    Returns:
        --fig: The figure instance
    '''
    
    if not fig:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    else:
        ax = fig.gca()
    
    if not data_limits:
        vmax, vmin = np.amax(values), np.amin(values)
    else:
        vmax, vmin = data_limits[1], data_limits[0]
    max_val = max(np.abs(vmax),np.abs(vmin))
    extent=(x_positions[0], x_positions[-1], times[-1], times[0]) 
    data = np.rot90(values,k=3)[:,::-1] * (gain/10.0+1.0)
    plt.imshow(data, extent=extent, vmax=max_val, vmin=-max_val,
                cmap=color_map, aspect='auto')

    if plot_picks_dir:
        plot_picks(ax, plot_picks_dir, pick_errors, marker='', color='black', picks_offset=picks_offset)

    if not xlim:
        xlim=(x_positions[0], x_positions[-1])

    color_key, color_data = None, None
    if colorbar:
        color_key = color_map
        if not data_limits:
            color_data = [-max_val,max_val]
        else:
            color_data = data_limits

    fig, ax = format_fig(fig, ax, xlab=xlab, ylab='Time ($\mu$s)', 
                    xlim=xlim, ylim=(min(times[-1], tmax), 0.0), 
                    color_key=color_key, color_data=color_data, **kwargs)

    return fig


def all_traces(values, times, fig=None, figsize=(8,6),
                tmax=1e20, title=None, ylab='Amplitude', xlab='Time ($\mu$s)', show=True,
                save_dir=False, plot_picks_dir=None, pick_errors=None,
                picks_offset=None, legend=False, show_orientation=False, 
                 position=None, inset_params=None, marker='',
                 xlim=None, pick_marker_kwargs=None, **kwargs):
    '''
    Function to plot all the traces given in values as a time series
    on the same axis

    Arguments:
        --values: A array/list of arrays containing the waveforms
        --times: An array of times against which the values are plotted
        --fig: The figure to plot the wiglle plot on
        --figsize: The figure size of the plot
        --tmax: The maximum time to plot for
        --title: A title for the plot
        --ylab: The y label for the plot
        --xlab: The x label for the plot
        --show: Set to True to show the plot
        --save_dir: The directory (or list of directories) to save the figure to.
        --plot_picks_dir: The directory where wave arrival picks are saved. This
            can also be a list of equal length to the number of traces in values,
            where the picks directories for each trace are given. The pick at 
            the location given in position is plotted. Position may also be a
            list of positions, one for each trace.
        --pick_errors: Which wave arrival pick errors to plot.
                       One of 'both', 'early', 'late', or None.
        --picks_offset: The amount to offset the arrival time picks from the
                 saved positions
        --legend: True to plot a legend for plotted traces.
        --show_orientation: Plot the orienation of an anisotropic rock.
                  show_orientation is the dictionary of kwargs for the plotting
                  function.
        --position: Position of the trace, if a single trace is plotted. Can
            also be a list of positions if multiple traces are plotted.
        --inset_params: A dictioanry containing parameters for a zoomed inset
        --marker: A marker to plot with
        --xlim: The x axis limits for the plot
        --pick_marker_kwargs: Properties for the pick markers (dictionary of
            kwargs)
        --**kwargs: The keyword arguments for the plotting
        
    Returns:
        --fig: The figure instance

    '''

    if not fig:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    else:
        ax = fig.gca()

    plot_kwargs = kwargs.copy()
    [plot_kwargs.pop(key) for key in list(plot_kwargs.keys()) if key in inspect.getargspec(format_fig)[0]]
    format_dict = kwargs.copy()     
    [format_dict.pop(key) for key in list(format_dict.keys()) if key not in inspect.getargspec(format_fig)[0]]
    
    lines = []
    try:
        dummy_var = len(values[0])
    except TypeError:
        values = [values]

    for value in values:
        lines.append(ax.plot(times, value, marker=marker, **plot_kwargs))

    if plot_picks_dir:
        if isinstance(plot_picks_dir, list):
            for i in range(len(plot_picks_dir)):
                if isinstance(position, list):
                    plot_picks(ax, plot_picks_dir[i], pick_errors, picks_offset=picks_offset, 
                                line=lines[i], position=position[i], **pick_marker_kwargs)
                else:
                    plot_picks(ax, plot_picks_dir[i], pick_errors, picks_offset=picks_offset, 
                                line=lines[i], position=position, **pick_marker_kwargs)
        else:
            plot_picks(ax, plot_picks_dir, pick_errors, picks_offset=picks_offset, line=lines[0], 
                                position=position, **pick_marker_kwargs)
    
    if show_orientation:    #Very specific application for rock anisotropy. Remove if not needed.
        fig = plot_anisotropy_reference(fig, angle=position, **show_orientation)
        
    if not xlim:
        xlim = (max(0.0,times[0]), min(times[-1], tmax))
    fig, ax = format_fig(fig, ax, title=title, xlab=xlab, ylab=ylab,
                        xlim=xlim, save_dir=None, show=False, legend=legend, **format_dict)  

    if inset_params:
        ax = zoomed_inset(fig, ax, **inset_params)

    # Just for saving
    format_fig(fig, ax, save_dir=save_dir, show=False, tick_fontsize=None)

    if show:        
        plt.show()
    
    return fig   
    

def picks_from_csv(filename, s_waves=False, s_wave_key=None):
    '''
    Function to retrieve wave arrival picks from a csv file.
    
    Arguments:
        --directory: The filename of the picks csv
        --s_waves: True if the picks are in the s-wave picks format
        --s_wave_key: The header key for the type of s_wave, if
            s_wave is True
        
    Returns:
        --headers: The column headings in the picks file
        --columns: The columns in the csv file
    '''
    
    try:
        with open(filename, 'r') as file:
            
            reader = list(csv.reader(file))
            
            if not s_waves:
                data = np.zeros((len(reader)-1,len(reader[0])))
                for i in range(len(reader[1:])):
                    data[i] = np.array([float(item) for item in reader[i+1]])

            else:
                data = np.zeros((len(reader)-1,4))
                pick_index = reader[0].index(s_wave_key)
                for i in range(len(reader[1:])):
                    row = reader[i+1]
                    pressure = float(row[0].split()[0])
                    pick = float(row[pick_index])
                    err = float(row[pick_index+1])
                    early_err, late_err = pick-err, pick+err
                    data[i] = np.array([pressure, pick, early_err, late_err])

        return reader[0], data.transpose()
    except:
        scan = filename[:filename.rfind('/')]
        scan = scan[scan.rfind('/')+1:]
        print('PlaceScan Plot Picks: Cannot find arrival time csv for {}. Arrival times will not be plotted.'.format(scan))
        return None, None
    

def plot_picks(ax, picks_dir, which_errors, color='r', polar=False,
               sample_diameter=None, label='', picks_offset=0., 
               picks_xlim=None, linewidth=1.5, s_waves=False, 
               s_wave_key=None, picks_x=False, line=None, 
               position=0., **kwargs):
    '''
    Function to plot the wave arrival picks.
    
    Arguments:
        --ax: The axis to plot onto
        --picks_dir: The directory where wave arrival picks are saved
        --which_errors: Which wave arrival pick errors to plot.
                        One of 'both', 'early', 'late', or None.
        --color: The plotting color
        --polar: Whether or not the ax are a polar axis.
        --sample_diameter: The diameter of the sample, if conversion to m/s is 
                    desired. Units of diameter must be in um.
        --label: A name for the arrival picks line
        --picks_offset: The amount to offset the arrival time picks from the
                 saved positions
        --picks_xlim: A tuple of the minimum and maximum x positions to plot picks for
        --linewidth: The width of the line to plot
        --s_waves: True if the picks are in the s-wave picks format
        --s_wave_key: The header key for the type of s_wave, if
            s_wave is True
        --picks_y: True if time is on the x axis
        --line: A line2D object to plot a signle arrival time pick on
            as a vertical bar.
        --position: If line is given, this is the position in the picks
            file that the line corresponds to.
        --**kwargs: The keyword arguments for the plotting
        
    Return:
        --ax: The axis with the picks plotted
        --x_pos: The x position of the plotted pick
        --line: The line plotted on the axis
    '''    

    headers, picks_data = picks_from_csv(picks_dir, s_waves=s_waves, s_wave_key=s_wave_key)
    
    x_pos = None
    if headers:
        x_pos = picks_data[0] + picks_offset
        if not picks_xlim:
            indices = [np.where(picks_data[i] != -1.)  for i in range(1,picks_data.shape[0])]
        else:
            indices = [np.where((picks_data[i2] != -1.) & (x_pos > picks_xlim[0])\
                & (x_pos < picks_xlim[1])) for i2 in range(1,picks_data.shape[0])]
        if sample_diameter:
            picks_data[1:] = sample_diameter*1000./picks_data[1:]
        if polar:
            x_pos = np.deg2rad(x_pos)
        
        p_picks = picks_data[1]
        if not picks_x and not line:
            line = ax.plot(x_pos[indices[0]], p_picks[indices[0]], linewidth=linewidth, color=color,
                    label=label, **kwargs)[0]
            error_function = ax.fill_between
        elif not line:
            line = ax.plot(p_picks[indices[0]], x_pos[indices[0]], linewidth=linewidth, color=color,
                    label=label, **kwargs)[0]
            error_function = ax.fill_betweenx
        else:
            pos = np.argmin(np.abs(x_pos-position))
            ax = pick_marker(ax, p_picks[pos], line=line, line_height=0.2, **kwargs)
    
        if which_errors == 'early' or which_errors == 'both':
            error_function(x_pos[indices[1]], picks_data[2][indices[1]],
                p_picks[indices[1]], color=color, alpha=0.2)              
        if which_errors == 'late' or which_errors == 'both':
            error_function(x_pos[indices[2]], picks_data[3][indices[2]], 
                    p_picks[indices[2]], color=color, alpha=0.2)    
    
        return ax, x_pos[indices[0]], line, [picks_data[1][indices[0]],picks_data[2][indices[1]],picks_data[3][indices[2]]]
    
    return ax, None, line, None

def arrival_times_plot(picks_dirs, polar=True, fig=None, figsize=(8,8),
                       title='', show=True, save_dir=None, pick_errors=None,
                       labels=None, legend_loc=None, color='orange', 
                       xlim=None, markers=None, colors=None, **kwargs):
    '''
    Function create a plot of just the arrival time picks
    
    Arguments:
        --picks_dirs: The directory or directories where the picks are saved
        --polar: Whether or not to create a polar plot
        --fig: The figure to plot the wiglle plot on
        --figsize: The figure size of the plot
        --title: A title for the plot
        --show: Set to True to show the plot
        --save_dir: The directory (or list of directories) to save the figure to.
        --pick_errors: Which wave arrival pick errors to plot.
            One of 'both', 'early', 'late', or None.
        --labels: List of labels for the legend
        --legend_loc: The position of the legend
        --color: A color for the sequential colormap. Can be 'grey', 'purple',
            'green', 'blue', orange', or 'red'.
        --xlim: The x limits for the plot
        --markers: A set of markers if more than one picks_dir is given
        --colors: A set of colors if more than one picks_dir is given
        --**kwargs: The keyword arguments for the plotting
        
    Returns:
        --fig: The figure with the plot
    '''
    
    if not fig:
        fig = plt.figure(figsize=figsize)
        if polar:
            ax = plt.subplot(111, projection='polar')
        else:
            ax = plt.subplot(111)
    else:
        ax = fig.gca()   
    
    if not isinstance(picks_dirs, list):
        picks_dirs = [picks_dirs]
    
    plot_kwargs = kwargs.copy()
    [plot_kwargs.pop(key) for key in list(plot_kwargs.keys()) if key in inspect.getargspec(format_fig)[0]]
    format_dict = kwargs.copy()     
    [format_dict.pop(key) for key in list(format_dict.keys()) if key not in inspect.getargspec(format_fig)[0]]
    
    if not labels:
        labels = ['']*len(picks_dirs)
    if not markers:
        markers = ['x','.','+','d','*','s']
    if not colors:
        colors = ['#d94901', '#2272b5','r','#6a52a3','g','y','c','b']
    lines, all_picks = [], []
    for i in range(len(picks_dirs)):
        ax, x_pos, line, picks = plot_picks(ax, picks_dirs[i], pick_errors, color=colors[i%len(colors)], polar=polar,
                   label=labels[i], marker=markers[i%len(markers)], **plot_kwargs) 
        lines.append(line)
        all_picks.append(picks)
    
    if not xlim:
        xlim=(x_pos[0],x_pos[-1])

    fig, ax = format_fig(fig, ax, title, save_dir=save_dir, show=show, grid=polar,
                         legend_loc=legend_loc, polar=polar, lines=lines,
                         xlim=xlim,**format_dict)

    return fig, all_picks     


def spread_waveform_plot(values, times, indep_var, fig=None, figsize=(8,6),
                amp_factor=2.0,tmax=1e20, xlab=r'Time ($\mu$s)', ylab='Pressure (MPa)', 
                time_vertical=False, picks_to_plot=[], labels=None, picks_label=None, 
                ylim=None, **kwargs):
    '''
    Plots waveforms as a function of pressure (or
    any other varibale for that matter). Essentially
    a sideways wiggle plot with a customized y-axis
    to plot waveforms against any variable desired.

    Arguments:
        --values: A array/list of arrays containing the waveforms
        --times: An array of times against which the values are plotted
        --indep_var: A list of independent variables (floats) to plot
            position each trace on the y-axis (e.g. a list of pressures 
            for each trace)
        --fig: The figure to plot the wiglle plot on
        --figsize: The figure size of the plot
        --amp_factor: Factor to multiply values by when plotting
        --tmax: The maximum time to plot for
        --xlab: The x label for the plot
        --ylab: The y label for the plot
        --time_vertical: True to plot the time on the y axis of the plot
        --picks_to_plot: A list of (x,y) tuples corresponding to the
            arrival time picks to plot. The spread values must
            not be added to the y values.
        --picks_label: A labels for a legend of the picks line
        --ylim: The ylimits of the plot
        --**kwargs: Keyqord arguments for figure formatting/saving 
            (see format_fig and matplotlib plot)

    Returns:
        --fig: The figure instance with the plot
    '''

    if not fig:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    else:
        ax = fig.gca()

    plot_kwargs = kwargs.copy()
    [plot_kwargs.pop(key) for key in list(plot_kwargs.keys()) if key in inspect.getargspec(format_fig)[0]]
    format_dict = kwargs.copy()     
    [format_dict.pop(key) for key in list(format_dict.keys()) if key not in inspect.getargspec(format_fig)[0]]
    
    if not labels:
        labels = [None]*len(values)

    for i, value in enumerate(values):
        plot_kwargs['label'] = labels[i]
        x, y = times, value*amp_factor+indep_var[i]
        if time_vertical:
            x, y = value*amp_factor+indep_var[i], times
        ax.plot(x, y, **plot_kwargs)

    if len(picks_to_plot) > 0:
        x_vals, y_vals = tuple(zip(*picks_to_plot))
        y_vals = [y_vals[i]*amp_factor+indep_var[i] for i in range(len(y_vals))]
        x, y = x_vals, y_vals
        if time_vertical:
            x, y = y_vals, x_vals
        ax.plot(x, y, lw=1.5,c='k',marker='', label=picks_label)
        if picks_label:
            format_dict['legend']=True
    
    xlim, ylim = (max(0.0,times[0]),min(times[-1], tmax)), ylim
    if time_vertical:
        xlim, ylim = ylim, (min(times[-1], tmax),max(0.0,times[0]))
        xlab, ylab = ylab, xlab
    fig, ax = format_fig(fig, ax, xlab=xlab, ylab=ylab, xlim=xlim, ylim=ylim, **format_dict)  

    return fig


def simple_plot_points(x, y, fig=None, errors=None, figsize=(8,6),
                        shaded_errors=False, **kwargs):
    '''
    A simple function to plot a series of (x, y) points.
    
    Arguments:
        --x: The x data
        --y: The y data
        --fig: A Figure
        --errors: Some errors in the y data to plot error bars for
        --figsize: The size of the figure
        --shaded_errors: True to plot shaded regions to denote the
            errors instead of error bars.
        --**kwargs: The keyword arguments for plotting
        
    Returns:
        None
    '''
    
    if not fig:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    else:
        ax = plt.gca()
    
    plot_kwargs = kwargs.copy()
    [plot_kwargs.pop(key) for key in list(plot_kwargs.keys()) if key in inspect.getargspec(format_fig)[0]]
    format_dict = kwargs.copy()     
    [format_dict.pop(key) for key in list(format_dict.keys()) if key not in inspect.getargspec(format_fig)[0]]    
    
    if isinstance(errors, type(None)) or shaded_errors:
        line = ax.plot(x, y, **plot_kwargs)
    elif not shaded_errors:   #For stick error bars
        plt.errorbar(x, y, yerr=errors, **plot_kwargs)
    
    if shaded_errors:  #For shaded regions of errors
        ax.fill_between(x, y, y-errors, color=line[0].get_color(), alpha=0.2)
        ax.fill_between(x, y, y+errors, color=line[0].get_color(), alpha=0.2)
         
    fig, ax = format_fig(fig, ax, **format_dict)
    
    return fig


def simple_plot_series(xs, ys, fig=None, labels=None, figsize=(8,6),
                **kwargs):
    '''
    A simple function to plot multiple set of 
    simple data points

    Arguments:
        --xs: A list of lists, containing the x points for teh series
        --ys: A list of lists, with the y points of the series
        --fig: A Figure instance to plot onto
        --labels: The labels for each data series
        --figsize: The size of teh Figure
        --kwargs: The keyword arguments for the plotting

    Returns:
        None
    '''

    if not fig:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    else:
        ax = plt.gca()
    
    plot_kwargs = kwargs.copy()
    [plot_kwargs.pop(key) for key in list(plot_kwargs.keys()) if key in inspect.getargspec(format_fig)[0]]
    format_dict = kwargs.copy()     
    [format_dict.pop(key) for key in list(format_dict.keys()) if key not in inspect.getargspec(format_fig)[0]]    
    
    for i, x in enumerate(xs):
        ax.plot(x, ys[i], label=labels[i], **plot_kwargs)

    fig, ax = format_fig(fig, ax, **format_dict)
    
    return fig


def wv_spect(values, times, fig=None, figsize=(8,10), trace_ylab=None, 
             ylab='Frequency (kHz)', sampling_rate=5e6, max_freq=None, 
                min_freq=.0, dbscale=False, vmin=.0, vmax=1., taper=False, 
                number_of_plots=2, **kwargs):
    '''
    Function to plot a multi-taper Wigner-Ville spectrogram of time series
    data.
    
    Arguments: 
        --values: A 1-D array containing 
        --times: An array of times against which the values are plotted
        --fig: The figure to plot the wiglle plot on
        --figsize: The figure size of the plot
        --trace_ylab: The y-axis label for the time series data
        --ylab: The y label for the plot
        --sampling_rate: The sampling rate of the time series data
        --max_freq: The maximum frequency to calculate the spectrogram for.
        --min_freq: The minimum frequency to calculate the spectrogram for.
        --dbscale: Either False (linear colour map), 'min' (decibel scale using
                   the minimum value as a reference), or 'median' (decibel scale
                   using the median power as the reference)
        --vmin: The minimum of the colour map dynamic range (fraction between 0 and 1)
        --vmax: The maximum of the colour map dynamic range (fraction between 0 and 1)
        --taper: The length of seconds at the start of the trace to taper with a
                 Hanning window.
        --number_of_plots: The number of plots to be contained on the figure.
                 Usually 2.
        --**kwargs: Keyqord arguments for figure formatting/saving (see format_fig)
                 
    Returns:
        --None (by default)
        --WV spectrum (if wv_only=True)search_win=[10.0,15.5]
    '''
    
    if taper:
        mute_ind = np.where(times <= taper)[0][-1]
        window = np.hanning(mute_ind*2)[:mute_ind]
        values[:mute_ind] = np.multiply(values[:mute_ind],window)
    
    if not max_freq:
        max_freq = sampling_rate / 2.
    
    wv = wigner_ville_spectrum(values, 1/sampling_rate, 1, smoothing_filter='gauss')
        
    frequency_divider = sampling_rate / 2. / max_freq
    if frequency_divider > 1.1:
        wv = wv[-int((wv.shape[0]-1.0)//frequency_divider):,:]
    if min_freq:
        wv = wv[:-int(min_freq*(wv.shape[0]-1.0)//max_freq),:] 
    
    if dbscale == 'median':
        wv = 10 * np.log10(np.abs(wv)/np.median(np.abs(wv)))
    elif dbscale == 'min':
        wv = 10 * np.log10(np.abs(wv)/np.amin(np.abs(wv)))
    else:
        wv = np.where(wv > 0., wv, 0.) #OR: np.sqrt(np.abs(wv))
    
    if not fig:
        fig = plt.figure(figsize=figsize) 
        
    if number_of_plots > 2:
        gs = gridspec.GridSpec(number_of_plots, 2, height_ratios=[2,3]+[2]*(number_of_plots-2),
                               width_ratios=[30,1])
    else:
        gs = gridspec.GridSpec(2, 2, height_ratios=[2,3], width_ratios=[30,1])

    ax1 = fig.add_subplot(gs[0])     
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    if number_of_plots > 2:
        [fig.add_subplot(gs[i], sharex=ax1) for i in range(4, number_of_plots*2-1,2)]
    
    ax1.plot(times, values, 'k', marker='')          #Plot the waveform
    ax1.xaxis.set_tick_params(direction='in') 
    
    extent = (times[0], times[-1], min_freq/1e3, max_freq/1e3)
    im = ax2.imshow(wv, interpolation='nearest', aspect='auto', extent=extent,
                    cmap="nipy_spectral", vmin=vmin)#, vmax=vmax) 
    
    cbar_axes = fig.add_subplot(gs[3])
    units_list = ['','(dB)']
    cb = plt.colorbar(im, cax = cbar_axes)
    cbar_axes.set_ylabel('Power {}'.format(units_list[dbscale!=False]),fontsize=16.0)
    cbar_axes.yaxis.set_label_position('right')
    cbar_axes.tick_params(labelsize=12.0)
    cb.formatter.set_powerlimits((-3,3)), cb.update_ticks()
    
    xlab = 'Time ($\mu$s)' if number_of_plots < 3 else ''
    format_kwargs = {key:val for key ,val in kwargs.items() if key not in ['show','save_dir']}
    fig, ax1 = format_fig(fig, ax1, ylab=trace_ylab, show=False, save_dir=False,
                         xlim=(times[0], times[-1]), **format_kwargs)
    
    kwargs.update({'title':None})
    fig, ax2 = format_fig(fig, ax2, xlab=xlab, ylab=ylab,
                         xlim=(times[0], times[-1]), **kwargs)

    return fig, wv


def animate_plots(plotting_functions, func_kwargs=None, update_interval=2,
                  repeat_delay=0., show=True, figsize=(8,6), titles=None, 
                  save_dir=None,forward_frame_num=False):
    '''
    Function to create an animation between different plots.
    the results can be saved as a gif.
    
    Arguments:
        --plotting_functions: A callable or list of callables to call
                 at each update of the plot
        --func_kwargs: A dictionary or list of dictionaries of keyword arguments
                       for the callables in plotting_functions
        --update_interval: The interval between animation updates (in seconds)
        --repeat_delay: The time delay (in seconds) between repeats of the animation
        --show: True to display the plot
        --figsize: The size of hte figure
        --titles: A list of titles to display for each update
        --save_dir: The directory to save the GIF to.
        --forward_frame_num: True to give the first argument of the plotting
            function as the frame number.
        
    Returns:
        None    
    '''
    
    def update(i, fig, plotting_functions, func_kwargs, titles):
        fig.clear()
        index = i % len(plotting_functions)
        if titles:
            func_kwargs[index]['title'] = titles[index]
        if forward_frame_num:
            plotting_functions[index](i,**func_kwargs[index])
        else:
            plotting_functions[index](**func_kwargs[index])
        fig.canvas.draw()
        
    
    import matplotlib.animation as animation
    
    num = len(plotting_functions)
    
    if not isinstance(plotting_functions, list):
        plotting_functions = [plotting_functions]
    if func_kwargs and not isinstance(func_kwargs, list):
        func_kwargs = [func_kwargs] * len(plotting_functions)
    if len(func_kwargs) > 1 and len(func_kwargs) != num:
        print('PlaceScan Animate Plots: The number of function keyword dictionaries\
                            does not match the number of plotting functions')
        return
    if titles and len(titles) < num:
        titles.append(['']* (len(titles) - num))
    
    fig = plt.figure(figsize=figsize)
    
    [kwargs.update({'fig':fig}) for kwargs in func_kwargs]
    [kwargs.update({'show':False}) for kwargs in func_kwargs]
    [kwargs.update({'save_dir':None}) for kwargs in func_kwargs]
    
    anim = animation.FuncAnimation(fig, update, interval=update_interval*1000, save_count=num,
                         repeat_delay=repeat_delay*1000, repeat=True, frames=range(num),
                         fargs=(fig, plotting_functions, func_kwargs, titles))
    
    if save_dir:
        if str(save_dir).find('mp4') != -1:
            #plt.rcParams['animation.ffmpeg_path'] = '/snap/bin/ffmpeg'
            writer = animation.FFMpegWriter(fps=1/update_interval, bitrate=1800, codec='ffmpeg', extra_args=['-vcodec', 'libx264'])
            anim.save(save_dir, dpi=80, writer=writer)
        else:
            writer = 'imagemagick'
            anim.save(save_dir, dpi=80, writer=writer, fps=1/update_interval)
        
    if show:
        plt.show()
    
    [kwargs.pop('fig') for kwargs in func_kwargs if 'fig' in kwargs]
    
    return fig
    
    
    
def plot_anisotropy_reference(fig, angle=0., initial_angle=0.0, clockwise=False,
                              scale=1., pos=(.5,.1)):
    '''
    Auxilliary function to plot a cartoon of the angular position
    of an anisotropic rock sample relative to the seismic ray
    through the rock. This is plotted as an inset axis on the provided
    ax. 0 degrees is slow direction, 90 is fast.
    
    Arguments:
        --angle: The angle of the rock layering
        --initial_angle: The initial angular position of the rock.
        --clockwise: True for clockwise rotation, False for anticlockwise
        
    Returns:
        --cartoon: The cartoon, read to plot on an axis.
    '''
    
    ax = plt.gca()
    
    height = scale*.1
    width = height*2.2
    
    cartoon_ax = fig.add_axes([pos[0], pos[1], width, height], frameon=False)
    num_lines = 5                # Number of bands
    t = 1 / (num_lines * 2 + 1)  # Thickness of Bands
    
    # Create the Cartoon with shapely
    circle = sg.Point(.5,.0).buffer(.5)
    lines = [sg.Polygon([(2*i*t+t,-.5),(2*i*t+t+t,-.5),(2*i*t+t+t,.5),(2*i*t+t,.5)])
                                  for i in range(num_lines)]
    lines = [circle.intersection(line) for line in lines]
    lines = [af.rotate(line, (-2*clockwise+1)*(angle+initial_angle), origin=(.5,.0))
                                  for line in lines]
    source = sg.Polygon([(-.6,-.02),(.1,-.02),(.1,.02),(-.6,.02)])
    rec = sg.Polygon([(.9,-.02),(1.6,-.02),(1.6,.02),(.9,.02)])
    
    # Plot the patches
    cartoon_ax.add_patch(descartes.PolygonPatch(source, fc='g',linewidth=0.))
    cartoon_ax.add_patch(descartes.PolygonPatch(rec, fc='r',linewidth=0.))
    cartoon_ax.add_patch(descartes.PolygonPatch(circle, fc='#c3c3c3'))
    [cartoon_ax.add_patch(descartes.PolygonPatch(line, fc='#6d6d6d',linewidth=0.))
                                   for line in lines]
    
    # Configure the axes
    cartoon_ax.set_xlim(-.6, 1.6); cartoon_ax.set_ylim(-.6, .6)
    cartoon_ax.set_xticks([]), cartoon_ax.set_yticks([])
    cartoon_ax.set_aspect('equal'), cartoon_ax.patch.set_alpha(0.0)
    
    plt.sca(ax)
    
    return fig
    

def psd_spectrum(values, times, sampling_rate, figsize=(8,9), tmax=1e20, 
                 title=None, ylab='Amplitude', plot_picks_dir=None, plot_trace=True,
                 pick_errors=None, max_freq=2000, fig=None, normalise_spectra=False,
                 min_freq=0., linear_scale=False, **kwargs):
    '''
    Function to plot the spectral power of a trace calculated using
    the scipy periodogram function

    Arguments:
        --values: A array of the amplitude values
        --times: An array of times against which the values are plotted
        --sampling_rate: The sampling rate of the data in seconds
        --figsize: The figure size of the plot
        --tmax: The maximum time to plot for
        --title: A title for the plot
        --ylab: The y label for the plot
        --plot_picks_dir: The directory where wave arrival picks are saved
        --plot_trace: True to plot the trace as well as the PSD
        --pick_errors: Which wave arrival pick errors to plot.
                       One of 'both', 'early', 'late', or None.
        --max_freq: The maximum frequency to plot for, in kHz
        --fig: A fig to plot on
        --normalise_spectra: True to plot each spectral line between 0 and 1.
        --min_freq: The minimum frequency to plot
        --linear_scale: True to plot the PSD on a linear scale
        --**kwargs: The keyword arguments for the plotting
        
    Returns:
        --fig: The figure instance

    '''
    from scipy.signal import periodogram

    units = 'nm$^2$/Hz'

    if plot_trace:
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=figsize, gridspec_kw={'height_ratios':[1, 2]})
        plt.sca(ax1)
    elif not fig:
        fig, ax2 = plt.subplots(1,1, figsize=figsize)
    else:
        ax2 = plt.gca()
    
    plot_kwargs = kwargs.copy()
    [plot_kwargs.pop(key) for key in list(plot_kwargs.keys()) if key in inspect.getargspec(format_fig)[0]]
    format_dict = kwargs.copy()     
    [format_dict.pop(key) for key in list(format_dict.keys()) if key not in inspect.getargspec(format_fig)[0]]
    
    if plot_trace:
        format_kwargs = {key:val for key ,val in format_dict.items() if key not in ['show','save_dir']}
        fig = all_traces([values], times, fig=fig,tmax=tmax, title=title, ylab=ylab, show=False, **plot_kwargs,**format_kwargs)
    
    #Calculate and plot the multitaper spectrum
    nex_pow2 = np.ceil(np.log2([len(values)])[0])
    freq, spec = periodogram(values, fs=sampling_rate, nfft=2**int(nex_pow2))
    freq /= 1e3
    freq, spec = freq[2:], spec[2:]   # Remove the first couple of samples, which are usually very negative
    freq_inds = np.where(freq > min_freq)
    freq, spec = freq[freq_inds], spec[freq_inds]
    freq_inds = np.where(freq < max_freq)
    freq, spec = freq[freq_inds], spec[freq_inds]
    if normalise_spectra:
        if isinstance(normalise_spectra, float):
            print('ok')
            spec /= normalise_spectra
        else:
            spec /= np.max(spec)
        units = 'arb. units'
    if not linear_scale:
        line = ax2.semilogy(freq, spec, **plot_kwargs)
    else:
        line = ax2.plot(freq, spec, **plot_kwargs)

    if plot_picks_dir:
        plot_picks(ax1, plot_picks_dir, pick_errors)

    xlim=(min_freq, min(sampling_rate/2,max_freq))

    fig, ax = format_fig(fig, ax2, title=None, xlab='Frequency (kHz)', ylab='Power Spectral Density ({})'.format(units),
                            xlim=xlim, **format_dict)
    
    return fig   

    
def multitaper_spect(values, times, sampling_rate, figsize=(8,9), tmax=1e20, 
                 title=None, ylab='Amplitude', plot_picks_dir=None, plot_trace=True,
                 pick_errors=None, max_freq=2000, fig=None, normalise_spectra=False,
                 plot_uncertainties=False, linear_scale=False, min_freq=0, **kwargs):
    '''
    Function to plot the spectral power of a trace calculated using
    a multitaper spectrum.

    Arguments:
        --values: A array of the amplitude values
        --times: An array of times against which the values are plotted
        --sampling_rate: The sampling rate of the data in seconds
        --figsize: The figure size of the plot
        --tmax: The maximum time to plot for
        --title: A title for the plot
        --ylab: The y label for the plot
        --plot_picks_dir: The directory where wave arrival picks are saved
        --plot_trace: True to plot the trace as well as the PSD
        --pick_errors: Which wave arrival pick errors to plot.
                       One of 'both', 'early', 'late', or None.
        --max_freq: The maximum frequency to plot for, in kHz
        --fig: A fig to plot on
        --normalise_spectra: True to plot each spectral line between 0 and 1.
        --plot_uncertainties: True to plot the uncertainties for the spectra.
        --min_freq: The minimum frequency to plot
        --linear_scale: True to plot the PSD on a linear scale
        --**kwargs: The keyword arguments for the plotting
        
    Returns:
        --fig: The figure instance

    '''

    units = 'nm$^2$/Hz'

    if plot_trace:
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=figsize, gridspec_kw={'height_ratios':[1, 2]})
        plt.sca(ax1)
    elif not fig:
        fig, ax2 = plt.subplots(1,1, figsize=figsize)
    else:
        ax2 = plt.gca()
    
    plot_kwargs = kwargs.copy()
    [plot_kwargs.pop(key) for key in list(plot_kwargs.keys()) if key in inspect.getargspec(format_fig)[0]]
    format_dict = kwargs.copy()     
    [format_dict.pop(key) for key in list(format_dict.keys()) if key not in inspect.getargspec(format_fig)[0]]
    
    if plot_trace:
        format_kwargs = {key:val for key ,val in format_dict.items() if key not in ['show','save_dir']}
        fig = all_traces([values], times, fig=fig,tmax=tmax, title=title, ylab=ylab, show=False, **plot_kwargs,**format_kwargs)
    
    #Calculate and plot the multitaper spectrum
    nex_pow2 = np.ceil(np.log2([len(values)])[0])
    spec, freq, jackknife, _, _ = mtspec(
        data=values, delta=1/sampling_rate, time_bandwidth=5.,
        statistics=True)#, nfft=2**int(nex_pow2))
    freq /= 1e3
    if normalise_spectra:
        if isinstance(normalise_spectra, float):
            print('ok')
            spec /= normalise_spectra
        else:
            spec /= np.max(spec)
        units = 'arb. units'
    if not linear_scale:
        line = ax2.semilogy(freq, spec, **plot_kwargs)
    else:
        line = ax2.plot(freq, spec, **plot_kwargs)

    if plot_uncertainties:
        ax2.fill_between(freq, jackknife[:, 0], jackknife[:, 1], alpha=0.1, color=line[0].get_color())

    if plot_picks_dir:
        plot_picks(ax1, plot_picks_dir, pick_errors)

    fig, ax = format_fig(fig, ax2, title=None, xlab='Frequency (kHz)', ylab='Power Spectral Density ({})'.format(units),
                            xlim=(min_freq, min(sampling_rate/2,max_freq)), **format_dict)
    
    return fig   
    

def cross_correlation(a, b, sampling_rate, fig=None, figsize=(8,6), plot=True, 
                        return_function=False, **kwargs):
    '''
    Function which performs the cross correlation of two
    traces in the time domain, and plots the result. A positive
    max_lag means that the first trace leads the second trace.
    A negative max_lag means the second trace leads the first.
    
    Arguments:
        --a: The first trace
        --b: The second trace
        --fig: A figure instance to plot onto
        --plot: True to create a plot of the correlation function
        --figsize: The size of the figure
        --return_function: True to return the cross-correlation function and
                time arrays
        --**kwargs: The keyword arguments for plotting and saving
        
    Returns:
        --fig: the figure with the plot
        --max_lag: The time at which the maximum correlation coefficient occurs
    '''
    
    from scipy.signal import correlate
    
    if not fig and plot:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    elif plot:
        ax = fig.gca()
    
    plot_kwargs = kwargs.copy()
    [plot_kwargs.pop(key) for key in list(plot_kwargs.keys()) if key in inspect.getargspec(format_fig)[0]]
    format_dict = kwargs.copy()     
    [format_dict.pop(key) for key in list(format_dict.keys()) if key not in inspect.getargspec(format_fig)[0]]

    norm_factor = (np.sum(a ** 2)) ** 0.5 * (np.sum(b ** 2)) ** 0.5
    corr = correlate(a-np.mean(a),b-np.mean(b)) / norm_factor
    times = np.arange(-np.floor(len(corr)/2),np.floor(len(corr)/2)+1) * 1. / sampling_rate * 1e6
    
    max_lag = times[np.argmax(corr)] * -1.  #Needs to be multiplied by -1 to get sign of max_lag correct

    if plot:
        ax.plot(times,corr, **plot_kwargs)
        ax.axvline(x=-1.*max_lag, linestyle='--', linewidth=1,color='r')
        ax.text(0.05,0.9, '$r_{max}$='+str(max_lag), transform=ax.transAxes)
        
        fig, ax = format_fig(fig, ax, xlab='Lag Time ($\mu$s)',xlim=(times[0],times[-1]), **format_dict)
    
    if not return_function:
        return fig, max_lag
    else:
        return fig, max_lag, corr, times
    
    
def zoomed_inset(fig, ax, region=[1,2,1,2], zoom=2, aspect=2, inset_loc=2,
                axis_ticks=False, manual_loc=None, manual_marks=None):
    '''
    Function to plot a zoomed inset on a plot. The region of
    data to be zoomed is specified, along with the amount of zoom,
    aspect ratio of the zoomed axis relative to the main axis, and
    the location of the inset.
    
    Arguments:
        --fig: The Figure to plot onto
        --ax: The main axis containing the original data
        --region: The region to zoom in (in data coordinates of the
                  main axis)
        --zoom: The amount of zoom
        --aspect: The aspect ratio of teh zoomed axis, relative to the
               aspect ratio of the main axis
        --inset_loc: The location to place the inset on the main axis
               1 is upper left, 2 is lower left, 3 is lower right, and
               4 is upper right
        --axis_ticks: False to hide the axis ticks of the inset, True
            to display the ticks, 'x' for x only, or 'y' for y only
        --manual_loc: Manual override of the inset position. Given
            list of two floats: [xloc, yloc] of (right, top)
        --manual_marks: Manual override of the marker lines which  
            denote the inset. A list of two integers from 1-4
            corresponding to the corners to use. Can also have
            -1 for no marker on a corner.
               
    Returns:
        --ax: The main axis, with the inset axis plotted on it
    
    '''
    
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    
    ax_xlow, ax_xhigh = ax.get_xlim()[0], ax.get_xlim()[1]
    ax_ylow, ax_yhigh = ax.get_ylim()[0],ax.get_ylim()[1]
    figW, figH = fig.get_size_inches()
    _, _, w, h = ax.get_position().bounds
    disp_ratio = (figH * h) / (figW * w)
    data_ratio = (ax_yhigh-ax_ylow) / (ax_xhigh-ax_xlow)
    ax_aspect = disp_ratio / data_ratio
    
    #Get the location for the inset axis
    if inset_loc < 3:
        xloc = ((ax_xhigh-ax_xlow)*.02+ax_xlow)+(region[1]-region[0])*zoom
    else:
        xloc = (ax_xhigh-ax_xlow)*.98+ax_xlow
    if 1 < inset_loc < 4:
        yloc = ((ax_yhigh-ax_ylow)*.02+ax_ylow)+(region[3]-region[2])*zoom*(.5+.5/aspect)
    else:
        yloc = ((ax_yhigh-ax_ylow)*.98+ax_ylow)+(region[3]-region[2])*zoom*(.5-.5/aspect)
    
    if not manual_loc:
        bbox = [xloc,yloc]
    else:
        bbox = manual_loc

    axins = zoomed_inset_axes(ax, zoom,axes_kwargs={'aspect':ax_aspect/aspect},bbox_transform=ax.transData,bbox_to_anchor=bbox, borderpad=0.0)  
    
    #Using a loop simply to utilise break functionality
    for dummy_var in '1':
        if not manual_marks or not isinstance(manual_marks,list) or len(manual_marks) != 2:
            mark1 = 2-inset_loc%2
            mark2 = 4-inset_loc%2
        else:
            if manual_marks[0] == None or manual_marks[1]==None:
                break
            mark1, mark2 = manual_marks[0], manual_marks[1]
        mark_inset(ax, axins, loc1=mark1, loc2=mark2, fc="none", ec="0.5")
    
    for line in ax.get_lines():
       new_line = axins.plot(line.get_xdata(),line.get_ydata(), marker='')
       new_line[0].set_linewidth(line.get_lw())
       new_line[0].set_ls(line.get_ls())
       new_line[0].set_color(line.get_color())
    
    axins.set_xlim(region[0], region[1])
    axins.set_ylim(region[2], region[3])
    
    if not axis_ticks or axis_ticks == 'x':
        axins.set_yticks([])
    if not axis_ticks or axis_ticks == 'y':
        axins.set_xticks([])
    axins.tick_params(axis='both',direction='in',labelsize=11.0)
    
    return ax
    
    
def get_integer_ticks(minv, maxv):
    '''
    Small function which tries to get integer ticks
    for an axis based on the max and min values
    '''

    acceptable_numbers = [3,4,5,6] #The acceptable number of ticks
    sig_digits = 3                 #Number of significant digits

    for num in acceptable_numbers[::-1]:
        if (maxv-minv) % (num-1):
            ticks = np.linspace(minv, maxv,num)
            break
    else:
        ticks = np.linspace(minv, maxv,4)
    
    for i, num in enumerate(ticks):
        num, counter = str(num), 0
        for char in num:
            if char not in '.0':
                break
            counter += 1
        num = num[:counter+sig_digits+1]
        ticks[i] = num

    return ticks

def pick_marker(ax, pick, line=None, line_height=0.2,
                absolute_height=None,marker='s',
                markersize=3.,**kwargs):
    '''
    Function to plot a vertical bar at the location
    of an arrival time pick on a waveform plot

    Arguments:
        --ax: The axis to plot the pick on
        --pick: The arrival time pick
        --line: The line2D object of the waveform which
            the pick refers to. If not given, the first
            plotted line on the axis will be used.
        --line_height: The height of the vertical marker
            bar (value between 0 and 1 where 1 is the full
            height of the axis)
        --marker: The type of marker to use
        --markersize: The size of the marker
        --kwargs: The keyword arguments for the marker
        
    Returns:
        --ax: The axis object        
    '''

    if not line:
        line = ax.lines[0]
    
    try:
        y_data, x_data = line.get_ydata(), line.get_xdata()
    except AttributeError:
        line = line[0]
        y_data, x_data = line.get_ydata(), line.get_xdata()
    
    line_color = line.get_color()
    if 'color' not in kwargs:
        if 'c' not in kwargs:
            m_color = line_color
        else:
            m_color = kwargs.pop('c')
    else:
        m_color = kwargs.pop('color')

    pick_index = np.argmin(np.abs(x_data-pick))
    pick_y = y_data[pick_index]

    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    x_range, y_range = x_max-x_min, y_max-y_min

    if not absolute_height:
        y_coord1 = pick_y - line_height/2 * y_range
        y_coord2 = pick_y + line_height/2 * y_range
    else:
        y_coord1 = pick_y - absolute_height/2
        y_coord2 = pick_y + absolute_height/2 

    ax.plot([pick]*2,[y_coord1, y_coord2], marker='',c='k',**kwargs)
    ax.plot(pick,y_coord2,marker=marker,markersize=markersize,color=m_color)

    return ax

def plot_arrow(fig, ax, base_pos, head_pos, width=1,
                head_length=.5, head_width=.5, head_angle=0, 
                color='k', show=False, save_dir=None):
    '''
    Function to plot an arrow on a figure

    Arguments:
        --fig: The figure to plot on
        --ax: The axis to plot onto
        --base_pos: An (x, y) tuple specifying the position
            of the arrow base, in data coordinates
        --head_pos: An (x, y) tuple specifying the position
            of the arrow head, in data coordinates
        --width: The width of the arrow (in plotting width)
        --head_length: The length of the arrow head, as a
            fraction of the total arrow length.
        --head_width: The width of the arrow head, as a fraction
            of the head_length
        --head_angle: The amount to sweep in the back of the
            arrow head, as a fraction between 0 and 1
        --color: The color of the arrow.
        --show: True to show the figure after the arrow is plotted.
        --save_dir: A directory to save the figure to afterward.

    Returns:
        --fig: The figure
        --ax: The axis
    '''

    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    corr = (xlims[1]-xlims[0])/(ylims[1]-ylims[0])

    x0, y0 = base_pos[0], base_pos[1]
    x1, y1 = head_pos[0], head_pos[1]
    angle = np.arctan((y1-y0)/((x1-x0)/corr))
    length = np.linalg.norm(np.array(head_pos)-np.array(base_pos))

    head_length = head_length*length
    head_width = head_width*head_length
    head_angle = head_angle%1.
    
    cosi, sini = np.cos(angle), np.sin(angle)
    base_x, base_y = x1-head_length*cosi*corr, y1-head_length*sini
    ar1_x, ar1_y = base_x+(sini*(head_width/2))*corr, base_y-cosi*head_width/2
    ar2_x, ar2_y = base_x-(sini*(head_width/2))*corr, base_y+cosi*head_width/2
    ar3_x, ar3_y = base_x+(head_angle*head_length)*cosi*corr, base_y+(head_angle*head_length)*sini

    plt.sca(ax)
    ax.plot([x0,ar3_x],[y0,ar3_y],color=color,lw=width,marker='')
    head = plt.Polygon([[ar1_x,ar1_y],[x1,y1],[ar2_x,ar2_y],[ar3_x,ar3_y]],color=color)
    
    ax.add_patch(head)

    # Just for saving
    format_fig(fig, ax, save_dir=save_dir, show=show, tick_fontsize=None)

    return fig, ax


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
        max_vals = [np.amax(np.abs(array)) for array in data]
        for i in range(len(max_vals)):
            if max_vals[i] == 0.:
                max_vals[i] = 1.
        return np.array([data[i]/max_vals[i] for i in range(len(data))])
    if mode == 'scan':
        return data/np.amax(np.abs(data))
    if isinstance(mode, (int,float)) and not isinstance(mode, bool):
        return data/mode
    return data
        
        
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
        
def apply_taper(taper, values, times):
    '''
    This function applies a taper to a section of plot data. 
    
    Arguments:
        --taper: is in the form of a two or three element list or tuple. 
            The first element is the start time of the taper (usually 0.), 
            the second element is the end time of the taper, and the optional 
            third element is the intensity of the taper. This must be a value 
            between 0 and 1, where 0 is the least aggressive tapering, and 1 
            is the most aggressive (1 essentially mutes all the data within 
            the tapering bounds).
        --values: The plotting data
        --times: The plotting times

    Returns:
        --values: The plotting values with taper applied
        --times: The plotting times
    '''

    try:
        start_i = np.argmin(np.abs(times-taper[0]))
        end_i = np.argmin(np.abs(times-taper[1]))
        full_size = end_i-start_i + 1 
        values_size = len(values[0])

        try:
            intensity = taper[2]
            if not (0. <= intensity <= 1.0):
                raise ValueError
        except IndexError:
            intensity = 0.

        intensity = 1. - intensity
        taper_size = int((full_size)*intensity)
        zero_size = full_size-taper_size
        taper = np.hanning(taper_size*2)[:taper_size]
        zeros = np.zeros(zero_size)
        front_ones  = np.zeros(start_i)+1.
        back_ones = np.zeros(values_size-1-end_i)+1.
        full_taper = np.concatenate((front_ones,zeros,taper,back_ones))

        for i in range(len(values)):
            values[i] = values[i]*full_taper
    except:
        print('PlaceScan: Invalid taper for plotting. No taper applied.')

    return values, times

def set_color_cycle(colors_=None,markers_=None, only_colors=True):
    '''
    Function to set the color cylcle
    '''
    from cycler import cycler

    if colors_ and markers_:
        mpl.rcParams['axes.prop_cycle'] = cycler(c=colors_,marker=markers_)
    elif colors_:
        mpl.rcParams['axes.prop_cycle'] = cycler(c=colors_,marker=markers)
    elif markers_:
        mpl.rcParams['axes.prop_cycle'] = cycler(c=colors,marker=markers_)
    elif not only_colors:
        mpl.rcParams['axes.prop_cycle'] = cycler(c=colors,marker=markers)
    elif only_colors:
        mpl.rcParams['axes.prop_cycle'] = cycler(c=colors)

def set_format(latex_on=True):
    """Set the latex formatting"""
    global LATEX_FORMAT
    LATEX_FORMAT = latex_on
    if LATEX_FORMAT:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    else:
        plt.rc('text', usetex=False)
        plt.rc('font', family='sans-serif')


#blue, red, Purple,  green, orange,  yellow
colors = ['#feaa38', '#2272b5','r', '#6a52a3','#248c45','#d94901']   #Another red: '#cc191d'  #['#6a52a3','#248c45']#
markers = ['s','o','^','d','x','+']
set_color_cycle()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

