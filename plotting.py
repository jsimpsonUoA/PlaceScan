'''
Python functions to facilitate plotting of scan data. These functions
are used mostly in conjunction with PlaceScan.

Jonathan Simpson, jsim921@aucklanduni.ac.nz
Masters project, PAL Lab UoA, 26/03/18
'''

import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def format_fig(fig, ax, title, xlab=None, ylab=None, xlim=None, ylim=None, save_dir=None,
               show=True, lab_font=16.0, title_font=20.0, tick_fontsize=14.0,
               grid=False, legend=False, legend_loc='upper right'):
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
        --title_font: The title font size
        --tick_fontsize: The axis tick fontsize
        --grid: Show a grid on the plot

    Returns:
        --fig: The figure isntance
        --ax: The axis instance
    '''

    if title:
        plt.title(title, fontsize=title_font)
    
    if xlab:
        ax.set_xlabel(xlab, fontsize=lab_font)
    if ylab:
        ax.set_ylabel(ylab, fontsize=lab_font)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim) 

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

    if grid:
        ax.set_rlabel_position(340)
        ax.yaxis.get_major_locator().base.set_params(nbins=6)
        ax.grid(which='major', linestyle=':')

    if legend:
        plt.legend(loc=legend_loc,fontsize=tick_fontsize)

    if save_dir:
        if isinstance(save_dir, list):
            for _dir in save_dir:
                fig.savefig(_dir, bbox_inches='tight')
        else:
            fig.savefig(save_dir, bbox_inches='tight')
    
    if show:
        plt.show()

    return fig, ax



def wiggle_plot(values, times, x_positions, fig=None, figsize=(8,6),amp_factor=8.0,
                tmax=1e20, title=None, xlabel='Position ($^\circ$)', show=True,
                save_dir=False, plot_picks_dir=None, pick_errors=None):
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
        --title: A title for the plot
        --xlabel: The x label for the plot
        --show: Set to True to show the plot
        --save_dir: The directory (or list of directories) to save the figure to.
        --plot_picks_dir: The directory where wave arrival picks are saved
        --pick_errors: Which wave arrival pick errors to plot.
                       One of 'both', 'early', 'late', or None.
        
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
        ax.plot(data, times, color='black', linewidth=0.5)
        ax.fill_betweenx(times, data, x_positions[i], where=data>x_positions[i],
                         color='black', rasterized=False)

    if plot_picks_dir:
        plot_picks(ax, plot_picks_dir, pick_errors)
    
    maxx = x_positions[-1]+(x_positions[-1]+x_positions[0])*0.01
    minx = x_positions[0]-(x_positions[-1]+x_positions[0])*0.01

    fig, ax = format_fig(fig, ax, title, xlabel, 'Time ($\mu$s)', (minx, maxx),
                    (min(times[-1], tmax), 0.0), save_dir, show)

    return fig



def variable_density(values, times, x_positions, fig=None, figsize=(9,7), gain=0.0,
                tmax=1e20, title=None, xlabel='Position ($^\circ$)', show=True,
                save_dir=False, plot_picks_dir=None, pick_errors=None):
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
        --title: A title for the plot
        --xlabel: The x label for the plot
        --show: Set to True to show the plot
        --save_dir: The directory (or list of directories) to save the figure to.
        --plot_picks_dir: The directory where wave arrival picks are saved
        --pick_errors: Which wave arrival pick errors to plot.
                       One of 'both', 'early', 'late', or None.
        
    Returns:
        --fig: The figure instance
    '''
    
    if not fig:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    else:
        ax = fig.gca()
    
    vmax, vmin = np.amax(values), np.amin(values)
    extent=(x_positions[0], x_positions[-1], times[-1], times[0]) 
    data = np.flip(np.rot90(values,k=3),1) * (gain/10.0+1.0)
    plt.imshow(data, extent=extent, vmax=vmax, vmin=vmin,
                cmap='binary', aspect='auto')

    if plot_picks_dir:
        plot_picks(ax, plot_picks_dir, pick_errors)

    fig, ax = format_fig(fig, ax, title, xlabel, 'Time ($\mu$s)', 
                      (x_positions[0], x_positions[-1]), 
                         (min(times[-1], tmax), 0.0), save_dir, show)

    return fig


def all_traces(values, times, fig=None, figsize=(9,7),
                tmax=1e20, title=None, ylabel='Amplitude', show=True,
                save_dir=False, plot_picks_dir=None, pick_errors=None, **kwargs):
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
        --ylabel: The y label for the plot
        --show: Set to True to show the plot
        --save_dir: The directory (or list of directories) to save the figure to.
        --plot_picks_dir: The directory where wave arrival picks are saved
        --pick_errors: Which wave arrival pick errors to plot.
                       One of 'both', 'early', 'late', or None.
        --**kwargs: The keyword arguments for the plotting
        
    Returns:
        --fig: The figure instance

    '''

    if not fig:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    else:
        ax = fig.gca()

    for value in values:
        ax.plot(times, value, **kwargs)

    if plot_picks_dir:
        plot_picks(ax, plot_picks_dir, pick_errors)

    fig, ax = format_fig(fig, ax, title, 'Time ($\mu$s)', ylabel,
                            (0.0, min(times[-1], tmax)), None, save_dir, show)

    return fig   
    

def picks_from_csv(filename):
    '''
    Function to retrieve wave arrival picks from a csv file.
    
    Arguments:
        --directory: The filename of the picks csv
        
    Returns:
        --headers: The column headings in the picks file
        --columns: The columns in the csv file
    '''
        
    with open(filename, 'r') as file:
        
        reader = list(csv.reader(file))
        data = np.zeros((len(reader)-1,len(reader[0])))
        
        for i in range(len(reader[1:])):
            data[i] = np.array([float(item) for item in reader[i+1]])
            
    return reader[0], data.transpose()
    

def plot_picks(ax, picks_dir, which_errors, color='r', polar=False,
               sample_diameter=None, label='', **kwargs):
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
        --**kwargs: The keyword arguments for the plotting
        
    Return:
        --ax: The axis with the picks plotted
    '''    

    headers, picks_data = picks_from_csv(picks_dir)
    x_pos = picks_data[0]
    indices = [np.where(picks_data[i] != -1.) for i in range(1,picks_data.shape[0])]
    if sample_diameter:
        picks_data[1:] = sample_diameter/picks_data[1:]
    if polar:
        x_pos = np.deg2rad(x_pos)
    
    p_picks = picks_data[1]
    ax.plot(x_pos[indices[0]], p_picks[indices[0]], linewidth=0.8, color=color,
            label=label, **kwargs)

    if which_errors == 'early' or which_errors == 'both':
        ax.fill_between(x_pos[indices[1]], picks_data[2][indices[1]],
               p_picks[indices[1]], color=color, alpha=0.2)
    if which_errors == 'late' or which_errors == 'both':
        ax.fill_between(x_pos[indices[2]], picks_data[3][indices[2]], 
                p_picks[indices[2]], color=color, alpha=0.2)    
    
    return ax
    

def arrival_times_plot(picks_dirs, polar=True, fig=None, figsize=(8,8),
                       title='', show=True, save_dir=None, pick_errors=None,
                       labels=None, legend_loc=None, **kwargs):
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
    
    colors = ['r','b','g','y']
    for i in range(len(picks_dirs)):
        plot_picks(ax, picks_dirs[i], pick_errors, color=colors[i], polar=polar,
                   label=labels[i], **kwargs)
    
    fig, ax = format_fig(fig, ax, title, save_dir=save_dir, tick_fontsize=12.0,
                         show=show, grid=polar, legend=len(labels)>0, legend_loc=legend_loc)

    return fig      