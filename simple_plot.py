'''
Plot some simple plots of a scan using PlaceScan
'''
import glob
import numpy as np

import PlaceScan as ps
from PlaceScan import PlaceScan
import plotting as plot
import analysis as an

master_dir = '/mnt/office_machine/home/jsim921/Dropbox/jonathan_masters/data/'
pressures = ['1','2','4','6','8','10','12','14','16']
sample = 'pmv1'

scan_dirs = [master_dir+'{0}/{0}_{1}MPa/'.format(sample,pressure) for pressure in pressures]
#scan_dirs = [master_dir+'7f-schist-2/7f-schist-2_1MPa/', master_dir+'pmv1/pmv1_1MPa/', master_dir+'umv/umv_1MPa/', master_dir+'1a842/1a842_1MPa/']
#scan_dirs = glob.glob(master_dir+'time_delay_tests/*')

#labels = ['Transducers']+['Lasers']*6
labels = ['{} MPa'.format(pressure) for pressure in pressures]*2
fig_dir = '/mnt/office_machine/home/jsim921/Dropbox/jonathan_masters/figs/{}/'.format(sample)

scans = [PlaceScan(scan, scan_type='rotation', trace_field='ATS660-trace',apply_formatting=False,divide_energy=False) for scan in scan_dirs]
#scans += [PlaceScan(scan_dirs[0]+'/', scan_type='single', trace_field='MDO3014-trace',apply_formatting=False,divide_energy=False)]

offset = 0#-66
#[scan.reposition(x0=offset) for scan in scans]
#[scan.mute([(0,6),(156,170)],zero=True) for scan in scans]
#scans[-1].trace_data = scans[-1].trace_data *1.2 #For jacket control 14MPa
#[scan.change_polarity() for scan in scans]
#[scan.expand() for scan in scans]
#[scan.average_traces() for scan in scans]
#[scan.mute_by_signal(threshold=0.5,analog=True, max_analog_sig=1.033,sig_plot=True) for scan in scans]
size = len(scans[0].trace_data)
norm = max(scans[0].max_amp(),scans[-1].max_amp())
#scans[0].expand()
#cartoon_params = {'initial_angle':-40.0, 'scale':2, 'pos':(.53,.11)}

#  Wiggle Plots
figs = [scans[i].wiggle_plot(normed='scan', bandpass=(1e2,3e6), dc_corr_seconds=5e-6, tmax=40.0, amp_factor=20.0, figsize=(8,6), save_dir=None, decimate=False, show=False, save_ext='_present', plot_picks=False, pick_errors=None, xlab=r'$\theta$ (degrees)', picks_offset=0,tick_fontsize=16.0,lab_font=24.,title=labels[i],title_font=24.) for i in range(len(scans[:-1]))]
fig = scans[-1].wiggle_plot(normed='scan', bandpass=(1e2,3e6), dc_corr_seconds=5e-6, tmax=40.0, figsize=(8,6),amp_factor=20.0, save_dir=None, decimate=False, save_ext='_present',show=True, plot_picks=False, pick_errors=None, xlab=r'$\theta$ (degrees)',picks_offset=0,tick_fontsize=16.0,lab_font=24.,title=labels[len(scans)-1],title_font=24.) #labels[len(scans)-1]

#  Variable Density Plots
#[scans[i].variable_density(normed='scan', bandpass=(1e2,2e6), dc_corr_seconds=0e-6, tmax=15, gain=50, save_dir=None, plot_picks=True, pick_errors='both', show=False,title=labels[i])  for i in range(len(scans[:-1]))]
#scans[-1].variable_density(normed='scan', bandpass=(1e2,2e6), dc_corr_seconds=0e-6, tmax=15, gain=50, save_dir=None, plot_picks=True, pick_errors='both',title=labels[len(scans)-1]) 

#  Waveform Plots
#[scan.trace_plot(normed=True, bandpass=(5e4,8e6), dc_corr_seconds=4e-6, tmin=6.0, tmax=8.0, trace_index=None, trace_int=None, averaging=size, figsize=(8,4), save_dir=None, save_ext='stacked_100kHz', linewidth=1.0,show=False, ylim=(-.5,.5)) for scan in scans[:-1]]
#scans[-1].trace_plot(normed=None, bandpass=(1e2,3e6), dc_corr_seconds=0e-6, tmin=0.0, tmax=20.0, position=60.0, trace_int=None, trace_index=None, averaging=None, figsize=(10,6), save_dir=None, save_ext='', linewidth=1.5, show=False)
#scans[-1].trace_plot(normed=None, bandpass=None, dc_corr_seconds=0e-6, tmin=0.0, tmax=20.0, position=None, trace_int=None, trace_index=0, averaging=None, figsize=(10,6), save_dir=None, save_ext='', linewidth=1.5, show=True)

#  Plot the Arrival Time Picks
#[scan.arrival_picks_plot(normed=False, bandpass=(5e4,2e6), trace_int=None, averaging=size, figsize=(12,4), save_dir=None, save_ext='', linewidth=1.0,show=False) for scan in scans[:-1]]
#scans[-1].arrival_picks_plot(picks_dirs=scan_dirs, scans=scans,save_dir=None, save_ext='', pick_errors=None, polar=True, sample_diameter=24.56, picks_xlim=(0.,156.),picks_offset=offset,polar_min=0,ylim=(0.,6000.),labels=labels, legend_loc=(1.,-.2), tick_fontsize=14.0, figsize=(6,6),linestyle='')


#  Wigner-Ville spectrogram
#scans[-1].wigner_spectrogram(bandpass=(1e4, .7e6), tmax=100, trace_ind=25, average=False, min_freq=.0, dbscale=False, vmin=0., vmax=25., taper=5., show=False, save_dir=None, save_ext='')
#scans[-1].wigner_spectrogram(bandpass=None, tmax=40, trNoneace_index=None,position=10, average=False, min_freq=.0, dbscale=False, vmin=0., vmax=25., taper=2., show=True, save_dir=None, save_ext='')
#scans[-1].wigner_spectrogram(bandpass=(1e2,1e7), tmax=50, trace_index=None, position=None, average=True, min_freq=.0,dbscale=False, vmin=0., vmax=25., taper=2., show=True, save_dir=None, save_ext='')

#  Power Spectral Density Plots
labels_ = labels#['Jacket', 'No Jacket', 'SG6 Slow', 'SG12 Slow']
#[scan.multitaper_spectrum(normed=False, bandpass=None, dc_corr_seconds=4e-6, tmax=80, plot_trace=False, position=183, trace_index=None, figsize=(8,6), save_dir=None, save_ext='slow_0psi', linewidth=1.5, color='b', title='SG-6 Fast Direction',show=False) for scan in scans[:-1]]
#scans[-1].multitaper_spectrum(normed=False, bandpass=None, dc_corr_seconds=0e-6, tmax=1e6, plot_trace=True, position=None, trace_index=None, figsize=(10,8), save_dir=None, save_ext='slow_0psi', linewidth=1.5, title='', show=True)
#ps.spectrum_comparison(bandpass=None,scans=scans, dc_corr_seconds=0e-6,positions=None,tmax=50.0,save_dir=None,save_ext='_jacket_pressure_comp_tb4',labels=labels_, normed=False, normalise_spectra=False,plot_uncertainties=False,picks_offset=offset, window_around_p=None, figsize=(8,6), legend=True, legend_loc='upper right',max_freq=2000)
#[-10,182,80,90]

#  Create an animated comparison plot between different scanskiwi1_1925nm_preamp_
func_kwargs = {'normed': 'scan', 'bandpass': (1e2,2e6), 'tmax': 30, 'dc_corr_seconds': 0e-06, 'amp_factor':15., 'show': False, 'save_dir': None, 'plot_picks':True,'tick_fontsize':16.0,'lab_font':24.,'title_font':25.}
#func_kwargs = [{'normed':False, 'bandpass':(1e2,2e6), 'dc_corr_     secondkiwi1_1925nm_preamp_s':6e-6, 'scans':scans, 'tmax':80, 'trace_index':i, 'labels':labels,'legend_fontsize':12.,'tick_fontsize':16.0,'lab_font':24.,'ylim':(-9.,6.), 'linewidth':2., 'show_orientation':False} for i in range(0,size)]
#scans[0].animated_comparison(plot_type='wiggle', scans=scans, func_kwargs=func_kwargs, save_dir=None, save_ext='gsnz_wig_comp', update_interval=.8, show=True, titles=labels, figsize=(14,9), repeat_delay=2.,file_type='.mp4')
#scans1[0].animated_comparison(plot_type='variable_density', scans=scans1, func_kwargs=func_kwargs, save_dir=None, save_ext='vd_comparison', update_interval=1., show=True, titles=labels, figsize=(8,6), repeat_delay=0.)

#  Create a single plot showing a comparison of traces at the same position in different scans.
#fig = scans[0].trace_comparison(normed=True, bandpass=(2e4,.5e6), dc_corr_seconds=4e-6,scans=scans, tmin=0.0, tmax=200, position=None, trace_index=27, save_dir=None, save_ext='slow_0psi', plot_picks=False, labels=labels, title=title1, show=True, legend_loc='lower right')
#scans[-1].trace_comparison(normed=False, bandpass=(1e2,5e6), dc_corr_seconds=0e-6,scans=scans, tmin=0.0, tmax=20.0, average=True, position=None, trace_index=None, save_dir=fig_dir, picks_offset=offset,window_around_p=None, save_ext='tape-no-tape',plot_picks=False, labels=labels, show=True,title='First Break Comparison', legend_loc='upper right',show_orientation=False,linewidth=1.)
#scans[-1].trace_comparison(normed=False, bandpass=(1e2,3e6),dc_corr_seconds=0e-6,scans=scans, show=False, tmin=0.0, tmax=20.0, average=None, position=150.0,trace_index=None, save_dir=None, picks_offset=offset, window_around_p=None, save_ext='',plot_picks=False, labels=labels, title='', legend_loc='upper right',show_orientation=False,figsize=(10,7),ylim=(-6,6),linewidth=1.5)
#scans[-1].trace_comparison(normed=False, bandpass=(1e2,3e6),dc_corr_seconds=0e-6,scans=scans, show=True, tmin=-0.0, tmax=20.0, average=None, position=60.0, trace_index=None, save_dir=None, picks_offset=offset, window_around_p=None, save_ext='',plot_picks=False, labels=labels, title='', legend_loc='upper right',show_orientation=False,figsize=(10,7),ylim=None,linewidth=1.5)
#fig = scans[-1].trace_comparison(normed=True, bandpass=(1e2,3e6),dc_corr_seconds=4e-6,scans=scans, show=False, tmin=None, tmax=20.0, average=None, position=35.,trace_index=None, save_dir=None, picks_offset=offset, window_around_p=None, save_ext='',plot_picks=False, labels=labels, title='', legend_loc='upper right',show_orientation=False,figsize=(10,7),ylim=None,linewidth=1.5)
fig = scans[-1].trace_comparison(normed=True, bandpass=(1e2,3e6),dc_corr_seconds=0e-6,scans=scans, show=True, tmin=None, tmax=10.0, average=None, position=None,trace_index=0, save_dir=None, picks_offset=offset, window_around_p=None, save_ext='',plot_picks=False, labels=labels, title='', legend_loc='upper right',show_orientation=False,figsize=(10,7),ylim=None,linewidth=1.5)


#    Cross-correlation plots
#fig, max_lag1 = scans[0].cross_correlation(scans=scans, bandpass=(1e2,5e6), show=True, tmin=2.0, tmax=14.0, save_dir=None)
#print('Cross-correlation Lag Time: {}'.format(round(max_lag1,2)))

#plot.simple_plot_points([680,980,1064,1450,1731,1925],[np.max(np.abs(scan._get_plot_data(tmax=1e6, tmin=0.0, normed=False, bandpass=(1e2,6e4))[0])) for scan in scanhhs],linestyle='',marker='o',color='g',xlab='Wavelength (nm)',ylab='Normalised Max Amplitude (a.u.)',markersize=8.,lab_font=16.,save_dir=[fig_dir+'kiwifruit_wavelength_comp.png',fig_dir+'kiwifruit_wavelength_comp.pdf'])

#Plot the anisotropies
#ys = [an.retrieve_data(scan.scan_dir[:scan.scan_dir[:-1].rfind('/')+1]+scan.scan_name[:scan.scan_name.rfind('_')]+'_anistorpies.csv')[1][1:] for scan in scans]
#ys = [[i*100. for i in list_] for list_ in ys]
#xs = [list([int(press) for press in pressures])]*len(ys)
#plot.simple_plot_series(xs, ys, labels=['Schist', 'Protomylonite', 'Ultramylonite', 'Cataclasite'], ylab='Anisotropy (%)', xlab='Confining Pressure (MPa)',
#                figsize=(10,6), marker='o', legend=True, legend_loc='upper right', ylim=(0,50), save_dir=None)





#For jacket comp
#, inset_params={'region':[7.0,7.8,-.03,.16],'inset_loc':2,'zoom':8,'aspect':4}

#Parameters of presentation plots
# linewidth=2.,show_orientation=False,tick_fontsize=16.0,lab_font=24.,legend_fontsize=14.

#TAN-01 mute and reposition:
#scans[0].reposition(steps=3)
#[scan.mute([(0,14),(168,180)]) for scan in scans]