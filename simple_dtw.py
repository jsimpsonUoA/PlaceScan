'''
Pick arrivals of a PlaceScan using Dynamic Time Warping analysis
'''

import glob
from PlaceScan import PlaceScan

master_dir = '/mnt/office_machine/home/jsim921/Dropbox/jonathan_masters/data/'
pressures = ['2','12']
sample = 'umv'
scan_dirs = [master_dir+'{0}/{0}_{1}MPa/'.format(sample,pressure) for pressure in pressures][::-1]
                  
scans = [PlaceScan(scan, scan_type='rotation', trace_field='ATS660-trace',apply_formatting=False) for scan in scan_dirs]
[scan.average_traces() for scan in scans]

#[scan.mute([(170,180)]) for scan in scans]
size = len(scans[0].trace_data)
print(size)

scans[0].dynamic_time_warping_pick(mode='scan', scans=scans, save_picks=False, number_of_iterations=size, start_position=80, start_index=0, bandpass=(1e2,.8e6),\
        dc_corr_seconds=5e-6, tmax=30.0, path_length=25.88, window_size=10., max_jump=.8, alpha=0.06, reverse=False, plot_lags=False, arrival_time_corr=-0.04,
        early_err_corr=-0.02, late_err_corr=0.02)

scans[0].wiggle_plot(normed='scan', bandpass=(1e2,.5e6), dc_corr_seconds=0e-6, tmax=30.0, amp_factor=15.0, save_dir=None, decimate=False, save_ext='',show=True, plot_picks=True, xlab=r'\theta (degrees)',picks_offset=0,tick_fontsize=16.0,lab_font=24.,title='',title_font=24.)

#scans[0].dtw_multiscan_picking(scans=scans, save_picks=True, bandpass=(1e2,5e6), tmax=30.0, path_length=25.88, window_size=10.,\
#        max_jump=1., max_adjacent_jump=.5, alpha=0.03, plot_lags=False, order_criteria='sooner', arrival_time_corr=-0.04,
#        early_err_corr=-0.02, late_err_corr=0.02)

#[scan.dynamic_time_warping_pick(mode='scan', scans=scans, save_picks=True, number_of_iterations=size, start_position=None, start_index=0, bandpass=(1e2,1e6),\
#        dc_corr_seconds=0e-6, tmax=30.0, path_length=25.88, window_size=10., max_jump=.5, alpha=0.00, reverse=True, plot_lags=False, arrival_time_corr=-0.04,
#        early_err_corr=-0.02, late_err_corr=0.02) for scan in scans]