'''
Plot some simple plots of a scan using PlaceScan
'''

from PlaceScan import PlaceScan



scan_dirs = [\
'/home/jonathan/Dropbox/jonathan_masters/data/tan/tan_0_129_f/',
'/home/jonathan/Dropbox/jonathan_masters/data/tan/tan_50_129_f/',
'/home/jonathan/Dropbox/jonathan_masters/data/tan/tan_100_129_f/'\
][:1]
fig_dir = '/home/jonathan/Dropbox/jonathan_masters/figs/aluminium/'

scans = [PlaceScan(scan_dir, scan_type='single') for scan_dir in scan_dirs]
size = len(scans[0].trace_data)

#[scan.mute([(0,4),(172,180)]) for scan in scans]
#[scan.mute([(79,91),(179,191)]) for scan in scans]
#[scan.mute([(339,341)]) for scan in scans]



#[scan.wiggle_plot(normed=True, bandpass=(1e4, 1e6), tmax=400, amp_factor=3.0, save_dir=None, show=False) for scan in scans[:-1]]
#scans[-1].wiggle_plot(normed=True, bandpass=(3e4,1e6), tmax=50, amp_factor=1.0, save_dir=fig_dir, save_ext='2',show=True, plot_picks=True, pick_errors='both')

#[scan.variable_density(normed=True, bandpass=(3e4,1e6), tmax=50, gain=3, save_dir=None, plot_picks=True, pick_errors='both')  for scan in scans[:-1]]
#scans[-1].variable_density(normed=True, bandpass=(3e4,1e6), tmax=100, gain=3, save_dir=None, plot_picks=True, pick_errors='both') 

#[scan.trace_plot(normed=False, bandpass=(1e3, 0.1e6), trace_int=None, averaging=size, figsize=(12,4), save_dir=None, save_ext='stacked_100kHz', linewidth=1.0,show=False) for scan in scans[:-1]]
scans[-1].trace_plot(normed=False, bandpass=(1e4, 8e5), tmax=1000, trace_int=None, averaging=size, figsize=(12,3), save_dir=fig_dir, save_ext='stacked', linewidth=1.0)

#[scan.arrival_picks_plot(normed=False, bandpass=(1e3, 0.1e6), trace_int=None, averaging=size, figsize=(12,4), save_dir=None, save_ext='stacked_100kHz', linewidth=1.0,show=False) for scan in scans[:-1]]
#scans[-1].arrival_picks_plot(picks_dirs=scan_dirs, save_dir=fig_dir, save_ext='', pick_errors=None, polar=True, sample_diameter=None, labels=['0.00 MPa','0.35 MPa','0.70 MPa'], legend_loc=(0.87,-0.1))
