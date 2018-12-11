'''
Pick arrivals of a PlaceScan

Press 'q' to record the position of the cursor, 'w' to skip the trace,
and 'Delete' to undo the last pick.
'''

import glob
from PlaceScan import PlaceScan

scan_dir = '/home/jsim921/Dropbox/timber_work/Data/timber_rot_sg6_00-1/'
picks_dir = scan_dir+'p_wave_picks.csv'

scan = PlaceScan(scan_dir, scan_type='rotation', trace_field='ATS9440-trace')

#scan.expand()
#scan.mute_by_signal(50, analog=True, sig_plot=True)

size = len(scan.trace_data)
print(size)

#  Manual Arrival time picking
scan.manual_pick(bandpass=None, tmax=50, amp_factor=20.0, save_file=None) 


#  Automatic Spectrogram-based arrival time picking
#scan.spectrogram_auto_pick(normed=True, bandpass=None, tmax=50, search_window=[5.0e-6,22.0e-6], trace_index=None, picks_save_dir=picks_dir, average=False, show=True, taper=3., save_dir=None, max_pick=False, threshold_pick=True, threshold=2.5, save_ext='')