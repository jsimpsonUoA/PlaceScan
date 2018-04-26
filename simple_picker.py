'''
Pick arrivals of a PlaceScan

Press 'q' to record the position of the cursor, 'w' to skip the trace,
and 'Delete' to undo the last pick.
'''

from PlaceScan import PlaceScan

scan_dir = "/home/jsim921/Dropbox/jonathan_masters/data/tan/tan_100_129_f/"
picks_dir = scan_dir+'p_wave_picks.csv'

scan = PlaceScan(scan_dir)
scan.pick_arrivals(bandpass=None, tmax=50, amp_factor=3.0, save_file=None) 