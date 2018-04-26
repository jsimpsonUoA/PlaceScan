'''
Combine two or more scans simply using PlaceScan
'''

from PlaceScan import PlaceScan

scan1_dir = "/home/jonathan/Dropbox/jonathan_masters/data/tan/tan_100_129_f/"
scan2_dir = "/home/jonathan/Dropbox/jonathan_masters/data/tan/tan_0_129_part2/"
new_scan_dir = "/home/jonathan/Dropbox/jonathan_masters/data/tan/tan_0_129/"

scan1 = PlaceScan(scan1_dir)
scan1.reposition(steps=-1)
#scan1.write(scan1_dir)

#scan2 = PlaceScan(scan2_dir)

#new_scan = combine_scans([scan1, scan2], append_index=1, save_dir=None, reposition=False)
scan1.wiggle_plot(bandpass=(1e4, 2e6), tmax=50, amp_factor=3.0) 

