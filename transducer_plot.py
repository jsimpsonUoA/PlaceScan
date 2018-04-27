
from PlaceScan import PlaceScan

scan_dirs = [\
'/home/jsim921/Dropbox/jonathan_masters/Aluminium_transducer_test_Kiara/aluminium_transducer_test_14.5mm_preamp1/',
'/home/jsim921/Dropbox/jonathan_masters/Aluminium_transducer_test_Kiara/aluminium_transducer_test_24.5mm_preamp1/',
'/home/jsim921/Dropbox/jonathan_masters/Aluminium_transducer_test_Kiara/aluminium_transducer_test_34.5mm_preamp1/',
'/home/jsim921/Dropbox/jonathan_masters/Aluminium_transducer_test_Kiara/aluminium_transducer_test_44.5mm_preamp1/',
'/home/jsim921/Dropbox/jonathan_masters/Aluminium_transducer_test_Kiara/aluminium_transducer_test_54.5mm_preamp1/',
'/home/jsim921/Dropbox/jonathan_masters/Aluminium_transducer_test_Kiara/aluminium_transducer_test_64.5mm_preamp2/'\
][:]


scans = [PlaceScan(scan_dir, scan_type='single', trace_field='DPO3014-trace') for scan_dir in scan_dirs]

new_scan = scans[0].combine(scans[1:], append_index=0, reposition=True)
size = len(new_scan.trace_data)

new_scan.wiggle_plot(normed=True, bandpass=(1e4,5e5), tmax=400, amp_factor=0.8, save_dir=None, save_ext='2',show=True, plot_picks=False, pick_errors='both')

new_scan.trace_plot(normed=False, bandpass=None, tmax=1000, trace_int=None, averaging=size, figsize=(12,3), save_dir=None, save_ext='stacked', linewidth=1.0)