'''
A simple script to perform analyses on PlaceScan data
'''

import PlaceScan as ps
import plotting as pt
import analysis

master_dir = '/mnt/office_machine/home/jsim921/Dropbox/jonathan_masters/data/'
pressures = ['1','2','4','6','8','10','12','14','16']
sample = 'pmv1'
scan_dirs = [master_dir+'{0}/{0}_{1}MPa/'.format(sample,pressure) for pressure in pressures]
data_file = '/mnt/office_machine/home/jsim921/Dropbox/jonathan_masters/data/alpine_fault_rock_data.csv'

labels = ['{} MPa'.format(pressure) for pressure in pressures]

scans = [ps.PlaceScan(scan, scan_type='rotation', trace_field='ATS660-trace',apply_formatting=False,divide_energy=False) for scan in scan_dirs]

#Calcualte all rock physics data
#rp_data = scans[0].rock_physics_calculator(labels, data_file,scans=scans, save_ext='', picks_offset=0., picks_xlim=(17., 156.))
rp_data = scans[0].rock_physics_calculator(labels, data_file,scans=scans, save_ext='', picks_offset=0., picks_xlim=(10., 180.), s_wave_picks_dir=master_dir+'{}/{}_s-picks.csv'.format(sample,sample))
#pt.simple_plot_points(range(len(scans)), rp_data.ans)

variables = ['anisotropy','epsilon','gamma','delta','c11', 'c33', 'c13', 'c55','c66']
analysis.construct_latex_table(master_dir, 'pmv1', labels, variables,'test_table')