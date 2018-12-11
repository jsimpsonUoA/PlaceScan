'''
A python script to backup the arrival time picks from PlaceScan data 
folders to a backup location. A list of samples to backup the data for
is specified.

Written by Jonathan Simpson, jsim921@aucklanduni.ac.nz
PAL Lab UoA, November 2018
'''

import os
import glob
import shutil

####################################################################################################
samples = ['7f-schist-2', 'pmv1', 'umv', '1a842']  #List of the samples as strings
picks_file_suffix = '_p-picks.csv'                 #The last part of each picks file common to all

master_scan_directory = '/home/jsim921/Dropbox/jonathan_masters/data/'
master_backup_folder = '/home/jsim921/Dropbox/jonathan_masters/data/picks_backup/'
####################################################################################################

proceed = input('Picks Backup: Continue?? [y/N]')
if proceed != 'y':
    exit()

for sample in samples:
    print('Picks Backup: Copying picks for {}'.format(sample))

    dir_ = master_scan_directory + sample + '/**/*' + picks_file_suffix 
    picks_dirs = glob.glob(dir_, recursive=True)

    backup_dir = master_backup_folder + sample
    if not os.path.isdir(backup_dir):
        os.mkdir(backup_dir)

    for picks in picks_dirs:
        name = picks[picks.rfind('/'):]
        shutil.copy(picks, backup_dir + name)