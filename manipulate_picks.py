'''
A python script to manipulate the arrival time picks from PlaceScan data 
folders after they have been created. this saves having to go through the
picking routines for the PlaceScan objects if simple corrections are being
added to each of the picks files.

Written by Jonathan Simpson, jsim921@aucklanduni.ac.nz
PAL Lab UoA, November 2018
'''

import os
import glob
import shutil

import picking as pk

####################################################################################################
samples = ['umv']  #List of the samples as strings
picks_file_suffix = '_p-picks.csv'                 #The last part of each picks file common to all

master_scan_directory = '/home/jsim921/Dropbox/jonathan_masters/data/'
####################################################################################################


def main(samples, suffix, master_dir):
    '''
    The main function to run the script. Add or comment out
    function calls here in to change what manipulate picks does.
    '''

    check_user()
    dirs = get_picks_dirs(samples, suffix, master_dir)
    
    for dir_ in dirs:
        #pk.correct_picks(dir_, pick_corr=0.0, early_err_corr=0.0, late_err_corr=0.0)
        pk.smooth_picks_by_av(dir_, num_of_traces=5)


####################################################################################################


def check_user():
    '''
    Check if the user wants to proceed
    '''
    proceed = input('Picks Manipulate: Continue?? [y/N]: ')
    if proceed != 'y':
        exit()


def get_picks_dirs(samples, suffix, master_dir):
    '''
    A function to get all the picks directories in a sample
    '''
    
    dirs = []
    for sample in samples:
        print('Manipulate Picks: Working on sample {}'.format(sample))

        dir_ = master_scan_directory + sample + '/**/*' + picks_file_suffix 
        picks_dirs = glob.glob(dir_, recursive=True)
        dirs += picks_dirs

    return dirs

if __name__ == "__main__":
    main(samples, picks_file_suffix, master_scan_directory)