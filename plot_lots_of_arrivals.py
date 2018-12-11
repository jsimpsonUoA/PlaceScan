import plotting as pl

main_dir = '/home/jonathan/Dropbox/timber_work/Data/sg6_90_rot-1/p_wave_picks.csv'
dirs = [main_dir+'-'+str(i) for i in range(36)]

pl.arrival_times_plot(dirs, save_dir=None, pick_errors=None, polar=False,picks_offset=0., labels=list(range(87)), tick_fontsize=14.0, figsize=(6,6),legend_fontsize=1.)