'''
Example script to plot some waveform data using PlaceScan
'''

from placescan.main import PlaceScan

#Initialise the scans
scan1 = PlaceScan('example-data/1MPa-scan/', scan_type='rotation', trace_field='ATS660-trace',apply_formatting=False,divide_energy=False)
scan16 = PlaceScan('example-data/16MPa-scan/', scan_type='rotation', trace_field='ATS660-trace',apply_formatting=False,divide_energy=False)

#Mute some of the traces
scan1.mute([(213,360)])
scan16.mute([(213,360)])

#Plot the traces at 90 degrees.
fig = scan1.trace_plot(bandpass=(1e2,3e6), tmax=30.0, position=90., show=False, marker='', title='1 MPa')
fig = scan16.trace_plot(bandpass=(1e2,3e6), tmax=30.0, position=90., show=True, marker='', title='16 MPa')

#Plot spectral estimates at 90 degrees.
fig = scan1.multitaper_spectrum(tmax=50., plot_trace=True,plot_uncertainties=True, position=90., max_freq=2500, marker='',show=False, title='1 MPa')
fig = scan16.multitaper_spectrum(tmax=50., plot_trace=True,plot_uncertainties=True, position=90., max_freq=2500, marker='', title='16 MPa')

# Plot traces from the two scans on top of each other
fig = scan1.trace_comparison(normed=True,colorbar_horiz=True,bandpass=(1e2,3e6),position=180.,scans=[scan1,scan16], tmax=30., marker='', labels=['1 MPa','16 MPa'])

#  Plot Wiggle Plots of the two scans
figs = scan1.wiggle_plot(xlim=(52,213),normed='scan', bandpass=(1e2,3e6), tmax=30.0,amp_factor=10.0, show=False,xlab=r'Group Angle (degrees)',title='1 MPa')
fig = scan16.wiggle_plot(xlim=(52,213),normed='scan', bandpass=(1e2,3e6), tmax=30.0,amp_factor=10.0, show=True,xlab=r'Group Angle (degrees)',title='16 MPa')

# There are many more methods aside from those demonstrated here for plotting, arrival time picking, organising, analysing, and processing PLACE data
# The documentation contained in the helper scripts give details. Note that many more parameters than those demonstrated here can be passed to the 
# plotting functions to fully customise the plots, including standard matplotlib keywords.
