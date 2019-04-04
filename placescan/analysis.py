'''
Python module with functions to do rock physics analyses.
This is written to be used in conjunction with the PlaceScan
class.

Jonathan Simpson, jsim921@aucklanduni.ac.nz
Masters project, PAL Lab UoA, November 2018
'''

import numpy as np 
import csv
import pickle
from uncertainties import ufloat
from uncertainties import unumpy as unp

import placescan.plotting as plotting
from placescan.c13_inversion import C13CalculatorExp
from placescan.true_c13 import C13Calculator

class RPCalculator():
    def __init__(self, sample_name, picks_dirs, rp_save_filename,
                labels, data_filename, save_data=True, s_wave_picks_dir=None, 
                picks_offset=0., picks_xlim=None, c13_invert=True,
                parameters_to_invert=[], use_common_max_min_positions=True,
                plot_c13=True, fast_slow_p_picks_dir=None):
        '''
        RPCalculator is a class to handle all the rock
        physics calculations that can be done using the 
        data from PlaceScan objects. In particular, the
        samples are assumed to have tranverse isotropy.
        Its usual usage is to give one or more scans for
        a single sample and do all calculations for that
        sample at one.

        Arguments:
            --sample_name: The name of the sample
            --picks_dirs: A directory or list of directories/filenames
                where the wave arrival picks are saved
            --rp_save_filename: The directory of the rp_data file to save the data to
            --labels: A label or list of labels to identify each entry/scan by
            --data_filename: The name of the csv file which contains 
                data about the sample (e.g. density, mass, height, etc). See the 
                retrieve_sample_data function for the correct format.
            --save_data: True to save the rock physics data, False not to.
            --s_wave_picks_dir: The path to the csv file that contains the 
                S-wave picks for the sample. Usually, this will contain s0 and
                h90 picks for tranversely isotropic media, for each scan (same
                format as described in retrieve_sample_data)
            --picks_offset: The amount to offset the arrival time picks from the
                saved positions
            --picks_xlim: A tuple of the minimum and maximum x positions to plot 
                picks for       
            --c13_invert: True to invert for the c13 parameter.
            --parameters_to_invert: A list of parameters to invert for in the c13
                inversion. See c13_inversion for more details.
            --use_common_max_min_positions: True to use the same x_positions for the
                theta = 0 and theta = 90 positions of a VTI sample. If True, then the
                position where the maximum velocity occurs the most will be theta = 90.
                If False, then the positions for the max and min velocities are used for
                each scan.
            --plot_c13: True to plot the result fo the inversion for c13
            --fast_slow_p_picks_dir: A file with the fast and slow P-wave picks
                in the same format as the s-wave picks, if these are different from
                what the calculator will automatically find.
        '''

        self.sample_name = sample_name
        self.picks_dirs = picks_dirs
        self.rp_save_filename = rp_save_filename
        self.labels = labels
        self.picks_offset = picks_offset
        self.picks_xlim = picks_xlim
        self.c13_invert = c13_invert
        self.save_data = save_data
        self.parameters_to_invert = parameters_to_invert
        self.use_common_max_min_positions = use_common_max_min_positions
        self.plot_c13 = plot_c13
        self.fast_slow_p_picks_dir = fast_slow_p_picks_dir

        self.data = {}    #All data, retrieved or cacluated, will be put in here. The next two dicts are for easy access

        self.sample_data = self.retrieve_sample_data(data_filename)[sample_name]
        self.get_s_wave_picks(s_wave_picks_dir)
        print(self.s_wave_picks)
        self.max_inds, self.min_inds = self.get_max_min_inds()  #Indices of max and min arrival times
        self.get_max_min_p_picks()

        self.data.update(self.sample_data)
        self.save_rp_data()

        # The order of the calculation is important
        self.ans, self.ans_err = self.calculate_anisotrpy()
        self.eps, self.eps_err = self.calculate_epsilon()
        self.cijs = self.calculate_cijs()
        self.delta, self.delta_err = self.calculate_delta()
        self.gam, self.gam_err = self.calculate_gamma()

        self.save_velocities()
        

    def save_rp_data(self):
        '''
        Function to save the picks data

        Arguments:
            --data_dict: A dictionary containing the data
            --data_key: The name to call the dataset in the json file
            
        Returns:
            --None
        '''
        
        if self.save_data:
            try:
                with open(self.rp_save_filename, 'rb') as f:
                    file_data = pickle.load(f)
            except FileNotFoundError:
                file_data = {}

            file_data.update(self.data)

            with open(self.rp_save_filename, 'wb') as f:
                pickle.dump(file_data, f)

    def retrieve_sample_data(self, filename, raise_exception=True):
        '''
        Function to retrieve data about samples saved in a
        separate csv file. The format of this file must be:

        <empty>  | Variable 1 | Variable 2 | ...
        Sample 1 |    ...     |    ...     | ...
        Sample 2 |    ...     |    ...     | ...
        :

        The variable and sample anmes must be strings, and the
        values will be converted to floats. The result is a 
        data dictionary for each sample, contained in a sample dict

        Arguments:
            --filename: The filename (path) of the csv file
            --raise_exception: If True, the program will be exited
                if the file at the given path cannot be found.

        Returns:
            --sample_dict: A dictionary data dictionaries
        '''

        try:
            with open(filename, 'r') as f:

                reader = list(csv.reader(f))
                data_keys = reader.pop(0)[1:]
                samples = [row.pop(0) for row in reader]
                sample_dict = {}

                try:
                    for i, row in enumerate(reader):
                        data = [float(item) for item in row]
                        sample_dict[samples[i]] = dict(zip(data_keys, data))
                except ValueError:
                    print('PlaceScan Analysis: Cannot retrieve sample data because some entries are not numbers.')
                    exit()

                sample_dict = self.create_samle_data_ufloats(sample_dict)

                return sample_dict

        except TypeError or FileNotFoundError:
            if raise_exception:
                print('PlaceScan Analysis: Cannot find sample data csv file.')
                exit()
            else:
                return {}
                
    def create_samle_data_ufloats(self, sample_dict):
        '''
        Helper function to retrieve_sample_data which
        converts values and errors as individual entries
        in the sample_data dict to single entries as ufloats
        '''
                
        samples = list(sample_dict.keys())

        for sample in samples:
            data_dict = sample_dict[sample]
            data_keys = list(data_dict.keys())
            for key in data_keys:
                if key+'_err' in data_keys:
                    value = ufloat(data_dict[key], data_dict[key+'_err'])
                    data_dict[key] = value
                    del data_dict[key+'_err']

            sample_dict[sample] = data_dict

        return sample_dict

    def get_s_wave_picks(self, s_wave_dir):
        '''
        Function to control retrieval of s-wave picks
        '''
        if s_wave_dir:
            self.s_wave_picks = self.retrieve_sample_data(s_wave_dir, raise_exception=False)
        else:
            self.s_wave_picks = dict(zip(self.labels,[{'sh90':ufloat(np.nan,np.nan),'s0':ufloat(np.nan,np.nan)}]*len(self.labels)))

    def picks_iterator(self):
        '''
        A generator used for iterating through the picks data in
        picks_dirs. Usually, picks_dirs will be a list of all the 
        scans for a sample.

        Arguments:
            None

        Yields:
            --i: The index
            --headers: The headers of the picks .csv
            --x_pos: A list of the x_positions
            --times: The pick times, as an array of ufloat objects
                containing the pick and its associated error.
        '''

        if not isinstance(self.picks_dirs, list):
                picks_dirs = [self.picks_dirs]
        else:
            picks_dirs = self.picks_dirs

        for i, dir_ in enumerate(picks_dirs):
            headers, picks_data = plotting.picks_from_csv(dir_)
            if headers:
                x_pos = picks_data[0] + self.picks_offset
                if not self.picks_xlim:
                    indices = [np.where(picks_data[i] != -1.)[0]  for i in range(1,picks_data.shape[0])]
                else:
                    indices = [np.where((picks_data[i2] != -1.) & (x_pos > self.picks_xlim[0])\
                        & (x_pos < self.picks_xlim[1]))[0] for i2 in range(1,picks_data.shape[0])]
                    x_pos = x_pos[np.where((x_pos > self.picks_xlim[0]) & (x_pos < self.picks_xlim[1]))]
                
                if len(indices[1]) > 0 and (len(indices[0]) != len(indices[1]) != len(indices[2])):
                    print('PlaceScan analysis: Not all picks have errors. Please correct the picks file.')
                    exit()
                elif len(indices[1]) == 0:
                    errors = np.array([0.]*len(indices[0]))
                else:
                    errors = abs(picks_data[2][indices[1]] - picks_data[3][indices[2]]) / 2. #An everage error

                times = unp.uarray(picks_data[1][indices[0]], errors)
                yield i, headers, x_pos, times

            else:
                yield i, [], [], []

    def swap_dicts_for_saving(self, dict_):
        '''
        Function to take a dictionary which contains
        sub-dictionaries containing the values of
        variables at each scan and change it in to
        a dictionary where each sub-dictionary belongs
        to a variable and contains the values of that
        variable for each scan.

        Arguments:
            --dict_: The dictionary to swap around
        '''

        keys = list(dict_.keys())
        new_dict = {}
        for key in keys:
            sdict = dict_[key]
            for skey, value in sdict.items():
                if skey not in new_dict:
                    new_dict[skey] = {key:value}
                else:
                    new_dict[skey][key] = value

        return new_dict

    def save_velocities(self):
        '''
        Save the wave velocities used to calculate
        the cijs to the data dictionary.
        '''

        diameter = self.sample_data['diameter']
        height = self.sample_data['height']
        s_vels = self.swap_dicts_for_saving(self.s_wave_picks)
        for wave in s_vels:
            wave = s_vels[wave]
            for scan in wave:
                wave[scan] = height / wave[scan] * 1e3            #### CAREFUL! Should it be height or diameter?
            
        p_vels = {'vp0':{},'vp90':{}}
        for i, headers, x_pos, times in self.picks_iterator():
            if headers:  
                v_max = diameter/self.p_wave_picks[self.labels[i]]['vp90']*1e3
                v_min = diameter/self.p_wave_picks[self.labels[i]]['vp0']*1e3
                p_vels['vp0'][self.labels[i]] = v_min
                p_vels['vp90'][self.labels[i]] = v_max

        self.data.update(s_vels)
        self.data.update(p_vels)
        self.save_rp_data()

    def get_max_min_inds(self):
        '''
        Function to determine the positions where the maximum and
        minimum arrival time picks occur for a sample. This is done
        by determining the position where the most max/min times
        occur across all the scans for a sample.

        Arguments:
            None

        Returns:
            --max_inds, min_inds: The indices in the scans of the max and min arrival times.
        '''

        min_inds, max_inds = [], []
        for i, headers, x_pos, times in self.picks_iterator():
            min_ind, max_ind = np.argmin(times), np.argmax(times)
            min_inds.append(min_ind), max_inds.append(max_ind)

        if self.use_common_max_min_positions:
            min_occ = np.array([min_inds.count(num) for num in min_inds])
            max_occ = np.array([max_inds.count(num) for num in max_inds])

            #Note: if there is not a single position that has more than one max/min, 
            #      then the last scan's max/min are used
            if max(min_occ) > 1:
                min_ind = min_inds[np.argmax(min_occ)]
            else:
                print('Using last scan min')
            if max(max_occ) > 1:
                max_ind = max_inds[np.argmax(max_occ)]
            else:
                print('Using last scan max')
                
            print('Fast direction: {}, Slow direction: {}'.format(x_pos[min_ind], x_pos[max_ind]))

            max_inds, min_inds = [max_ind]*(i+1), [min_ind]*(i+1)

        self.data['max_inds'] = max_inds
        self.data['min_inds'] = min_inds
        return max_inds, min_inds

    def get_max_min_p_picks(self):
        '''
        Function get the maximum and minimum P-wave
        arrival times, either from the maximum and minimum
        indices, or from a saved csv file of the picks
        '''

        if self.fast_slow_p_picks_dir:
            self.p_wave_picks = self.retrieve_sample_data(self.fast_slow_p_picks_dir, raise_exception=False)
            self.data['fast_slow_p_picks_dir'] = self.fast_slow_p_picks_dir
        else:
            p_wave_picks = {}
            for i, headers, x_pos, times in self.picks_iterator():
                if headers:  
                    min_pick, max_pick = times[self.min_inds[i]], times[self.max_inds[i]]
                    p_wave_picks[self.labels[i]] = {'vp90':min_pick,'vp0':max_pick}
            self.p_wave_picks = p_wave_picks
            self.data['fast_slow_p_picks_dir'] = 'Not Used'

    def calculate_anisotrpy(self):
        '''
        Function to calculate the anisotropy from a .csv file 
        of wave arrival picks from a PlaceScan. The results are
        saved in the rock physics (rp_data) file for the sample. 

        Arguments:
            None

        Returns:
            --anisotropies: A list of the anisotropy values.
            --errors: A list of the anisotropy errors
        '''

        anisotropies = []

        for i, headers, x_pos, times in self.picks_iterator():
            if headers:  
                v_max, v_min = 1/self.p_wave_picks[self.labels[i]]['vp90'], 1/self.p_wave_picks[self.labels[i]]['vp0']
                anisotropy = (v_max - v_min) / (.5 * (v_max + v_min))
            else:
                anisotropy = ufloat(np.nan, np.nan)

            anisotropies.append(anisotropy*100.)

        self.data['anisotropy'] = dict(zip(self.labels, anisotropies))
        self.save_rp_data()

        return [val.n for val in anisotropies], [val.s for val in anisotropies]


    def calculate_epsilon(self):
        '''
        Function to calculate the epsilon parameter from a .csv file 
        of wave arrival picks from a PlaceScan. The results are
        saved in the rock physics (rp_data) file for the sample.

        Arguments:
            None

        Returns:
            --epsilons: A list of the epsilon values.
            --errors: A list of the epsilon errors
        '''

        epsilons = []

        for i, headers, x_pos, times in self.picks_iterator():
            if headers:  
                v_max, v_min = 1/self.p_wave_picks[self.labels[i]]['vp90'], 1/self.p_wave_picks[self.labels[i]]['vp0']
                epsilon = (v_max**2 - v_min**2) / (2. * v_min**2)
            else:
                epsilon = ufloat(np.nan, np.nan)

            epsilons.append(epsilon)

        self.data['epsilon'] = dict(zip(self.labels, epsilons))
        self.save_rp_data()

        return [val.n for val in epsilons], [val.s for val in epsilons]


    def calculate_cijs(self):
        '''
        Function to caluclate the cij's of the elastic
        stiffness tensor for transversely isotropic samples. 
        If the diameter is in mm, the arrival time picks in
        microseconds, and the denisty in units of g/cm^3, then
        the cijs will have units of GPa

        Arguments:
            None

        Returns:
            -cijs: A list of dictionaries of the cij's for each scan
        '''

        diameter = self.sample_data['diameter']
        height = self.sample_data['height']
        print(self.sample_data)
        try:
            density = self.sample_data['density']
        except KeyError:
            density = self.sample_data['mass'] / (np.pi*(diameter/20.)**2 * self.sample_data['height']/10.)
            self.data['density'] = density

        cij_names = ['c11', 'c33', 'c13', 'c55', 'c66']
        cijs = []
        for i, headers, x_pos, times in self.picks_iterator():
            cij_dict = dict(zip(cij_names,[ufloat(np.nan, np.nan)]*5))

            if headers:  
                v_max, v_min = diameter/self.p_wave_picks[self.labels[i]]['vp90'], diameter/self.p_wave_picks[self.labels[i]]['vp0']
                cij_dict['c11'] = v_max**2 * density
                cij_dict['c33'] = v_min**2 * density

            try:
                sh90 = height/self.s_wave_picks[self.labels[i]]['sh90']    ### Watch out!! Should you use the height or the diameter??
                cij_dict['c66'] = sh90**2 * density
                missed_c66 = False
            except KeyError:
                missed_c66 = True

            try:
                s0 = height/self.s_wave_picks[self.labels[i]]['s0']      ### Watch out!! Should you use the height or the diameter??
                cij_dict['c55'] = s0**2 * density
                missed_c55 = False
            except KeyError:
                missed_c55 = True
                #cij_dict['c55'] = cij_dict['c66'] * 0.85  #JUST TESTING. REMOVE THIS IF YOU ACTUALLY HAVE C55 VALUES AND MAKE missed_c55 = True

            if self.c13_invert:
                #Experimental version
                #c13inv = C13CalculatorExp(diameter/times*1000., x_pos, cij_dict['c11'], cij_dict['c33'], 
                #        cij_dict['c55'], cij_dict['c66'], density*1000., theta90=x_pos[self.min_inds[i]], 
                #        parameters_to_invert=self.parameters_to_invert, diameter=diameter)
                #cij_dict['c13'] = c13inv.c13   

                #Operational version
                c11_inv, c33_inv = (diameter/times[self.min_inds[i]])**2 * density, (diameter/times[self.max_inds[i]])**2 * density
                c13inv = C13Calculator(diameter/times*1000., x_pos, c11_inv, c33_inv, 
                        cij_dict['c55'], cij_dict['c66'], density*1000., theta90=x_pos[self.min_inds[i]], 
                        diameter=diameter, plot_c13=self.plot_c13)
                cij_dict['c13'] = c13inv.c13   

                # If c55 and/or c66 were inverted for simultaneously with c13 AND these values were not
                # calculated from S-wave velocities, then c55 and/or c66 will be taken from the c13 inversion.
                if 'c55' in self.parameters_to_invert and missed_c55:
                    cij_dict['c55'] = c13inv.c55
                if 'c55' in self.parameters_to_invert and missed_c55:
                    cij_dict['c55'] = c13inv.c55

            cijs.append(cij_dict)

        for cij in cij_names:
            cij_data = [cij_dict[cij] for cij_dict in cijs]
            self.data[cij] = dict(zip(self.labels, cij_data))
            self.save_rp_data()

        return cijs

        
    def calculate_delta(self):
        '''
        Function to calculate the delta parameter from teh cijs
        previously calculated. The results are saved in the rock 
        physics (rp_data) file for the sample.

        Arguments:
            None

        Returns:
            --delta: A list of the delta values.
            --errors: A list of the delta errors
        '''

        deltas = []
        
        for i, label in enumerate(self.labels):
            c11, c33 = self.cijs[i]['c11'], self.cijs[i]['c33']
            c55, c13 = self.cijs[i]['c55'], self.cijs[i]['c13']
            null_float = ufloat(np.nan, np.nan)
            if c11!=null_float and c33!=null_float and c55!=null_float and c13!=null_float:  
                delta_top = 2*(c13+c55)*(c13+c55) - (c33-c55)*(c11+c33-2*c55)
                delta = delta_top / (2*c33*c33)
            else:
                delta = null_float

            deltas.append(delta)

        self.data['delta'] = dict(zip(self.labels, deltas))
        self.save_rp_data()

        return [val.n for val in deltas], [val.s for val in deltas]


    def calculate_gamma(self):
        '''
        Function to calculate the gamma parameter previously
        calculated cijs. The results are saved in the
        rock physics (rp_data) file for the sample.

        Arguments:
            None

        Returns:
            --gammas: A list of the gamma values.
            --errors: A list of the gamma errors
        '''

        gammas = []

        for i, label in enumerate(self.labels):
            c55, c66 = self.cijs[i]['c55'], self.cijs[i]['c66']
            null_float = ufloat(np.nan, np.nan)
            if c55!=null_float and c66!=null_float:
                gamma = (c66 - c55) / (2*c55)
            else:
                gamma = null_float

            gammas.append(gamma)

        self.data['gamma'] = dict(zip(self.labels, gammas))
        self.save_rp_data()

        return [val.n for val in gammas], [val.s for val in gammas]


def construct_latex_table(master_dir, samples, labels, variables,
                        table_name):
    '''
    A function to consrtuct a latex table from the 
    data contained in rp_data.p files. The data is assumed
    to be three-dimensional; that is, there are multiple 
    samples, each with multiple scans (described by labels),
    each with multiple variables. As such, at least one of
    these three parameters must be fixed to construct a table.

    Arguments:
        --master_dir: The master directory for all the data files
            (one above where the data files are saved)
        --samples: A string or list of strings giving the sample names
            to be shown in the table
        --labels: A string or list of strings giving the scan names
            to be shown in the table
        --variables: A string or list of strings gving the variables
            to be shown in the table
        --table_name: The name to call the table.

    Returns:
        None (the output is saved in a .txt file in the master directory)
    '''

    if isinstance(samples, str):
        indep = labels
        depnd = variables
        indexing_list = list(zip([samples]*len(indep)*len(depnd), depnd*len(indep), np.repeat(indep,len(depnd))))
    else:
        indep = samples
        if isinstance(labels, str):
            depnd = variables
            indexing_list = list(zip(np.repeat(indep,len(depnd)), depnd*len(indep), [labels]*len(indep)*len(depnd)))
        elif isinstance(variables, str):
            depnd = labels
            indexing_list = list(zip(np.repeat(indep,len(depnd)), [variables]*len(indep)*len(depnd), depnd*len(indep)))
        else:
            print('PlaceScan Analysis: One of samples, labels, or variables must be a single string for construct_latex_table')
            exit()

    values = []
    for i, keys in enumerate(indexing_list):
        sample_dict = retrieve_rp_data(master_dir+'{}/{}_rp_data.p'.format(keys[0], keys[0]))
        var_dict = sample_dict[keys[1]]  #This could be a dictionary of values of that variable at each scan, or a single value for all scans
        if isinstance(var_dict, dict):    
            value = var_dict[keys[2]]
        else:
            value = var_dict
        
        if str(value.n) != str(np.nan):
            values.append(r'{:FL}'.format(value))
        else:
            values.append(r'-')

    output_rows = []
    for i1 in range(len(indep)):
        new_row = r'{} & '.format(strc(indep[i1]))
        for i2 in range(len(depnd)):
            if i2 < len(depnd)-1:
                new_row += r'${}$ & '.format(str(values[i1*len(depnd)+i2]))
            else:
                new_row += r'${}$ \\'.format(str(values[i1*len(depnd)+i2]))
        output_rows.append(new_row+'\n')

    # Any latex formatting can be done in here.
    headers = ''.join([r'  & ']+list([r'{} & '.format(strc(var)) for var in depnd[:-1]])+[r'{} \\'.format(strc(depnd[-1]))])
    with open(master_dir+'/latex_tables/'+table_name+'.txt', 'w') as f:
        f.write(headers+'\n')
        f.write(r'\midrule'+'\n')
        f.writelines(output_rows)

    

def retrieve_rp_data(filename, key=None):
    '''
    A function to retrieve saved pickle data

    Arguments:
        --filename: The filename of the rp_data.p file
        --key: An optional key to retrieve just the 
            data contained in the dictionary corresponding
            to the key.

    Returns:
        --data: The data dictionary.
    '''

    with open(filename, 'rb') as file:
        
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            data = {}
        
        if key:
            try:
                return data[key]
            except KeyError:
                return {}

    return data

def strc(string):
    '''
    Function to convert a dictionary key string into
    a latex-presentable string using the hard-coded
    string_conv_dict. This is used for the row and
    column headers, and are returned in bold font

    Arguments:
        --string: The string to convert

    Returns:
        --string: The converted string if a conversion exits.
            Otherwise, this is the original string (in bold)
    '''

    try:
        return r'\textbf{'+string_conv_dict[string]+r'}'
    except KeyError:
        return r'\textbf{'+string+r'}'


string_conv_dict = {'vp90':r'\boldmath$V_{P90}$ (m/s)','s0':r'\boldmath$V_{S0}$ (m/s)','sh90':r'\boldmath$V_{S90}$ (m/s)','vp0':r'\boldmath$V_{P0}$ (m/s)','mass':'Mass (g)','diameter':r'Diameter (mm)', 'height':r'Height (mm)','density':r'Density (g/cm\boldmath$^3$)','epsilon':r'\boldmath$\varepsilon$', 'anisotropy':r'Anisotropy (\%)', 'gamma':r'\boldmath$\gamma$', 'delta':r'\boldmath$\delta$', 'c11':r'\boldmath$c_{11}$ (GPa)', 'c33':r'\boldmath$c_{33}$ (GPa)', 'c13':r'\boldmath$c_{13}$ (GPa)', 'c55':r'\boldmath$c_{55}$ (GPa)', 'c66':r'\boldmath$c_{66}$ (GPa)', '7f-schist-2':'Schist', 'pmv1':'Protomylonite', 'umv':'Mylonite', '1a842':'Cataclasite'}