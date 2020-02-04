'''
This script performs an inversion for the c13
elastic stiffness value in a VTI rock sample
using laser ultrasound data recorded at different
angles around the sample. The method follows that
outlined in: Blum et. al. (2013), Noncontacting 
benchtop measurements of the elastic properties of
shales. The theory is presented in more detail in
Tsvankin (2012), Seismic Signatures and Analysis
of Reflection Data in Anisotropic Media.

The input requires the elastic constants c11, c33, 
and c55, as well as the group velocities around the
sample

THIS IS AN EXPERIMENTAL VERSION. SEE true_c13 FOR
THE OPERTATIONAL VERSION.

Written by Jonathan Simpson, UoA PALab, December 2018
'''

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev
from uncertainties import ufloat

import matplotlib.pyplot as plt

import placescan.plotting as plot


class C13CalculatorExp():
    def __init__(self, group_vels, group_angles, c11, c33, c55, c66,
                 density, parameters_to_invert=[], theta90=None,
                 diameter=1.0):
        '''
        C13Calculator is a class to calculate the c13 elastic
        modulus for VTI media by fitting a curve to known
        group velocities. This is done using the theory for 
        TI media described in chpater 1 of Tsvankin, with a maximum
        bound on the c13 parameter given by equation 1.52

        Arguments:
            --group_vels: An array of P-wave group velocities as a
                (as a function of angle)
            --group_angles: An array of equal length to group_vels
                containing the anlges for the velocity values. The
                angles must be in degrees.
            --c11: The c11 elastic modulus for the sample (in GPa)
            --c33: The c33 elastic modulus for the sample (in GPa)
            --c55: The c55 elastic modulus for the sample (in GPa)
            --c66: The c66 elastic modulus for the sample (in GPa)
            --density: The density of the sample (in kg/m^3)
            --parameters_to_invert: A list of parameters to invert for
                in the curve fit. These can be one or more of 'c11',
                'c33', 'c55', 'c66', 'theta0'. Obviously, the inversion
                is less constrained the more parameters you invert for.
            --theta90: The group angle at which the c11 value was
                taken (i.e. the fast P-wave direction). If None,
                the location of the fastest group velocity is used.
                Must be in degrees.
            --diameter: The diameter of the sample (in mm)
        '''

        #Fast direction is usually better constrained. To get the slow direction,
        #we find the fast direction and take theta = 0 to be 90 degrees from that
        if theta90:
            self.theta90 = group_angles[np.argmin(np.abs(group_angles-theta90))]
        else:
            self.theta90 = group_angles[np.argmax(group_vels)]
            
        if min(group_angles) <= self.theta90 + 90. <= max(group_angles):
            self.theta0 = (self.theta90 + 90.0)  / 180. * np.pi
        else:
            self.theta0 = (self.theta90 - 90.0)  / 180. * np.pi

        self.parameters_to_invert = parameters_to_invert
        self.diameter = diameter.n / 1000.
        self.group_vels = np.array([group_vel.n for group_vel in group_vels])
        self.group_vel_err = np.array([group_vel.s for group_vel in group_vels])
        self.group_angles = np.deg2rad(group_angles) # - self.theta0
        self.c11, self.c11_err = c11.n * 1e9, c11.s * 1e9
        self.c33, self.c33_err = c33.n * 1e9, c33.s * 1e9
        self.c55, self.c55_err = c55.n * 1e9, c55.s * 1e9
        self.c66, self.c66_err = c66.n * 1e9, c66.s * 1e9
        self.theta0_err = self.theta0 * 0.03  # 3% error

        self.density = density.n
        self.ep = (self.c11 - self.c33) / (2*self.c33)   #Epsilon parameter

        # An array of phase angles. Number of points must be even when including the end point.
        self.phase_ang = np.linspace(0.0, np.pi, 30000) 
        self.counter = 0

        #Invert for c13
        self.find_c13()


        plt.show()

    def find_c13(self):
        '''
        Function which inverts for the c13 value by fitting
        a curve in the form given for TI media in Tsvankin
        to the measured data. This function returns the best
        fit c13.
        '''

        c13_max = np.sqrt(self.c33*(self.c11-self.c66))
        c13_min = np.sqrt(self.c33*(self.c11-2*self.c66)+self.c66**2)-self.c66
        initial_guess = c13_min*1.1#(c13_max + c13_min) / 2
        
        self.c55 = self.c66*0.75   #REMOVE THIS LINE. I'M FORCING C55 NOT TO BE INVERTED
        c11_min,c33_min,c55_min,c66_min,theta_min = self.c11-1.,self.c33-1.,self.c66-1.,self.c66-1.,self.theta0-1.e-4
        c11_max,c33_max,c55_max,c66_max,theta_max = self.c11+1.,self.c33+1.,self.c66+1.,self.c66+1.,self.theta0+1.e4
        if 'c11' in self.parameters_to_invert:
            c11_min, c11_max = 0, np.inf
            if np.isnan(self.c11):
                self.c11 = 50e9
        if 'c33' in self.parameters_to_invert:
            c33_min, c33_max = 0, np.inf
            if np.isnan(self.c33):
                self.c33 = self.c11*.75
        if 'c55' in self.parameters_to_invert:
            c55_min, c55_max = 0.,self.c66*2.#self.c66*2., self.c66
            if np.isnan(self.c55):
                self.c55 = self.c66*0.75
        if 'c66' in self.parameters_to_invert:
            c66_min, c66_max = 0, np.inf
            if np.isnan(self.c66):
                self.c66 = self.c55
        if 'theta0' in self.parameters_to_invert:
            theta_min, theta_max = -2*np.pi, 2*np.pi
            if np.isnan(self.theta0):
                self.theta0 = 0.

        #self.plot_cost_function([c13_min,c55_min],[c13_max,c55_max])
        #scipy.curve_fit inversion using the exact equations for phase velocity
        #best_fit, covariance = curve_fit(self.forward_model, self.group_angles, self.group_vels, 
        #         p0=[initial_guess, self.c11, self.c33, self.c55, self.c66, self.theta0], 
        #         bounds=([c13_min/2,c11_min,c33_min,c55_min/2,c66_min,theta_min],[c13_max,c11_max,c33_max,c55_max,c66_max,theta_max]), 
        #         max_nfev=1000, xtol=1e-13, ftol=1e-13, gtol=1e-13,method='trf')

        #errors = np.sqrt(np.diagonal(covariance))


        # ODR inversion using exact equations for phase velocity
        best_fit, errors = self.odr_inversion()


        # Inversion using strong anisotropy approximation for phase velocity
        #best_fit, errors = self.strong_anisotropy_inversion()
        #self.c13 = best_fit[0]
        #self.c13_err = errors[0]         
        
        self.c13, self.c55, self.c66 = best_fit[0], best_fit[3], best_fit[4]
        self.c13_err,  self.c55_err,  self.c66_err = errors[0], errors[3], errors[4]

        print(round(self.c13/1e9),round(self.c13_err/1e9),round(self.c55/1e9),round(self.c55_err/1e9))
        print('Runs: {0}; c13: {1}; c11: {2}; c33: {3}; c55: {4}; c66: {5}'.format(self.counter, round(self.c13/1e9), round(self.c11/1e9), round(self.c33/1e9), round(self.c55/1e9), round(self.c66/1e9)))
        #print(np.sqrt(cov[0])/1e9,np.sqrt(cov[2])/1e9,np.sqrt(cov[3])/1e9)

        fig=plt.figure(figsize=(8,4.5))
        fig = plot.simple_plot_points(np.rad2deg(self.group_angles), self.group_vels, marker='x',linestyle='',color=plot.colors[0], fig=fig,show=False)
        fig = plot.simple_plot_points(np.rad2deg(self.group_angles), self.forward_model(self.group_angles,self.c13, best_fit[1], best_fit[2],best_fit[3], best_fit[4],best_fit[5]),
                            color=plot.colors[1],linestyle='-',fig=fig,lab_font=16.,ylab='P-wave Group Velocity (m/s)',xlab='Group Angle (degrees)',
                            save_dir=None)#'7f-schist-2_inversrion_{}.pdf'.format(int(self.c11/1e9)))
        

        self.c13 = ufloat(self.c13/ 1e9, self.c13_err/ 1e9) 
        self.c55 = ufloat(self.c55/ 1e9, self.c55_err/ 1e9) 
        self.c66 = ufloat(self.c66/ 1e9, self.c66_err/ 1e9) 
        print(self.c13, initial_guess/1e9)
        
    def odr_inversion(self, initial_guess=None):
        '''
        Just another curve fit/inversion method
        to find c13, using scipy's ODR class. Does
        the same job as curve_fit.
        '''

        def odr_forward_model(parameters, xdata):
            c13, c11, c33, c55, c66, theta = parameters
            return self.forward_model(xdata, c13, c11, c33, c55, c66, theta)

        from scipy.odr import ODR, RealData, Model

        c13_max = np.sqrt(self.c33*(self.c11-self.c66))
        c13_min = np.sqrt(self.c33*(self.c11-2*self.c66)+self.c66**2)-self.c66
        if not initial_guess:    
            initial_guess = c13_min*1.1
        print(self.c11, self.c33, self.c55, self.c66, self.theta0)
        print('ok',round(c13_min/1e9),round(c13_max/1e9))

        parameters = ['c11','c33', 'c55', 'c66', 'theta0']
        fixed_params = [0,0,0,0,0]
        for i, item in enumerate(parameters):
            if item in self.parameters_to_invert:
                fixed_params[i] = 1
        fixed_params = [1,1,1,0,0,1] #MAKE THIS [1]+fixed_params. I'M FORCING C55 NOT TO BE INVERTED

        model = Model(odr_forward_model,estimate=[initial_guess, self.c11, self.c33, self.c55, self.c66, self.theta0])
        data = RealData(self.group_angles, self.group_vels, sy=self.group_vel_err)
        fit = ODR(data, model, 
                  beta0=[initial_guess, self.c11, self.c33, self.c55, self.c66, self.theta0],
                  ifixb=fixed_params)
        output = fit.run()
        best_fit = output.beta
        errors = output.sd_beta

        return best_fit, errors

    def forward_model(self, xdata, c13, c11, c33, c55, c66, theta):
        '''
        The forward model function used in the nonlinear
        inversion curve fitting. The function calculates
        the group velocity for a given c13 parameter.

        Because the model group angles are also dependent on
        the value of c13, we cannot simply use xdata (which 
        are the provided measured group angles) in the forward
        model. As such, the model group anlges for the given
        c13 are calculated, and then interpolated onto the xdata
        points.
        '''
        self.counter += 1
        if 'c11' in self.parameters_to_invert:
            self.c11 = c11
        if 'c33' in self.parameters_to_invert:
            self.c33 = c33
        if 'c55' in self.parameters_to_invert:
            self.c55 = self.c55#MAKE THIS C55 AGSAIN c55. I'M FORCING C55 NOT TO BE INVERTED
        if 'c66' in self.parameters_to_invert:
            self.c66 = c66
        if 'theta0' in self.parameters_to_invert:
            theta_to_add = theta
        else:
            theta_to_add = self.theta0

        phase_vel = self.phase_vel(c13)
        phase_vel_deriv = self.phase_vel_derivative(c13)
        mod_group_vels = self.model_group_vel(phase_vel, phase_vel_deriv)
        mod_group_ang = self.model_group_anlges(phase_vel, phase_vel_deriv) + theta_to_add
        
        tck = splrep(mod_group_ang, mod_group_vels, s=0)
        mod_group_vels = splev(xdata, tck, der=0)
        
        #plt.plot(xdata, mod_group_vels,'-r')
        #plt.plot(xdata, self.group_vels, '-b')
        #plt.show()

        return mod_group_vels

    def strong_anisotropy_inversion(self):
        '''
        A function to perform an ODR inversion with 
        the strong anisotropy approximation. This method
        inverts for c13 and c55, and follows the
        methodology of Xie, et. al. (2016), "Experimental
        and theoretical enhancement of the inversion 
        accuracy of the Thomsen parameter δ in organic-rich
        shale"
        '''

        def odr_forward_model(parameters, xdata):
            delta, theta_m, theta = parameters
            return self.strong_anisotropy_forward_model(xdata, delta, theta_m, theta)

        def odr_forward_model2(xdata, delta, theta_m, theta):
            return self.strong_anisotropy_forward_model(xdata, delta, theta_m, theta)            

        from scipy.odr import ODR, RealData, Model

        c13_max = np.sqrt(self.c33*self.c11)
        c13_min = np.sqrt(self.c33*(self.c11-2*self.c66)+self.c66**2)-self.c66   
        c13_0 = c13_max*1.1#(c13_max + c13_min) / 2

        delta = lambda x: ((x+ self.c55)**2 - (self.c33-self.c55)**2) / (2*self.c33*(self.c33-self.c55))

        theta_m0 = np.arctan(np.sqrt((self.c33 - self.c55) / (self.c11 - self.c55)))
        delta0 = delta(c13_0)
        delta_min = -delta(c13_min)
        delta_max = delta(c13_max)

        print('ok',round(c13_min/1e9),round(c13_max/1e9))

        parameters = ['delta','theta_m', 'theta0']
        fixed_params = [1,1,0]
        if 'theta0' in self.parameters_to_invert:
            fixed_params = [1,1,1]

        #model = Model(odr_forward_model)
        #data = RealData(self.group_angles, self.group_vels, sy=self.group_vel_err)
        #fit = ODR(data, model, 
        #          beta0=[delta0, theta_m0, self.theta0],
        #          ifixb=fixed_params)
        #output = fit.run()

        ###  ODR inversion  ###
        #best_fit = output.beta
        #errors = output.sd_beta
        #######################

        ####  scipy.curve_fit inversion  ####
        best_fit, cov = curve_fit(odr_forward_model2, self.group_angles, self.group_vels, 
                 p0=[delta0, theta_m0, self.theta0], 
                 bounds=([delta_min, -np.pi/2, 0.],[delta_max, np.pi/2, 2*np.pi]), 
                 max_nfev=1000, xtol=1e-13, ftol=1e-13)       
        errors = np.sqrt(np.diagonal(cov))
        ######################################

        print([delta0, theta_m0, self.theta0], best_fit)

        self.theta0 = best_fit[2]
        self.c55 = (self.c33-self.c11*(np.tan(best_fit[1])**2))/(1-np.tan(best_fit[1])**2)
        c13 = lambda x: np.sqrt(x * (2*self.c33*(self.c33-self.c55)) + (self.c33-self.c55)**2) - self.c55

        c13_err = np.abs(c13(best_fit[0]-errors[0]) - c13(best_fit[0]+errors[0])) / 2.

        plt.plot(self.group_angles, self.group_vels, 'xb')
        plt.plot(self.group_angles, self.strong_anisotropy_forward_model(self.group_angles,best_fit[0], best_fit[1], best_fit[2]))
        #plt.(3000,6000)

        return [c13(best_fit[0])], [c13_err]


    def strong_anisotropy_forward_model(self, xdata, delta, theta_m, theta):
        '''
        A function to perform an inversion using the 
        strong anisotropy approximation. This method
        inverts for c13 and c55, and follows the
        methodology of Xie, et. al. (2016), "Experimental
        and theoretical enhancement of the inversion 
        accuracy of the Thomsen parameter δ in organic-rich
        shale"
        '''

        theta_to_add=0.
        if 'theta0' in self.parameters_to_invert:
            theta_to_add = theta

        #Calculate the phase velocity and is derivative
        numer = 2 * np.sin(theta_m)**2 * np.sin(self.phase_ang)**2 * np.cos(self.phase_ang)**2
        denom = 1 - np.cos(2*theta_m)*np.cos(2*self.phase_ang)
        last_term =  2*(self.ep-delta)*(numer/denom)
        second_term = 2*self.ep*np.sin(self.phase_ang)**2
        phase_vel_squared = (self.c33/self.density) * (1 + second_term - last_term)

        phase_vel = np.sqrt(phase_vel_squared)
        phase_vel_deriv = np.gradient(phase_vel)

        #Group velocities from phase velocities
        mod_group_vels = self.model_group_vel(phase_vel, phase_vel_deriv)
        mod_group_ang = self.model_group_anlges(phase_vel, phase_vel_deriv) + theta_to_add

        tck = splrep(mod_group_ang, mod_group_vels, s=0)
        mod_group_vels = splev(xdata, tck, der=0)

        return mod_group_vels

    def phase_vel(self, c13):
        '''
        Function which returns the phase velocity as a function
        of angle, using Equation 1.43 in Tsvankin and a given
        c13. Part of the forward model.
        '''

        first_term = (self.c11+self.c55) * np.sin(self.phase_ang)**2
        second_term = (self.c33+self.c55) * np.cos(self.phase_ang)**2
        third_term = self.sqrt_part(c13)

        phase_vel_squared = (first_term + second_term + third_term) / (2 * self.density)

        return np.sqrt(phase_vel_squared)

    def sqrt_part(self, c13):
        '''
        Function to calculate the last term (the square-root part)
        of equation 1.43 in Tsvankin (the phase velocity equation).
        '''
    
        sqrt_part = (self.c11-self.c55) * np.sin(self.phase_ang)**2
        sqrt_part = sqrt_part - (self.c33-self.c55) * np.cos(self.phase_ang)**2
        sqrt_part = sqrt_part**2 + 4 * (c13+self.c55)**2 * np.sin(self.phase_ang)**2 * np.cos(self.phase_ang)**2
        sqrt_part = np.sqrt(sqrt_part)

        return sqrt_part

    def phase_vel_derivative(self, c13):
        '''
        Function which returns the derivative of the phase with
        respect to phase angle, using the analytical derivative
        of Equation 1.43 in Tsvankin and a given c13. Part of
        the forward model.

        The expression I'm using is (in Latex):
        \[\frac{dV}{d\theta}=\frac{\sin\theta\cos\theta}{2\rho V}\left[c_{11}-c_{33}\pm\frac{1}{\sqrt{u}}\left(\left[c_{11}+c_{33}-2c_{55}\right]\left[\left(c_{11}-c_{55}\right)\sin^2\theta-\left(c_{33}-c_{55}\right)\cos^2\theta\right]+2\left(c_{13}+c_{55}\right)^2\left(\cos^2\theta-\sin^2\theta\right)\right)\right]\]
        where \(\quad u=\left[\left(c_{11}-c_{55}\right)\sin^2\theta-\left(c_{33}-c_{55}\right)\cos^2\theta\right]^2+4\left(c_{13}+c_{55}\right)^2\sin^2\theta\cos^2\theta\)
        '''           

        coeff = (np.sin(self.phase_ang)*np.cos(self.phase_ang)) / (2*self.density*self.phase_vel(c13))

        piece1 = self.c11+self.c33-2*self.c55
        piece2 = (self.c11-self.c55)*(np.sin(self.phase_ang)**2) - (self.c33-self.c55)*(np.cos(self.phase_ang)**2)
        piece3 = 2*(c13+self.c55)**2 * (np.cos(self.phase_ang)**2 - np.sin(self.phase_ang)**2)
        derivative_of_u = piece1 * piece2 + piece3

        main_term = self.c11 - self.c33 + (derivative_of_u/self.sqrt_part(c13))
        
        phase_vel_derivative = coeff * main_term
        
        return phase_vel_derivative

    def model_group_vel(self, phase_vel, phase_vel_deriv):
        '''
        Function to calculate the model group velocity,
        using equation 1.70 of Tsvankin and a given
        phase velocity
        '''

        model_group_vel = phase_vel * np.sqrt(1 + (phase_vel_deriv / phase_vel)**2)

        #Make three cycles worth
        model_group_vel = np.concatenate((model_group_vel[:-1], model_group_vel[:-1], model_group_vel))

        return model_group_vel

    def model_group_anlges(self, phase_vel, phase_vel_deriv):
        '''
        Function to calculate the model group angle,
        as a function of phase angle and velocity 
        using equation 1.70 of Tsvankin and a given 
        phase velocity.
        '''
        
        #Need to calculate this for 0-90 degrees, because of tangents
        first_quad_indices = np.where(self.phase_ang < np.pi/2.)
        first_quad_angles = self.phase_ang[first_quad_indices]      #Should be exactly half-length
        first_quad_vels = phase_vel[first_quad_indices]
        first_quad_deriv = phase_vel_deriv[first_quad_indices]
        quotient = first_quad_deriv / first_quad_vels

        #Go back to 0 to 180 degrees, based on the symmetry of VTI
        tan_group_angle = (np.tan(first_quad_angles) + quotient) / (1 - np.tan(first_quad_angles)*quotient)
        model_group_angles_first_quad = np.arctan(tan_group_angle)

        model_group_angles_second_quad = np.pi - model_group_angles_first_quad[::-1]
        model_group_angles = np.concatenate((model_group_angles_first_quad, model_group_angles_second_quad))
        
        #Make three complete cycles
        model_group_angles = np.concatenate((model_group_angles[:-1]-np.pi, model_group_angles[:-1], model_group_angles+np.pi))

        return model_group_angles

    def plot_cost_function(self,lower_bounds,upper_bounds,num_points=2500):
        '''
        Function to plot the cost function of 
        an inversion with two variables to get
        an idea of what the minimisation needs to
        obtain in the parameter space.

        lower_bounds is a two-element list with the
        lower bounds for the inversion of the c13 and c55.
        upper_bounds is similar.

        num_points is the number of points evaluated in
        the cost function.
        '''

        side_length = int(np.sqrt(num_points))

        c13s = np.linspace(lower_bounds[0],upper_bounds[0],side_length)
        c55s = np.linspace(lower_bounds[1],upper_bounds[1],side_length)

        cost_function = np.zeros((side_length,side_length))
        for y, c13 in enumerate(c13s):
            for x, c55 in enumerate(c55s):
                mod_group_vels = self.forward_model(self.group_angles,c13,self.c11,self.c33,c55,self.c66,self.theta0)
                #Do least-squares
                cost = np.sum(np.square(mod_group_vels - self.group_vels))
                cost_function[y,x] = cost
            #print(y)

        #Do an imshow
        plt.imshow(cost_function/1e9,cmap='viridis',extent=np.array([lower_bounds[1],upper_bounds[1],upper_bounds[0],lower_bounds[0]])/1e9,aspect='auto')
        plt.xlabel('c55 (GPa)'), plt.ylabel('c13 (GPa)')
        plt.show()