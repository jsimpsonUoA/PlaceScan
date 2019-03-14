'''
The true inversion code used for c13
after a lot of experimenting. It assumes
that inversion is only being made for
c13 and theta0. c55 and c66 are held constant,
and c11 and c33 are allowed to vary, but these
do not change much from the values calculated 
from the measured fast and slow P-waves (letting
c11 and c33 vary just avoids strnage results from
outlier velocities at the fast and slow positions).

See Tsvankin, "Seismic
Signatures and Analysis of Reflection Data in
Anisotropic Media", chapter 1 for the equations 
used in this inversion.

The method follows that in Blum et. al. (2013),
"Noncontacting benchtop measurements of the 
elastic properties of shales"
'''

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev
from uncertainties import ufloat
from scipy.odr import ODR, RealData, Model

import matplotlib.pyplot as plt

import PlaceScan.plotting as plot


class C13Calculator():
    def __init__(self, group_vels, group_angles, c11, c33, c55, c66,
                 density, theta90=None, diameter=1.0, plot_c13=True):
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
            --theta90: The group angle at which the c11 value was
                taken (i.e. the fast P-wave direction). If None,
                the location of the fastest group velocity is used.
                Must be in degrees.
            --diameter: The diameter of the sample (in mm)
        '''

        self.theta90 = group_angles[np.argmax(group_vels)]
            
        if min(group_angles) <= self.theta90 + 90. <= max(group_angles):
            self.theta0 = (self.theta90 + 90.0)  / 180. * np.pi
        else:
            self.theta0 = (self.theta90 - 90.0)  / 180. * np.pi

        self.diameter = diameter.n / 1000.
        self.group_vels = np.array([group_vel.n for group_vel in group_vels])
        self.group_vel_err = np.array([group_vel.s for group_vel in group_vels])
        self.group_angles = np.deg2rad(group_angles) # - self.theta0
        self.c11, self.c11_err = c11.n * 1e9, c11.s * 1e9
        self.c33, self.c33_err = c33.n * 1e9, c33.s * 1e9
        self.c55, self.c55_err = c55.n * 1e9, c55.s * 1e9
        self.c66, self.c66_err = c66.n * 1e9, c66.s * 1e9
        self.theta0_err = self.theta0 * 0.03  # 3% error
        self.fig = None
        
        self.density = density.n
        # An array of phase angles. Number of points must be even when including the end point.
        self.phase_ang = np.linspace(0.0, np.pi, 30000) 
        self.counter = 0

        #Invert for c13
        self.find_c13()

        #Check if c13 falls within reasonable limits
        limits = 'NO'
        if (self.c13_max-self.c13 > 0.) and (self.c13-self.c13_min > 0.):
            limits = 'YES'

        #Put c13 into a ufloat form
        self.c13 = ufloat(self.c13/ 1e9, self.c13_err/ 1e9) 

        #Print the Results:
        print('c13 Inversion Results: ({} Interations)'.format(self.counter))
        print('c13: {0}. Initial Guess: {1}'.format(self.c13, round(self.c13_0/1e9,1)))
        print('Min c13: {0}, Best Estimate: {1}, Max c13: {2}'.format(round(self.c13_min/1e9,1), round(self.c13.n,1), round(self.c13_max/1e9,1)))
        print('c13 Falls in Limits? {}'.format(limits))
        print()

        #Plot the results:
        if plot_c13:
            self.plot_result(ylim=(3000,6000))

    def find_c13(self):
        '''
        Find the c13 parameter by fitting a curve to 
        the measured group velocities using the scipy
        ODR library. Since c13 is par tof the equation
        for the curve, the c13 value corresponding to
        the best fit curve is the result of the inversion.
        '''

        self.c13_max = np.sqrt(self.c33*(self.c11))       #The most physically reasonable maximum for c13
        self.c13_min = np.sqrt(self.c33*(self.c11-2*self.c66)+self.c66**2)-self.c66
        self.c13_0 = self.c13_min*1.2

        model = Model(self.forward_model,estimate=[self.c13_0, self.theta0, self.c11, self.c33])
        data = RealData(self.group_angles, self.group_vels, sy=self.group_vel_err)
        fit = ODR(data, model, beta0=[self.c13_0, self.theta0, self.c11, self.c33], ifixb=[1,1,1,1])
        output = fit.run()

        best_fit = output.beta
        errors = output.sd_beta

        self.c13 = best_fit[0]
        self.c13_err = errors[0]
        self.theta0 = best_fit[1]

    def forward_model(self, parameters, xdata):
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

        c13, theta, c11, c33 = parameters
        self.c11, self.c33 = c11, c33

        phase_vel = self.phase_vel(c13)
        phase_vel_deriv = self.phase_vel_derivative(c13)
        mod_group_vels = self.model_group_vel(phase_vel, phase_vel_deriv)
        mod_group_ang = self.model_group_anlges(phase_vel, phase_vel_deriv) + theta

        #Interpolate the model group angles to be at the same points as the measured angles.
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
        lower bounds for the inversion of the c13 and theta0.
        upper_bounds is similar.

        num_points is the number of points evaluated in
        the cost function.
        '''

        side_length = int(np.sqrt(num_points))

        c13s = np.linspace(lower_bounds[0],upper_bounds[0],side_length)
        theta0s = np.linspace(lower_bounds[1],upper_bounds[1],side_length)

        cost_function = np.zeros((side_length,side_length))
        for y, c13 in enumerate(c13s):
            for x, theta0 in enumerate(theta0s):
                mod_group_vels = self.forward_model((c13,theta0, self.c11, self.c33),self.group_angles)

                #Do least-squares
                cost = np.sum(np.square(mod_group_vels - self.group_vels))
                cost_function[y,x] = cost

        #Do an imshow
        plt.imshow(cost_function/1e9,cmap='viridis',extent=[lower_bounds[1],upper_bounds[1],upper_bounds[0]/1e9,lower_bounds[0]/1e9],aspect='auto')
        plt.xlabel('theta0'), plt.ylabel('c13 (GPa)')
        plt.show()


    def plot_result(self, show=True, fig=None, **kwargs):
        '''
        A function to plot the result of the inversion,
        showing the original data and the best fit
        model from the inversion
        '''

        if not fig:
            self.fig = plt.figure(figsize=(8,4.5))
        else:
            self.fig = fig
        self.fig = plot.simple_plot_points(np.rad2deg(self.group_angles), self.group_vels, errors=self.group_vel_err, shaded_errors=True, marker='x',linestyle='-',fig=self.fig,show=False)
        self.fig = plot.simple_plot_points(np.rad2deg(self.group_angles), self.forward_model((self.c13.n*1e9, self.theta0,self.c11,self.c33),self.group_angles),
                            linestyle='-',fig=self.fig,lab_font=16.,ylab='P-wave Group Velocity (m/s)',xlab='Group Angle (degrees)', show=False, marker='', **kwargs)
        self.fig.savefig('v{}.png'.format(round(self.c33/1e9)))
        if show:    
            plt.show()


    def calcualte_at_45deg(self, vel):
        '''
        Function which caluclates the group angles
        and velocities for a given P-wave velocity as if that
        velocity was obtained at 45 degrees to the symmetry
        axis.
        '''
        print(vel)
        c11, c33, c55, c66 = ufloat(self.c11,self.c11_err), ufloat(self.c33,self.c33_err),ufloat(self.c55,self.c55_err),ufloat(self.c66,self.c66_err)
        c13_p1 = (c11+c55-2*self.density*vel**2)
        c13_p2 = (c33+c55-2*self.density*vel**2)
        c13 = (c13_p1*c13_p2)**0.5 - c55
        print(c13/1e9)
        
        mod_group_vels = self.forward_model((c13.n, self.theta0,self.c11,self.c33),self.group_angles)

        return self.group_angles, mod_group_vels, c13