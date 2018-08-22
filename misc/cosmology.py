import numpy as np

class AngularDiameterDistance(object):
    
    def __init__(self):

        self.z_max = 15.0
        self.dz = 1e-3
        self.Omega_m = 0.27
        self.Omega_Lambda = 0.73
        self.c = 299792 # in km/s
        self.H_0 = 70   # in km/s/Mpc
        
        self.ang_dist_z = np.arange(0, self.z_max, self.dz)
        self.ang_dist_integral = np.cumsum(self.dz / self.hubble_parameter(self.ang_dist_z))
    
    def hubble_parameter(self, z):
        return self.H_0 * (self.Omega_m * (1.0 + z)**3 + self.Omega_Lambda)**0.5
    
    def evaluate(self, z_1, z_2):        
        da_z1 = self.evaluate_0(z_1)
        da_z2 = self.evaluate_0(z_2)
        da_z1_z2 = da_z2 - ((1.0 + z_1) / (1.0 + z_2))*da_z1
        return da_z1_z2
        
    def evaluate_0(self, z_2):        
        da_z2 = (self.c / (1.0 + z_2)) * np.interp(z_2, self.ang_dist_z, self.ang_dist_integral)
        return da_z2