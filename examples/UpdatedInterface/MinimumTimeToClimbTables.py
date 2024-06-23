import numpy as np
import asset_asrl.VectorFunctions as vf
import matplotlib.pyplot as plt


'''
Aerodynamic Data taken from ICLOS2s implentation of the problem
http://www.ee.ic.ac.uk/ICLOCS/ExampleMinFuelClimb.html
'''

########################################################################################
## Aerodynamic Data for airplane as function of mach number
## Added in dummy data at Mach .6 and .75 to remove ossiclattions from fit
AeroMach = [     0,    0.4, .6,  .75,   0.8,    0.9,  1.0,    1.2, 1.4,   1.6, 1.8]
Clalpha =  [ 3.44,   3.44, 3.44,3.44,  3.44,   3.58, 4.44,   3.44, 3.01,  2.86, 2.44]
CD0     = [ 0.013, 0.013, 0.013,0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035]
eta     = [ 0.54,   0.54, 0.54,0.54, 0.54,   0.75, 0.79,   0.78, 0.89,   0.93, 0.93]
####################################################################################


####################################################################################
## Density and SOS data as function of altitude from 1976 US atmosphere model
AtmosData = [
    [-2000, 1.478e+00, 3.479e+02],
    [0, 1.225e+00, 3.403e+02],
    [2000, 1.007e+00, 3.325e+02],
    [4000, 8.193e-01, 3.246e+02],
    [6000, 6.601e-01, 3.165e+02],
    [8000, 5.258e-01, 3.081e+02],
    [10000, 4.135e-01, 2.995e+02],
    [12000, 3.119e-01, 2.951e+02],
    [14000, 2.279e-01, 2.951e+02],
    [16000, 1.665e-01, 2.951e+02],
    [18000, 1.216e-01, 2.951e+02],
    [20000, 8.891e-02, 2.951e+02],
    [22000, 6.451e-02, 2.964e+02],
    [24000, 4.694e-02, 2.977e+02],
    [26000, 3.426e-02, 2.991e+02],
    [28000, 2.508e-02, 3.004e+02],
    [30000, 1.841e-02, 3.017e+02],
    [32000, 1.355e-02, 3.030e+02],
    [34000, 9.887e-03, 3.065e+02],
    [36000, 7.257e-03, 3.101e+02],
    [38000, 5.366e-03, 3.137e+02],
    [40000, 3.995e-03, 3.172e+02],
    [42000, 2.995e-03, 3.207e+02],
    [44000, 2.259e-03, 3.241e+02],
    [46000, 1.714e-03, 3.275e+02],
    [48000, 1.317e-03, 3.298e+02],
    [50000, 1.027e-03, 3.298e+02],
    [52000, 8.055e-04, 3.288e+02],
    [54000, 6.389e-04, 3.254e+02],
    [56000, 5.044e-04, 3.220e+02],
    [58000, 3.962e-04, 3.186e+02],
    [60000, 3.096e-04, 3.151e+02],
    [62000, 2.407e-04, 3.115e+02],
    [64000, 1.860e-04, 3.080e+02],
    [66000, 1.429e-04, 3.044e+02],
    [68000, 1.091e-04, 3.007e+02],
    [70000, 8.281e-05, 2.971e+02],
    [72000, 6.236e-05, 2.934e+02],
    [74000, 4.637e-05, 2.907e+02],
    [76000, 3.430e-05, 2.880e+02],
    [78000, 2.523e-05, 2.853e+02],
    [80000, 1.845e-05, 2.825e+02],
    [82000, 1.341e-05, 2.797e+02],
    [84000, 9.690e-06, 2.769e+02],
    [86000, 6.955e-06, 2.741e+02]
]

AtmosData = np.array(AtmosData).T
alts =  AtmosData[0]
rhos = AtmosData[1]
soss = AtmosData[2]

####################################################################################
# Thrust vs Mach and altitude
# Solution wants to fly exactly at sea level. 
# Padded with negative altitude data to prevent reads outside of table bounds


ThrustMach = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
ThrustAlt = 304.8*np.array([-.5,0, 5, 10, 15, 20, 25, 30, 40, 50, 70]);

ThrustData = 4448.2 * np.array([
    [24.2, 24.2, 24.0, 20.3, 17.3, 14.5, 12.2, 10.2, 5.7, 3.4, 0.1],
    [28.0, 28.0, 24.6, 21.1, 18.1, 15.2, 12.8, 10.7, 6.5, 3.9, 0.2],
    [28.3, 28.3, 25.2, 21.9, 18.7, 15.9, 13.4, 11.2, 7.3, 4.4, 0.4],
    [30.8, 30.8, 27.2, 23.8, 20.5, 17.3, 14.7, 12.3, 8.1, 4.9, 0.8],
    [34.5, 34.5, 30.3, 26.6, 23.2, 19.8, 16.8, 14.1, 9.4, 5.6, 1.1],
    [37.9, 37.9, 34.3, 30.4, 26.8, 23.3, 19.8, 16.8, 11.2, 6.8, 1.4],
    [36.1, 36.1, 38.0, 34.9, 31.3, 27.3, 23.6, 20.1, 13.4, 8.3, 1.7],
    [36.1, 36.1, 36.6, 38.5, 36.1, 31.6, 28.1, 24.2, 16.2, 10.0, 2.2],
    [36.1, 36.1, 35.2, 42.1, 38.7, 35.7, 32.0, 28.1, 19.3, 11.9, 2.9],
    [36.1, 36.1, 33.8, 45.7, 41.3, 39.8, 34.6, 31.1, 21.7, 13.3, 3.1]
]).T

####################################################################

####################################################################
## Load Data into Tables

rhoTab = vf.InterpTable1D(alts,rhos,kind='cubic')
sosTab = vf.InterpTable1D(alts,soss,kind='cubic')
ClalphaTab = vf.InterpTable1D(AeroMach,Clalpha,kind='cubic')
etaTab = vf.InterpTable1D(AeroMach,eta,kind='cubic')
CD0Tab = vf.InterpTable1D(AeroMach,CD0,kind='cubic')
ThrustTab = vf.InterpTable2D(ThrustMach,ThrustAlt,ThrustData,kind='cubic')

if __name__ == "__main__":
    
    from matplotlib import cm

    Mach_t = np.linspace(0,1.8,1000)
    alt_t = np.linspace(0,ThrustAlt[-1],1000)
    
    Ms,As = np.meshgrid(Mach_t,alt_t)
    
    fig1,axs1 = plt.subplots(3,1)
    
    axs1[0].plot(Mach_t,ClalphaTab(Mach_t).T)
    axs1[1].plot(Mach_t,etaTab(Mach_t).T)
    axs1[2].plot(Mach_t,CD0Tab(Mach_t).T)
    
    axs1[0].scatter(AeroMach,Clalpha)
    axs1[1].scatter(AeroMach,eta)
    axs1[2].scatter(AeroMach,CD0)
    
    
    axs1[0].set_ylabel("CLalpha")
    axs1[1].set_ylabel("eta")
    axs1[2].set_ylabel("CD0")
    axs1[2].set_xlabel("Mach")

    fig2,axs2 = plt.subplots(2,1)

    axs2[0].plot(alt_t,rhoTab(alt_t).T)
    axs2[1].plot(alt_t,sosTab(alt_t).T)
    
    axs2[0].set_ylabel("Density(kg/m^3)")
    axs2[1].set_ylabel("SOS(m/s)")
    axs2[1].set_xlabel("Altitude (m)")
    
    fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})
    cs =  ax3.plot_surface(Ms,As,ThrustTab(Ms,As),cmap=cm.coolwarm)
    
    ax3.set_xlabel("Mach")
    ax3.set_ylabel("Altitude (m)")
    ax3.set_zlabel("Thrust (N)")

    plt.show()
    
    
    
    
    
    

    
    
    
    
    





