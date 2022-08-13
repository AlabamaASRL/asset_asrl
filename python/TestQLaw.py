# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:56:38 2021

@author: Jared
"""
from QLaw import LyapSteer
from AstroModels import TwoBody
# Plotting imports
import asset as ast
import spiceypy as spice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import MKgSecConstants as c
import Date

################################################################################
# Setup
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
TModes = oc.TranscriptionModes
spice.furnsh("BasicKernel.txt")

def GetEphemTraj(body,startDay,endDay,numstep,LU=1.0,TU=1.0,\
    Frame='ECLIPJ2000',Center='SOLAR SYSTEM BARYCENTER'):
    startET = spice.str2et(startDay)
    endET   = spice.str2et(endDay)
    times = [startET + (endET - startET)*x/numstep for x in range(numstep)]
    
    states=[]
    t0 = times[0]
    for t in times:
        X = np.zeros((7))
        X[0:6] = spice.spkezr(body,t,Frame,'NONE',Center)[0]
        X[0:3] *= 1000.0/LU # Convert to m
        X[3:6] *= 1000.0*TU/LU
        X[6] = (t-t0)/TU
        states.append(X)
    return states

class LTModel(oc.ode_x_u.ode):
    def __init__(self, mu, ltacc):
        Xvars = 6
        Uvars = 3
        ############################################################
        args = oc.ODEArguments(Xvars, Uvars)
        r = args.head3()
        v = args.segment3(3)
        u = args.tail3()
        g = r.normalized_power3() * (-mu)
        thrust = u * ltacc
        acc = g + thrust
        ode = vf.stack([v, acc])
        #############################################################
        super().__init__(ode, Xvars, Uvars)

if __name__ == "__main__":
 
    lstar     = c.AU
    vstar     = np.sqrt(c.MuSun/lstar)
    tstar     = np.sqrt(lstar**3/c.MuSun)
    astar     = c.MuSun/lstar**2
      
    engacc = (1./1800)/astar

    YEAR = 365*24*3600/tstar
    data_start    = 'Nov 2, 2024'
    data_end      = 'Jan 1, 2049'
    datapoints = 15000
    
    EarthDat  = GetEphemTraj("EARTH",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    MarsDat = GetEphemTraj("MARS BARYCENTER",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    MercuryDat = GetEphemTraj("MERCURY BARYCENTER",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    VenusDat   = GetEphemTraj("VENUS BARYCENTER",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    PH27Dat = GetEphemTraj("54186922",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    

    
    t0 = spice.utc2et(data_start)
    te = spice.utc2et(data_end)-t0
    
    EarthTab  = oc.LGLInterpTable(6,EarthDat,datapoints)
    MarsTab = oc.LGLInterpTable(6,MarsDat,datapoints)
    MercuryTab = oc.LGLInterpTable(6,MercuryDat,datapoints)
    VenusTab   = oc.LGLInterpTable(6,VenusDat,datapoints)
    PH27Tab = oc.LGLInterpTable(6,PH27Dat,datapoints)
    
    ED =  np.array(EarthDat).T
    MD = np.array(MercuryDat).T
    VD =  np.array(VenusDat).T
    PD =  np.array(PH27Dat).T
    MARS = np.array(MarsDat).T
    
    ode = LTModel(1.0, .1)
    integ = ode.integrator(.01, LyapSteer(PH27Tab), range(0, 10))
    #integ.Adaptive = True
    Initstate = np.zeros(10)
    Initstate[0] = 1.0
    Initstate[4] = 1.0
    Initstate[7] = 1.0

    traj = integ.integrate_dense(Initstate, 15, 500)
    TT = np.array(traj).T


    fig = plt.figure(figsize = (9, 9))
    axes = fig.add_subplot(projection = "3d")
    axes.set(xlim=(-1, 1), ylim = (-1, 1), zlim = (-1, 1))
    axes.plot(ED[0]*lstar/c.AU,ED[1]*lstar/c.AU,ED[2]*lstar/c.AU,color ='g')
    axes.plot(VD[0]*lstar/c.AU,VD[1]*lstar/c.AU,VD[2]*lstar/c.AU,color = 'r')
    axes.plot(MARS[0]*lstar/c.AU,MARS[1]*lstar/c.AU,MARS[2]*lstar/c.AU,color = 'purple')
    axes.plot(MD[0]*lstar/c.AU,MD[1]*lstar/c.AU,MD[2]*lstar/c.AU,color = 'blue')
    axes.plot(PD[0]*lstar/c.AU,PD[1]*lstar/c.AU,PD[2]*lstar/c.AU,color = 'k')
    axes.plot(TT[0], TT[1], TT[2])
    
    fig2 = plt.figure(figsize = (9, 9))
    axes2 = fig2.add_subplot()
    axes2.plot(TT[6], TT[7])
    axes2.plot(TT[6], TT[8])
    axes2.plot(TT[6], TT[9])
    
    plt.show()
    
    
    