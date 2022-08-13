# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 12:27:29 2021

@author: Jared
"""
import asset as ast

import spiceypy as spice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import MKgSecConstants as c
import Date

# Setup
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
TModes = oc.TranscriptionModes
spice.furnsh("BasicKernel.txt")

class Kepler(oc.ode_6.ode):
    def __init__(self, mu):
        Xvars = 6
       
        ############################################################
        args = oc.ODEArguments(Xvars)
        r = args.head3()
        v = args.segment3(3)
        g = r.normalized_power3() * (-mu)
        ode = vf.stack([v, g])
        #############################################################
        super().__init__(ode, Xvars)
        
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

def FlybyAngleBound(tab,mubod,minrad):
    X = Args(8)
    v0 = X.head3()
    t0 = X[3]
    
    v1 = X.tail(4).head3()
    t1 = X.tail(4)[3]
    
    BodyV =  oc.InterpFunction(tab,range(3,6)).vf()
    v0dir =(BodyV.eval(t0)-v0).normalized()
    v1dir =(BodyV.eval(t1)-v1).normalized()
    
    vInf2  =(BodyV.eval(t0)-v0).squared_norm()
    
    delta      = vf.arccos(vf.dot(v0dir,v1dir))
    deltaMax   = np.pi - 2*vf.arccos(1.0/(1.0 + minrad*vInf2/mubod)) 
    return   (delta - deltaMax)

def AltitudeCon(tab, minrad):
    X = Args(4)
    x = X.head(3)
    
    BodyP =oc.InterpFunction(tab, range(0, 3)).vf()
    Bodypos = BodyP.eval(X.tail(1))
    
    dist = vf.norm(x - Bodypos)
    
    return dist-minrad

def GenTable(bodies, startdate, enddate, lstar, tstar, mu = 1.0, datapoints = 15000):
    #assign tables to each body
    Ubodies = np.unique(bodies)
    bodydata = []
    bodytab = {}
    for body in Ubodies:
        bodydata = GetEphemTraj(body, startdate, enddate, datapoints, LU=lstar, TU = tstar, Center="SUN")
        bodytab[body] = [oc.LGLInterpTable(6, bodydata, datapoints)]
    return bodytab
    
        
def GenFlyby(bodies, minrad, startdate, enddate, ToF, lstar, tstar, bodytab, mu = 1.0):
    #bodies is number of flybys, start is first body in bodies
    #ToF is the list of time of flights for each arc
    Ubodies = np.unique(bodies)
    bodydata = []
    
    #relative time of flight from beginning of sequence
    tofrel = []
    tofrel.append(0)
    for time in ToF:
        tofrel.append(tofrel[-1] + time)
    
    #state at for each tof of the relevant bodies
    bodystates = []
    for i, body in enumerate(bodies):
        state = bodytab[body][0].Interpolate(tofrel[i])
        bodystates.append(state)
    
    ode   = ast.Astro.Kepler.ode(mu)
    kpint = ode.integrator(.001)
    
    full_seq = []
    state1 = bodystates[0]
    DVs = []
    for i, body in enumerate(bodies[:-1]):
        
        state2 = bodystates[i+1]
        tof = state2[6] - state1[6]
        v1, v2 = ast.Astro.lambert_izzo(state1[:3], state2[:3], tof, mu, True)
        DV = np.linalg.norm(v1-state1[3:6]) + np.linalg.norm(v2-state2[3:6])
        DVs.append(DV)
        
        propstate = np.zeros(7)
        propstate[:3] = state1[:3]
        propstate[3:6] = v1
        propstate[6] = tofrel[i]
        
        arc = kpint.integrate_dense(propstate, tofrel[i+1], 500)
        
        phase = ode.phase(TModes.CentralShooting)
        phase.setTraj(arc, 500)
        phase.addBoundaryValue(PhaseRegs.Front,[0, 1, 2, 6],
                               [propstate[0], propstate[1], propstate[2], propstate[6]])
        
        phase.addInequalCon(PhaseRegs.Back, AltitudeCon(bodytab[bodies[i+1]][0], minrad[i+1]), [0, 1, 2, 6])
        #phase.solve()
        state1 = phase.returnTraj()[-1]
        full_seq.append(phase.returnTraj())
        
    return full_seq, DVs
        

lstar     = c.AU
vstar     = np.sqrt(c.MuSun/lstar)
tstar     = np.sqrt(lstar**3/c.MuSun)
astar     = c.MuSun/lstar**2
  
engacc = (1./1800)/astar

data_start = "Jan 1, 2024"
data_end = "Jan 1, 2050"
datapoints = 15000

EarthDat  = GetEphemTraj("EARTH",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
MarsDat = GetEphemTraj("MARS BARYCENTER",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
VenusDat   = GetEphemTraj("VENUS BARYCENTER",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
PH27Dat = GetEphemTraj("54186922",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SOLAR SYSTEM BARYCENTER")


ED =  np.array(EarthDat).T
MD = np.array(MarsDat).T
VD =  np.array(VenusDat).T
PD =  np.array(PH27Dat).T
    
bodies = ["EARTH", "MARS BARYCENTER", "EARTH"]


ToF = [1.5*np.pi, 1.5*np.pi, 2.2*np.pi]
minrad = [(c.RadiusEarth + 300.0*1000.0)/lstar, (c.RadiusMars + 300.0*1000.0)/lstar, (c.RadiusEarth + 300.0*1000.0)/lstar]


Tables = GenTable(bodies,data_start, data_end,lstar, tstar)


TOF1 = np.linspace(.7, 6., 40)
TOF2 = np.linspace(.7, 6., 40)
TotalDvs = []

for i in range(0, len(TOF1)):
    for j in range(0, len(TOF2)):
        ToF = [TOF1[i], TOF2[j]]
        Seq, DVs = GenFlyby(bodies, minrad, data_start, data_end, ToF, lstar, tstar, Tables)
        TotalDvs.append(np.sum(DVs))
print(TotalDvs)

T = np.array(Seq).T

plt.plot(T[0], T[1])
plt.plot(ED[0]*lstar/c.AU,ED[1]*lstar/c.AU, color ='g')
plt.plot(MD[0]*lstar/c.AU,MD[1]*lstar/c.AU, color ='r')
plt.show()
    
        
    
        
    
    