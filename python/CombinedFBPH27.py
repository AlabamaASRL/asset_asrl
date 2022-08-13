# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:30:57 2021

@author: Jared
"""


import pykep as pk
import pygmo as pg
import pygmo_plugins_nonfree as ppnf
from pykep.orbit_plots import plot_planet, plot_lambert
import numpy as np
from pykep.examples import add_gradient
from pykep import AU, DAY2SEC
from AstroModels import TwoBody

# Plotting imports
import asset as ast
import spiceypy as spice
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import MKgSecConstants as c
from datetime import datetime
import Date

pk.util.load_spice_kernel("Spice/de432s.bsp")
pk.util.load_spice_kernel("Spice/naif0012.tls")
pk.util.load_spice_kernel("Spice/gm_de431.tpc")
pk.util.load_spice_kernel("Spice/54186922.bsp")
ph27 = pk.planet.spice('54186922', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 1, 1, 1)

###Sequencing

Seq =["Earth", "Venus", "Venus", "Venus"]

#seq= [pk.planet.jpl_lp('earth'), pk.planet.jpl_lp('venus'), pk.planet.jpl_lp('earth'), pk.planet.jpl_lp('earth'),
#           pk.planet.jpl_lp('venus'), pk.planet.jpl_lp('venus'), pk.planet.jpl_lp('venus'), 
#           pk.planet.jpl_lp('venus'), pk.planet.jpl_lp('venus'), pk.planet.jpl_lp('venus'), ph27]
seq = []
seq.append(pk.planet.jpl_lp('earth'))
for name in Seq:
    seq.append(pk.planet.jpl_lp(name))
seq.append(ph27)
udp = pk.trajopt.mga(
     seq = seq,
     t0 = [8500, 10000], # This is in mjd2000
     tof = [[.8*365., 1.8*365.], [.8*365., 1.8*365.],[.8*365., 1.8*365.],[.8*365., 1.8*365.],
            [.8*365., 1.8*365.]], # This is in days
     vinf = 3.2, # This is in km/s
     orbit_insertion = False,
     multi_objective = False
)


prob = pg.problem(udp)

prob.c_tol = 1e-6

uda = pg.nlopt("slsqp")

uda2 = pg.mbh(uda, 5, 0.1)
algo = pg.algorithm(uda2)
algo.set_verbosity(1)

pop = pg.population(prob, 500)
# And optimize
pop = algo.evolve(pop)

ode = TwoBody(1.32712440018e20, 1.496e11)
tbinteg = ode.integrator(.001)

t = pop.champion_x[0]
lambert_vec = []
states = []
dts = np.array(pop.champion_x[1:])
statetime = 0
times = []
for i in range(1, len(pop.champion_x)):
    t += pop.champion_x[i]
    t2 = pk.epoch(t)
    t1 = pk.epoch(t - pop.champion_x[i])
    dt = (t2.mjd2000 - t1.mjd2000)*DAY2SEC
    planet1R, planet1V = seq[i-1].eph(t1)
    planet2R, planet2V = seq[i].eph(t2)
    lambert_arc = pk.lambert_problem(r1 = planet1R, r2 = planet2R, tof = dt, mu = pk.MU_SUN, max_revs = 0)
    lambert_vec.append(lambert_arc)
    r1 = np.asarray(planet1R)
    v1 = np.asarray(lambert_arc.get_v1()[0])
    planetv = np.asarray(planet1V)
    states.append(np.concatenate((r1/ode.lstar, v1/ode.vstar, [statetime])))
    statetime += dts[i-1]*86400.0/ode.tstar
    times.append(statetime)

    
'''
arc1 = tbinteg.integrate_dense(states[0], states[1][6], 500)
T1 = np.array(arc1).T

arc2 = tbinteg.integrate_dense(states[1], states[2][6], 500)
T2 = np.array(arc2).T

arc3 = tbinteg.integrate_dense(states[2], states[3][6], 500)
T3 = np.array(arc3).T

arc4 = tbinteg.integrate_dense(states[3], states[4][6], 500)
T4 = np.array(arc4).T

arc5 = tbinteg.integrate_dense(states[4], states[5][6], 500)
T5 = np.array(arc5).T

arc6 = tbinteg.integrate_dense(states[5], states[6][6], 500)
T6 = np.array(arc6).T

arc7 = tbinteg.integrate_dense(states[6], statetime, 500)
T7 = np.array(arc7).T
'''
fig = plt.figure(figsize = (13,9))
ax = fig.add_subplot(projection = "3d")
ax.set(xlim = (-1.0, 1.0), ylim=(-1.0, 1.0), zlim = (-1.0, 1.0))
vec = []
vec.append(float(pk.epoch(pop.champion_x[0]).jd))
for i in range(0, len(times)):
    arc = tbinteg.integrate_dense(states[i], times[i], 500)
    vec.append(arc)
    T1 = np.array(arc).T
    ax.plot(T1[0], T1[1], T1[2])


ax.legend()

plt.show()

#vec = [float(pk.epoch(pop.champion_x[0]).jd), arc1, arc2, arc3, arc4, arc5, arc6, arc7]


################################################################################
# Setup
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
TModes = oc.TranscriptionModes
spice.furnsh("BasicKernel.txt")

###############################################################################
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

def VinfMatchCon(tab):
    X = Args(8)
    v0 = X.head3()
    t0 = X[3]
    
    v1 = X.tail(4).head3()
    t1 = X.tail(4)[3]
    
    BodyV =  oc.InterpFunction(tab,range(3,6)).vf()
    vInfPlus=(BodyV.eval(t0)-v0).norm()
    vInfMinus=(BodyV.eval(t1)-v1).norm()
    return (vInfPlus-vInfMinus)
    
def RendCon(tab):
    XT = Args(7)
    x = XT.head(6)
    t = XT[6]
    fun = oc.InterpFunction(tab,range(0,6)).vf()
    return fun.eval(t) - x

def PosCon(tab):
    XT = Args(4)
    x = XT.head(3)
    t = XT[3]
    fun = oc.InterpFunction(tab,range(0,3)).vf()
    return fun.eval(t) - x
def VinfFunc(tab):
    XT = Args(4)
    x = XT.head(3)
    t = XT[3]
    fun = oc.InterpFunction(tab,range(3,6)).vf()
    return fun.eval(t) - x

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


def FlybyAngleBoundTest(tab,mubod,minrad):
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
    deltaMax   = np.pi - 2*vf.arccos(mubod/(mubod + minrad*vInf2)) 
    return   (deltaMax)

def DoEverythingInc(ODE, IGs, BodyData, StartTable, EndTable, DepVinf, NumSegs = 256):
    ocp = oc.OptimalControlProblem()
    for i,IG in enumerate(IGs):
        ocp.addPhase(ODE.phase(TModes.LGL3, IG, NumSegs))
        ocp.Phase(-1).addLUNormBound(PhaseRegs.Path,[7,8,9],.01,1.0)
        
        
    ocp.addForwardLinkEqualCon(0, len(IGs)-1, [0,1,2,6])
    ocp.Phase(0).addLowerVarBound(PhaseRegs.Front,6,0.0,1.0)
    ocp.Phase(0).addEqualCon(PhaseRegs.Front, PosCon(StartTable),[0,1,2,6]) # Start at Earth
    
    ocp.Phase(0).addUpperFuncBound(PhaseRegs.Front,VinfFunc(StartTable).squared_norm(),
                              [3,4,5,6],(DepVinf/vstar)**2,1.0)
    
    ocp.Phase(-1).addEqualCon(PhaseRegs.Back, PosCon(EndTable),[0,1,2,6])
    #ocp.Phase(-1).addStateObjective(PhaseRegs.Back, VinfFunc(EndTable).squared_norm()*10.0,[3,4,5,6])
    
    def Inc():
        X = Args(6)
        h = X.head3().cross(X.tail3()).normalized()
        return h[2]**2
    ocp.Phase(-1).addStateObjective(PhaseRegs.Back,Inc(),[0,1,2,3,4,5]) 
    
    ocp.Phase(0).addUpperVarBound(PhaseRegs.Back,6,ocp.Phase(-1).returnTraj()[-1][6]*1.05,1.0)

    for i,Data in enumerate(BodyData):
        Table = Data[0]
        MuValue = Data[2]
        BodyRad = Data[1]
        
        ocp.Phase(i).addEqualCon(PhaseRegs.Back, PosCon(Table),[0,1,2,6])
        ocp.addLinkEqualCon(VinfMatchCon(Table),oc.LinkFlags.BackToFront,[[i,i + 1]],[3,4,5,6])
        ocp.addLinkInequalCon(FlybyAngleBound(Table,MuValue,BodyRad),oc.LinkFlags.BackToFront,[[i,i+1]],[3,4,5,6])
        
    ocp.optimizer.OptLSMode =ast.Solvers.LineSearchModes.L1
    
    #ocp.transcribe(True,True)
    ocp.solve_optimize()
    
    for i,Data in enumerate(BodyData):
        T1 = ocp.Phase(i).returnTraj()
        T2 = ocp.Phase(i+1).returnTraj()
        Table = Data[0]
        MuValue = Data[2]
        BodyRad = Data[1]
        
        X = np.zeros((8))
        X[0:4]= T1[-1][3:7]
        X[4:8]= T2[0][3:7]
        F = FlybyAngleBoundTest(Table,MuValue,BodyRad)
        F2 = FlybyAngleBound(Table,MuValue,BodyRad)
        
        
        
        
    return [ocp.Phase(i).returnTraj() for i in range(len(IGs))]


def DoEverything(ODE, IGs, BodyData, StartTable, EndTable, DepVinf, NumSegs = 256):
    ocp = oc.OptimalControlProblem()
    for i,IG in enumerate(IGs):
        ocp.addPhase(ODE.phase(TModes.LGL3, IG, NumSegs))
        ocp.Phase(-1).addLUNormBound(PhaseRegs.Path,[7,8,9],.01,1.0)
        
        
    ocp.addForwardLinkEqualCon(0, len(IGs)-1, [0,1,2,6])
    ocp.Phase(0).addLowerVarBound(PhaseRegs.Front,6,0.0,1.0)
    ocp.Phase(0).addEqualCon(PhaseRegs.Front, PosCon(StartTable),[0,1,2,6]) # Start at Earth
    
    ocp.Phase(0).addUpperFuncBound(PhaseRegs.Front,VinfFunc(StartTable).squared_norm(),
                              [3,4,5,6],(DepVinf/vstar)**2,1.0)
    
    ocp.Phase(-1).addEqualCon(PhaseRegs.Back, PosCon(EndTable),[0,1,2,6])
    ocp.Phase(-1).addStateObjective(PhaseRegs.Back, VinfFunc(EndTable).squared_norm()*10.0,[3,4,5,6])
    
    def Inc():
        X = Args(6)
        h = X.head3().cross(X.tail3()).normalized()
        return h[2]**2
    #ocp.Phase(-1).addStateObjective(PhaseRegs.Back,Inc(),[0,1,2,3,4,5]) 
    
    #ocp.Phase(-1).addUpperVarBound(PhaseRegs.Back,6,ocp.Phase(-1).returnTraj()[-1][6]*1.05,1.0)

    for i,Data in enumerate(BodyData):
        Table = Data[0]
        MuValue = Data[2]
        BodyRad = Data[1]
        
        ocp.Phase(i).addEqualCon(PhaseRegs.Back, PosCon(Table),[0,1,2,6])
        ocp.addLinkEqualCon(VinfMatchCon(Table),oc.LinkFlags.BackToFront,[[i,i + 1]],[3,4,5,6])
        ocp.addLinkInequalCon(FlybyAngleBound(Table,MuValue,BodyRad),oc.LinkFlags.BackToFront,[[i,i+1]],[3,4,5,6])
        
    ocp.optimizer.OptLSMode =ast.Solvers.LineSearchModes.L1
    ocp.optimizer.QPThreads = 16
    ocp.optimizer.MaxIters = 1000
    
    #ocp.transcribe(True,True)
    ocp.solve_optimize()
    
    for i,Data in enumerate(BodyData):
        T1 = ocp.Phase(i).returnTraj()
        T2 = ocp.Phase(i+1).returnTraj()
        Table = Data[0]
        MuValue = Data[2]
        BodyRad = Data[1]
        
        X = np.zeros((8))
        X[0:4]= T1[-1][3:7]
        X[4:8]= T2[0][3:7]
        F = FlybyAngleBoundTest(Table,MuValue,BodyRad)
        F2 = FlybyAngleBound(Table,MuValue,BodyRad)
        
        
        
        
    return [ocp.Phase(i).returnTraj() for i in range(len(IGs))]
        
    

    

if __name__ == "__main__":
 
    lstar     = c.AU
    vstar     = np.sqrt(c.MuSun/lstar)
    tstar     = np.sqrt(lstar**3/c.MuSun)
    astar     = c.MuSun/lstar**2
      
    engacc = (1./1800)/astar
    
    print(engacc)
    #Seq =["Venus", "Earth","Venus", "Venus", "Venus", "Venus"]
    #LoadIG = np.load("IG.npy", allow_pickle = True)
    LoadIG = vec
    print(LoadIG[0])
    print(Date.jd_to_date(LoadIG[0]))
    date = Date.jd_to_date(LoadIG[0])
    startdayJD = datetime(int(date[0]), int(date[1]), int(date[2]))
    print(startdayJD)
    startdayMDY = startdayJD.strftime("%b %d, %Y")

    YEAR = 365*24*3600/tstar
    data_start    = startdayMDY
    print(data_start)
    data_end      = 'Jan 1, 2049'
    datapoints = 15000
    
    EarthDat  = GetEphemTraj("EARTH",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    MercuryDat = GetEphemTraj("Mercury",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    VenusDat   = GetEphemTraj("Venus",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    PH27Dat = GetEphemTraj("54186922",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    
    t0 = spice.utc2et(data_start)
    te = spice.utc2et(data_end)-t0
    
    EarthTab  = oc.LGLInterpTable(6,EarthDat,datapoints)
    MercuryTab = oc.LGLInterpTable(6,MercuryDat,datapoints)
    VenusTab   = oc.LGLInterpTable(6,VenusDat,datapoints)
    PH27Tab = oc.LGLInterpTable(6,PH27Dat,datapoints)
    
    ED =  np.array(EarthDat).T
    MD = np.array(MercuryDat).T
    VD =  np.array(VenusDat).T
    PD =  np.array(PH27Dat).T
    
    Trajs =[]
    for i, traj in enumerate(LoadIG[1:]):
        Traj=[]
        for j, state in enumerate(traj):
            X= np.zeros((10))
            X[0:7]=state
            X[6]+=2*YEAR
            n = state[3:6]/np.linalg.norm(state[3:6])
            X[7:10]= n*.3
            Traj.append(X)
        Trajs.append(Traj)
    
    Stuff ={}
    Stuff["Earth"] = [EarthTab,(c.RadiusEarth + 300.0*1000.0)/lstar, c.MuEarth*(tstar**2)/(lstar**3)]
    Stuff["Venus"] = [VenusTab,(c.RadiusVenus + 300.0*1000.0)/lstar, c.MuVenus*(tstar**2)/(lstar**3)]
    Stuff["Mercury"] = [MercuryTab,(c.RadiusMercury + 100.0*1000.0)/lstar, c.MuMercury*(tstar**2)/(lstar**3)]
    
    BodyData =[Stuff[S] for S in Seq]
    
    StartTable = EarthTab
    EndTable  = PH27Tab
    
    ode = LTModel(1,.1)
    #ode = TwoBody(1.32712440018e20, 1.496e11)
    
    Trajs2 = DoEverything(ode, Trajs, BodyData, StartTable, EndTable, 3200.0, NumSegs = 256)
    #Trajs3 = DoEverything(ode, Trajs2, BodyData, StartTable, EndTable, 5000.0, NumSegs = 256)
    
    times = []
    for i,traj in enumerate(Trajs2):
        times.append(traj[-1][6])
    
    fig = plt.figure(figsize = (9, 9))
    axes = fig.add_subplot(projection = "3d")
    axes.plot(ED[0]*lstar/c.AU,ED[1]*lstar/c.AU,ED[2]*lstar/c.AU,color ='g')
    axes.plot(VD[0]*lstar/c.AU,VD[1]*lstar/c.AU,VD[2]*lstar/c.AU,color = 'r')
    axes.plot(MD[0]*lstar/c.AU,MD[1]*lstar/c.AU,MD[2]*lstar/c.AU,color = 'blue')
    axes.plot(PD[0]*lstar/c.AU,PD[1]*lstar/c.AU,PD[2]*lstar/c.AU,color = 'k')
    T = []
    
    
    fig2 = plt.figure(figsize = (9, 9))
    axes2 = fig2.add_subplot()
    
    for traj in Trajs2:
        TT = np.array(traj).T
        T.append(TT)
        axes.plot(TT[0], TT[1], TT[2])
        axes2.plot(TT[6],(TT[7]**2 +TT[8]**2+TT[9]**2 )**.5)
        
    for i,seq in enumerate(Seq):
        state = Stuff[seq][0].Interpolate(times[i])
        axes.scatter(state[0], state[1], state[2])

    axes.scatter(0,0,0,color= 'y')
    axes.set_xlabel('X (AU)')
    axes.set_ylabel('Y (AU)')
    axes.set(xlim=(-1, 1), ylim = (-1, 1), zlim = (-1, 1))
    #axes.set_aspect('equal')
    #axes.grid(True)
    
    
    plt.show()

    breakpoint()
