import numpy as np
import asset_asrl as ast
from QuatPlot import AnimSlew,PlotSlew,CompSlew,PlotSlew2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from SpherePlot import octant_points,octant_points2,OctPlot1,OctSubPLot
import time
import multiprocessing as mp
import os
import matplotlib.tri as tri
import random

norm = np.linalg.norm
vf    = ast.VectorFunctions
oc    = ast.OptimalControl
Args  = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
solvs = ast.Solvers

###############################################################################
Ivec_csat     = np.array([1,2,2.6]) 
    
Ivec_neascout = np.array([15.975,16.525,32.217])   
Ivec_neascout = Ivec_neascout/Ivec_neascout[0]

Ivec_hubble   = np.array([36046,86868, 93848])   
Ivec_hubble   = Ivec_hubble/Ivec_hubble[0]

Ivec_cassini   = np.array([4721,8157.3, 8810.8])   
Ivec_cassini   = Ivec_cassini/Ivec_cassini[0]

Ivec_iss   = np.array([10212044,31943467, 40080923])   
Ivec_iss   = Ivec_iss/Ivec_iss[0]

###############################################################################
def normalize(x):return x/np.linalg.norm(x)
def WriteData(traj,name,Folder = 'Data'):
    if(Folder != ''):
        if not os.path.exists(Folder+'/'):
            os.makedirs(Folder+'/')
    np.save(Folder+'/' + name+ '.npy',traj)
def ReadData(name,Folder = 'Data'):
   return np.load(Folder+'/' + name+ '.npy',allow_pickle=True)

###############################################################################
def normalize(x): return np.array(x)/np.linalg.norm(x)


class QuatModel(oc.ODEBase):
    def __init__(self,Ivec):
        Xvars = 7
        Uvars = 3
        #################################################
        XtU = oc.ODEArguments(Xvars,Uvars)
        
        q = XtU.XVec().head(4)
        w = XtU.XVec().tail(3)
        T = XtU.UVec()
        
        qdot = vf.quatProduct(q,w.padded_lower(1))/2.0
        L = w.cwiseProduct(Ivec)
        wdot = (L.cross(w) + T).cwiseQuotient(Ivec)
       
        ode = vf.stack([qdot,wdot])
        #################################################
        super().__init__(ode,Xvars,Uvars)


class EigModel(oc.ode_x_u.ode):
    def __init__(self):
        args = oc.ODEArguments(2, 1)
        ode = vf.stack([args[1], args[3]])
        super().__init__(ode, 2, 1)


def EigAxis(Ivec, Tmax, nvec, theta, Nsegs):

    alpha = (abs(theta)/theta)*Tmax/norm(nvec*Ivec)
    h = np.sqrt(theta/alpha)
    ts = np.linspace(0, 2*h, 50)
    IG = [np.zeros((4)) for t in ts]
    for i,t in enumerate(ts):
        u = alpha
        tdot = alpha*t
        thet = alpha*t*t/2
        if(t > h):
            tt = t-h
            u = -alpha
            tdot = alpha*(h-tt)
            thet = alpha*h*h/2 - alpha*tt*tt/2 + alpha*h*tt
        IG[i]=np.array([thet, tdot, t, u*.7])

    ode = EigModel()
    phase = ode.phase(Tmodes.Trapezoidal, IG, Nsegs)
    phase.setControlMode(oc.ControlModes.BlockConstant)
    phase.addBoundaryValue(PhaseRegs.Front, [0, 1, 2], [0, 0, 0])

    C1 = Ivec*nvec
    C2 = np.cross(nvec, C1)

    def SBoundFunc(c1, c2):
        args = Args(2)
        tdot = args[0]
        u = args[1]
        return u*c1 + (tdot**2)*c2

    for i in range(0, 3):
        F = SBoundFunc(C1[i], C2[i])
        phase.addUpperFuncBound(PhaseRegs.Path, F, [1, 3], Tmax, 1.0)
        phase.addLowerFuncBound(PhaseRegs.Path, F, [1, 3], -Tmax, 1.0)
        
        #phase.addLUFuncBound("Path",F,[1,3],-Tmax,Tmax,1.0)

    phase.addBoundaryValue(PhaseRegs.Back, [0, 1], [theta, 0])

    phase.addDeltaTimeObjective(1.0)
    phase.optimizer.OptLSMode = solvs.LineSearchModes.L1
    phase.optimizer.MaxLSIters = 1
    phase.optimizer.QPOrderingMode = solvs.QPOrderingModes.MINDEG
    
    #phase.setThreads(8,8)
    phase.setJetJobMode("optimize")

    phase.optimize()
    
    
    return phase.returnTraj()

def GetEigTraj(TrajF, Ivec, nvec):
    C1 = Ivec*nvec
    C2 = np.cross(nvec, C1)

    TrajR = []
    for T in TrajF:
        X = np.zeros((11))
        X[0:3] = nvec*np.sin(T[0]/2)
        X[3] = np.cos(T[0]/2)
        X[4:7] = nvec*T[1]
        X[7] = T[2]
        X[8:11] = T[3]*C1 + T[1]*T[1]*C2
        TrajR.append(X)

    return TrajR
from asset_asrl.OptimalControl.MeshErrorPlots import PhaseMeshErrorPlot

def FullAxis(Ivec, Tmax, nvec, theta, EigAxTraj, Nsegs):

    IG = GetEigTraj(EigAxTraj, Ivec, nvec)
    for I in IG:
        I[7] *= .99
        I[8:11] *= .6

    ode = QuatModel(Ivec)
    phase = ode.phase(Tmodes.LGL5, IG, Nsegs)

    #phase.setControlMode(oc.ControlModes.BlockConstant)
    phase.setControlMode(oc.ControlModes.NoSpline)

    phase.addBoundaryValue(PhaseRegs.Front, range(0, 8),
                           [0, 0, 0, 1, 0, 0, 0, 0])
    phase.addBoundaryValue(PhaseRegs.Back, range(4, 7), [0, 0, 0])
    phase.addLUVarBounds(PhaseRegs.Path, [8, 9, 10], -Tmax, Tmax, .01)

    def axang(q=Args(4)):
        n = q.head3().normalized()
        thetan = 2.0*vf.arctan(q.head3().norm()/q[3])
        return thetan*n - nvec*theta

    phase.addEqualCon(PhaseRegs.Back, axang(), range(0, 4))
    phase.addDeltaTimeObjective(1)
    
    f = Args(2)[0]-Args(2)[1] +.01
    #phase.addInequalCon("PairWisePath",f*.01,[7])
    

    phase.optimizer.OptLSMode = solvs.LineSearchModes.L1
    phase.optimizer.MaxLSIters = 1
    phase.optimizer.MaxAccIters = 100
    #phase.optimizer.QPOrderingMode = solvs.QPOrderingModes.MINDEG
    #phase.optimizer.set_OptBarMode("PROBE")
    phase.optimizer.BoundFraction = .997
    #phase.addIntegralObjective(-1*Args(3).squared_norm(),[8,9,10])
    phase.optimizer.deltaH = 1.0e-6
    phase.optimizer.KKTtol = 1.0e-10
    phase.optimizer.EContol = 1.0e-8
    phase.optimizer.QPParSolve = 1
    #phase.optimizer.BoundPush = .00001
    phase.AdaptiveMesh=True
    #phase.MeshErrorEstimator="integrator"
    phase.MinMeshIters = 1
    phase.DetectControlSwitches=True
    phase.RelSwitchTol = .1
    phase.AbsSwitchTol = .07
    phase.optimizer.PrintLevel=1
    phase.MeshTol=1.0e-7
    #phase.MeshRedFactor =1.1
    phase.MeshErrFactor=10
    phase.MaxMeshIters = 9
    #phase.setThreads(1,1)

    phase.sfactor1 = .1
    #phase.sfactor2 = .01

    phase.setJetJobMode("optimize")
    
    
    return phase



#####################################################


def Derp():
    
    IM = [1,2,3]
    
    ode = QuatModel(IM)
    integ = ode.integrator(.01)
    
    X0 = np.zeros((11))
    X0[0]=0
    X0[3]=1
    X0[4] =.5
    X0[5] =.5
    X0[6] =.5
    X0[5] = 1
    X0[8]=0.1
    
    dt = 100
    n  = 2000
    
    SpinUp = integ.integrate_dense(X0,dt,n)
    
    AnimSlew(SpinUp,Elev=30,Azim =15,time=15,save=False,Anim=False)

    nn=1000
    
    X0s = [X0]*nn
    dts = [dt]*nn
    
    t0 = time.perf_counter()
    X = integ.integrate_v(X0s,dts,False)
    tf = time.perf_counter()
    print(1000*(tf-t0))
    
    t0 = time.perf_counter()
    X = integ.integrate_v(X0s,dts,True)
    tf = time.perf_counter()
    print(1000*(tf-t0))
    
    input("S")
    
    

if __name__ == "__main__":
    Derp()  
    Ivec = np.array([1,2,2.6])
    nvec = normalize([1,1,1])
    theta = 2.1
    Tmax = 1
    Nsegs = 100
    
    

    T1 = EigAxis(Ivec, Tmax, nvec, theta, Nsegs)
    
    
    
    #phases = [FullAxis(Ivec, Tmax, nvec, theta, T1, Nsegs) for i in range(0,100)]
    
    phase = FullAxis(Ivec, Tmax, nvec, theta, T1, Nsegs)
    
    

    
    phase.optimize()

    PhaseMeshErrorPlot(phase,show=True)

    #phase.optimize()
    #phase.optimize()
    ss = phase.getSwitchStatesTmp()

    ode = QuatModel(Ivec)
    integ = ode.integrator(.1,phase.returnTrajTable())
    
    T2 = phase.returnTraj()
    T = np.array(T2).T
    T[7] = T[7]/T[7][-1]

    for s in ss:
        
        plt.plot([T[7][s]]*2,[-1,1],color='k',linestyle='dashed')
    #T2 = integ.integrate_dense(T2[0],T2[-1][7])
    
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("T")
    

    
    T = np.array(T2).T
    
    print(T[7][-1])


    T[7] = T[7]/T[7][-1]
    

   

    

    #for i in range(0,len(T[7])-1):
        
        #plt.plot(T[7][i:i+2],[T[8][i]]*2,marker='o',color='r')
        #plt.plot(T[7][i:i+2],[T[9][i]]*2,marker='o',color='g')
        #plt.plot(T[7][i:i+2],[T[10][i]]*2,marker='o',color='b')
        
    plt.plot(T[7],T[8],marker='o',color='r',label='T1')
    plt.plot(T[7],T[9],marker='o',color='g',label='T1')
    plt.plot(T[7],T[10],marker='o',color='b',label='T1')
    plt.legend()
    
    plt.show()
    

    
    #T2 = FullAxis(Ivec, Tmax, nvec, theta, T1, Nsegs)
    

   
    integ = ode.integrator(.1,phase.returnTrajTable())

    T2 = phase.returnTraj()

    T2 = integ.integrate_dense(T2[0],T2[-1][7],1000)
    
    AnimSlew(T2,Elev=30,Azim =15,time=15,save=False,Anim=False)










       