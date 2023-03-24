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
    phase = ode.phase(Tmodes.LGL3, IG, Nsegs)
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

    phase.addBoundaryValue(PhaseRegs.Back, [0, 1], [theta, 0])

    phase.addDeltaTimeObjective(1.0)
    phase.optimizer.OptLSMode = solvs.LineSearchModes.L1
    phase.optimizer.MaxLSIters = 1
    phase.optimizer.QPOrderingMode = solvs.QPOrderingModes.MINDEG

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
        I[7] *= .8
        I[8:11] *= .6

    ode = QuatModel(Ivec)
    phase = ode.phase(Tmodes.LGL3, IG, Nsegs)

    phase.setControlMode(oc.ControlModes.BlockConstant)
    phase.addBoundaryValue(PhaseRegs.Front, range(0, 8),
                           [0, 0, 0, 1, 0, 0, 0, 0])
    phase.addBoundaryValue(PhaseRegs.Back, range(4, 7), [0, 0, 0])
    phase.addLUVarBounds(PhaseRegs.Path, [8, 9, 10], -Tmax, Tmax, 0.01)

    def axang(q=Args(4)):
        n = q.head3().normalized()
        thetan = 2.0*vf.arctan(q.head3().norm()/q[3])
        return thetan*n - nvec*theta

    phase.addEqualCon(PhaseRegs.Back, axang(), range(0, 4))
    phase.addDeltaTimeObjective(1.0)

    phase.optimizer.OptLSMode = solvs.LineSearchModes.L1
    phase.optimizer.MaxLSIters = 1
    phase.optimizer.MaxAccIters = 100
    phase.optimizer.QPOrderingMode = solvs.QPOrderingModes.MINDEG
    phase.optimizer.BoundFraction = .997
    phase.optimizer.deltaH = 1.0e-6
    phase.optimizer.KKTtol = 1.0e-7
    phase.optimizer.EContol = 1.0e-8
    phase.optimizer.QPParSolve = 1
    #phase.AdaptiveMesh=True
    #phase.MeshErrorEstimator="integrator"
    phase.ForceOneMeshIter = True
    #phase.DetectControlSwitches=True
    phase.RelSwitchTol = .2
    phase.optimizer.PrintLevel=1
    phase.MeshTol=1.0e-7
    phase.MeshErrFactor=10
    
    phase.setJetJobMode("optimize")
    
    
    return phase



#####################################################

if __name__ == "__main__":

    Ivec = np.array([1,2,3])
    nvec = normalize([1,1,100])
    theta = 2.1
    Tmax = 1
    Nsegs = 30
    
    

    T1 = EigAxis(Ivec, Tmax, nvec, theta, Nsegs)
    
    phases = [FullAxis(Ivec, Tmax, nvec, theta, T1, Nsegs) for i in range(0,100)]
    
    phases = ast.Solvers.Jet.map(phases,16)
    
    
    T2 = phases[0].returnTraj()
    #T2 = FullAxis(Ivec, Tmax, nvec, theta, T1, Nsegs)
    

    I = 1
    J = 2
    
    IM = [1,2,3]
    
    ode = QuatModel(IM)
    integ = ode.integrator(.1)
    
    X0 = np.zeros((11))
    X0[0]=0
    X0[3]=1
    X0[4] =.5
    X0[5] =.5
    X0[6] =.5
    X0[5] = 1
    X0[8]=1
    
    dt = 10
    n = 2000
    
    SpinUp = integ.integrate_dense(X0,dt,n)
    X1 = SpinUp[-1]
    X1[8]=0
    Coast  = integ.integrate_dense(X1,X1[7]+dt,n)
    
    
    
    AnimSlew(T2,Elev=30,Azim =15,time=15,save=False)










       