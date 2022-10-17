import numpy as np
import asset as ast
from QuatPlot import AnimSlew,PlotSlew,CompSlew
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from DerivChecker import FDDerivChecker

norm      = np.linalg.norm
vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
solvs     = ast.Solvers

ast.PyMain()
  


class QuatModel(oc.ode_x_u.ode):
    def __init__(self,Ivec):
        Xvars = 7
        Uvars = 3
        ############################################################
        args = oc.ODEArguments(7,3)
        
        q    = args.head(4)#.normalized()
        w    = args.segment3(4)
        T    = args.tail3()
        
        qdot  = vf.quatProduct(q,w.padded_lower(1))/2.0
        L     = w.cwiseProduct(Ivec)
        wdot  = (L.cross(w) + T).cwiseQuotient(Ivec)
        ode = vf.stack(qdot,wdot )

        ##############################################################
        super().__init__(ode,Xvars,Uvars)
        

class EigModel(oc.ode_x_u.ode):
    def __init__(self):
        args = oc.ODEArguments(2,1)
        ode = vf.Stack([args[1],args[3]])
        super().__init__(ode,2,1)

def EigAxis(Ivec,Tmax,nvec,theta,Nsegs =200):
    
    alpha=Tmax/norm(nvec*Ivec)
    h = np.sqrt(theta/alpha)
    
    ts =  np.linspace(0,2*h,50)
    IG = []
    for t in ts:
        u =alpha
        tdot = alpha*t
        thet = alpha*t*t/2
        if(t>h):
            tt=t-h
            u =-alpha
            tdot = alpha*(h-tt)
            thet = alpha*h*h/2 - alpha*tt*tt/2 + alpha*h*tt
        IG.append([thet,tdot,t,u*.7])
        
    ode = EigModel()
    phase = ode.phase(Tmodes.LGL3,IG,Nsegs)
    phase.setControlMode(oc.ControlModes.BlockConstant)
    phase.addBoundaryValue(PhaseRegs.Front,[0,1,2],[0,0,0])
    
    C1 = Ivec*nvec
    C2 = np.cross(nvec,C1)
    
    def SBoundFunc(c1,c2):
        args = Args(2)
        tdot = args[0]
        u = args[1]
        return u*c1 + (tdot**2)*c2
    
    for i in range(0,3):
        F = SBoundFunc(C1[i],C2[i])
        phase.addUpperFuncBound(PhaseRegs.Path,F**2,[1,3], Tmax**2,1.0)
       
    phase.addBoundaryValue(PhaseRegs.Back ,[0,1],[theta,0])

    phase.addDeltaTimeObjective(1.0)
    phase.optimizer.OptLSMode = solvs.LineSearchModes.L1
    phase.optimizer.MaxLSIters = 1
    phase.optimizer.PrintLevel=1
    phase.optimizer.QPThreads =8
    phase.Threads=8
   
    phase.solve_optimize()
    TrajF = phase.returnTraj()
    
    TrajR =[]
    for T in TrajF:
        X = np.zeros((11))
        X[0:3] = nvec*np.sin(T[0]/2)
        X[3] = np.cos(T[0]/2)
        X[4:7] = nvec*T[1]
        X[7]=T[2]
        X[8:11] = T[3]*C1 + T[1]*T[1]*C2
        TrajR.append(X)
        
    return TrajR
    
def FullAxis(Ivec,Tmax,nvec,theta, IG,Nsegs=200):
    r = R.from_rotvec(theta * nvec)
    ode = QuatModel(Ivec)
    TrajIG = np.copy(IG)
    for T in TrajIG:
        T[8:11]*=.4
        
    phase= ode.phase(Tmodes.LGL3,TrajIG,Nsegs)
    #phase.setControlMode(oc.ControlModes.BlockConstant)
    phase.addBoundaryValue(PhaseRegs.Front,range(0,8),[0,0,0,1,0,0,0,0])
    phase.addBoundaryValue(PhaseRegs.Back,range(4,7),[0,0,0])
    phase.addLUVarBounds(PhaseRegs.Path,[8,9,10],-Tmax,Tmax,.1)
    
    
    def axang():
        q = Args(4)
        n = q.head3().normalized()
        thetan = 2.0*vf.arctan(q.head3().norm()/q[3])
        return thetan*n - nvec*theta
    
    phase.addEqualCon(PhaseRegs.Back,axang(),range(0,4))
    
    phase.addDeltaTimeObjective(1.0)
    phase.optimizer.OptLSMode = solvs.LineSearchModes.L1
    phase.optimizer.MaxLSIters =1
    phase.optimizer.QPThreads =8
    phase.Threads=8
    
    phase.optimizer.PrintLevel=1
    phase.optimizer.BoundFraction=.999
    phase.optimizer.deltaH=1.0e-5
    phase.optimizer.KKTtol=1.0e-6
    phase.optimizer.MaxAccIters=200

    phase.optimizer.set_OptLSMode("L1")
    phase.solve_optimize()
    
    TrajF = phase.returnTraj()
    
    return TrajF


def CalcManeuver(Ivec,nvec,thetadeg,Tmax=1):
    nvec = nvec/norm(nvec)
    theta = np.deg2rad(thetadeg)
    
    EAM = EigAxis(Ivec,Tmax,nvec,theta,256)
    
    qf = R.from_rotvec(theta * nvec).as_quat()
    
    TrajIG = []
    
    ts = np.linspace(0,1,100)
    
    for t in ts:
        X = np.zeros((11))
        X[0:4] = np.array([0,0,0,1])*(1-t) + qf*t
        print(X[0:4])
        #X[4]=.01
        X[7] = t*5
        #X[8]=.01
        TrajIG.append(X)
    
    
    
    OPT = FullAxis(Ivec,Tmax,nvec,theta,EAM,200)
    
    AnimSlew(EAM[1:-1],Anim=False,Ivec=Ivec)
    AnimSlew(OPT[1:-1],Anim=False,Ivec=Ivec)
    
if __name__ == "__main__":
    
    Ivec = np.array([1,2.0,2.6])    ## Inertia of a 6U cubesat
    Ivec = np.array([5621,4547,2364])/2364

    CalcManeuver(Ivec,[1,.07,.07],150.0)

    

