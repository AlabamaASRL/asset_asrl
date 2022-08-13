import numpy as np
import asset as ast
from QuatPlot import AnimSlew
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import IARM

norm = np.linalg.norm
vf    = ast.VectorFunctions
oc    = ast.OptimalControl
Args  = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags


class QuatModel(oc.ode_x_x.ode):
    def __init__(self,I):
        Xvars = 7
        Uvars = 3
        Ivars = Xvars + 1 + Uvars
        ############################################################
        args = vf.Arguments(Ivars)
        
        qvec = args.head3()
        q4   = args[3]
        w    = args.segment3(4)
        T    = args.tail3()
    
        qvdot = (w*q4 + vf.cross(qvec,w))*0.5
        q4dot = -0.5*(vf.dot(w,qvec))
        wd1   = T[0]/I[0] + ((I[1]-I[2])/(I[0]))*(w[1].dot(w[2]))
        wd2   = T[1]/I[1] + ((I[2]-I[0])/(I[1]))*(w[0].dot(w[2]))
        wd3   = T[2]/I[2] + ((I[0]-I[1])/(I[2]))*(w[0].dot(w[1]))
        ode = vf.Stack([qvdot,q4dot,wd1,wd2,wd3])
        ##############################################################
        super().__init__(ode,Xvars,Uvars)

class EigModel(oc.ode_x_x.ode):
    def __init__(self):
        args = oc.ODEArguments(2,1)
        ode = vf.Stack([args[1],args[3]])
        super().__init__(ode,2,1)

def EigAxis(Ivec,Tmax,nvec,theta, Scalar =True):
    
    
    h = 1.0*np.sqrt(theta*np.linalg.norm(nvec*Ivec)/(Tmax))

    C1 = Ivec*nvec
    C2 = np.cross(nvec,C1)
    
    def SBoundFunc(c1,c2):
        args = Args(2)
        tdot = args[0]
        u = args[1]
        return u*c1 + (tdot**2)*c2
    
    ts =  np.linspace(0,2*h,100)
    IG =[]
   
        
    IG = [[theta*t/(2*h),theta/(2*h),t,.0] for t in np.linspace(0,2*h,100)]
    
    ode = EigModel()
    phase = ode.phase(Tmodes.LGL3,IG,400)
    phase.addBoundaryValue(PhaseRegs.Front,[0,1,2],[0,0,0])
    phase.addBoundaryValue(PhaseRegs.Back ,[0,1],[theta,0])
    Fn =[]
    for i in range(0,3):
        F = SBoundFunc(C1[i],C2[i])
        Fn.append(F)
        if(Scalar==True):
            phase.addUpperFuncBound(PhaseRegs.Path,F,[1,3], Tmax,1.0)
            phase.addLowerFuncBound(PhaseRegs.Path,F,[1,3],-Tmax,1.0)
        else:
            if(i==2):
               Fc = vf.Stack(Fn).norm()
               phase.addUpperFuncBound(PhaseRegs.Path,Fc,[1,3], Tmax,1.0)

    phase.addDeltaTimeObjective(1.0)
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


        
    


## Principle Axis Inertia Matrix
Ivec = np.array([1,2,2.6])
Tmax = 1.00  ## Max Torque


ode = QuatModel(Ivec)

## initial quat, id leave it as is
Q0 = [0,0,0,1]


## Specify final orientation w/ axis angle
angle = 179.9
nvec = np.array([1.0,1,1])
nvec = nvec/np.linalg.norm(nvec)
theta=np.deg2rad(angle)
r = R.from_rotvec(theta * nvec)
QF = r.as_quat()

tsw = 1.0*np.sqrt(theta*np.linalg.norm(nvec*Ivec)/(Tmax))

#K = EigAxis(Ivec,Tmax,nvec,theta)

IG = np.zeros((11))
IG[0:4] = Q0
IG[10]=.0001


def Con(x):
    t=x[0]
    if(t<tsw):return nvec*Tmax*.8
    else:return -nvec*Tmax*.8
    
cont = vf.PyVectorFunction(1,3, Con)
dtinteg = oc.ode_x_x.integrator(ode,.05,cont,[7])

Traj = dtinteg.integrate_dense(IG,2*tsw,2000)
import time

t0 = time.perf_counter()
Traj =IARM.IARM_bvp(Ivec,Tmax*1.0,600,QF,Nsteps=15,verbose=True,fdstep=1.0e-7,tol=1.0e-8,MaxIters=160)
tf = time.perf_counter()

K = EigAxis(Ivec,Tmax,nvec,theta)

print((tf-t0)*1000.0)
print(Traj[-1][7])

input("s")
#AnimSlew(Traj[1:-1],Anim=True,Elev=45,Azim=315,Ivec=Ivec)
AnimSlew(Traj[1:-1],Anim=True,Elev=45,Azim=315,Ivec=Ivec)
AnimSlew(K[1:-1],Anim=False,Elev=45,Azim=315,Ivec=Ivec)




## Try other transcriptions/more points if its not solving

TrajIG = np.copy(Traj)
for T in TrajIG: T[8:11]*=.98
phase= ode.phase(Tmodes.LGL3,TrajIG,300)
## Block constant seems to work best on this problem
phase.setControlMode(oc.ControlModes.BlockConstant)
phase.addBoundaryValue(PhaseRegs.Front,range(0,8),IG[0:8])
phase.addBoundaryValue(PhaseRegs.Back,range(4,7),[0,0,0])
Tmax =1  ## Max Torque

phase.addLUVarBounds(PhaseRegs.Path,[8,9,10],-Tmax,Tmax,1.0)

#phase.addUpperFuncBound(PhaseRegs.Path,Args(3).norm(),[8,9,10],Tmax,1.0)
phase.addEqualCon(PhaseRegs.Back,(Args(4).normalized() - QF)/1000.0,range(0,4))
phase.addDeltaTimeObjective(1.0)
#phase.addIntegralObjective(0.5*Args(3).squared_norm(),[8,9,10])
#phase.addUpperVarBound(PhaseRegs.Back,7,4*tsw,1.0)



#phase.optimizer.OptLSMode = ast.LineSearchModes.L1
phase.optimizer.deltaH=1.0e-6
phase.optimizer.KKTtol=1.0e-7
phase.optimize()

Traj  = phase.returnTraj()

AnimSlew(Traj[1:-1],Anim=True,Elev=45,Azim=315,Ivec=Ivec,sp=3)
AnimSlew(Traj[1:-1],Anim=False,Elev=45,Azim=315,Ivec=Ivec)

