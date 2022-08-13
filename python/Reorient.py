import numpy as np
import asset as ast
from QuatPlot import AnimSlew
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import scipy as scipy

print("s")
vf    = ast.VectorFunctions
oc    = ast.OptimalControl
Args  = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
sol = ast.Solvers


'''
In this example we will demonstrate defining an ode model and numerically integrating
The model in question will be quaternion based attitude dynamics with controllable
torque values, we will then write a

'''

class QuatModel(oc.ode_x_u.ode):
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

Ivec = np.array([150,190.5,250])
h   =320
ode = QuatModel(Ivec)



integ = ode.integrator(.01)
IG = np.zeros((11))


IG[3]=1.0
IG[4]= .0
IG[5]= 0
IG[6]= 0
IG[8]=0
IG[10]=.0001
Tmax = 1
tsw  = 14.0
nvec = np.array([.5,.5,.5])
nvec = nvec/np.linalg.norm(nvec)

ode.vf().SuperTest(IG,1000000)
input("s")
print(nvec*Ivec)
def Con(x):
    t=x[0]
    if(t<tsw):
        return nvec*Tmax*.9
    else:
        return -nvec*Tmax*.9
    
cont = vf.PyVectorFunction(1,3, Con)
dtinteg = oc.ode_x_u.integrator(ode,.001,cont,[7])

Traj = dtinteg.integrate_dense(IG,2*tsw,2000)
AnimSlew(Traj,Anim=True,Elev=45,Azim=315,Ivec=Ivec)

phase= ode.phase(Tmodes.LGL3,Traj,500)
phase.addBoundaryValue(PhaseRegs.Front,range(0,8),[0,0,0,1,0,0,0,0])
phase.addBoundaryValue(PhaseRegs.Back,range(4,7),[0,0,0])
#phase.setControlMode(oc.ControlModes.BlockConstant)
angle = 100
nvec = np.array([1,1,1])
nvec = nvec/np.linalg.norm(nvec)
r = R.from_rotvec(np.deg2rad(angle) * nvec)
q = r.as_quat()
    
phase.addEqualCon(PhaseRegs.Back,(Args(4).normalized() - q),range(0,4))
phase.addUpperFuncBound(PhaseRegs.Path,Args(3).norm(),[8,9,10],Tmax,1.0)
#phase.addLUVarBounds(PhaseRegs.Path,[8,9,10],-Tmax,Tmax,1.0)
phase.addDeltaTimeObjective(1.0)
#phase.addIntegralObjective(0.5*Args(3).vf().squared_norm(),[7,8,9])

phase.optimizer.deltaH=1.0e-6
phase.optimizer.KKTtol=1.0e-6

phase.optimizer.QPThreads=6

#phase.addUpperVarBound(PhaseRegs.Back,7,h)
#phase.addBoundaryValue(PhaseRegs.Back,[7],[40])
#phase.addDeltaTimeObjective(1.0)

phase.solve_optimize()

V = 0.0637494


Traj  = phase.returnTraj()


AnimSlew(Traj[1:-1],Anim=False,Elev=45,Azim=315,Ivec=Ivec)






