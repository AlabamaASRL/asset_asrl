import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from inspect import getmembers, isfunction, isclass
import time as time

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

class FreeFlyingRobotODE(oc.ode_x_x.ode):
    def __init__(self,alpha,beta):
        
        Xvars = 6
        Uvars = 4
        Ivars = Xvars + 1 + Uvars
        ############################################################
        args = vf.Arguments(Ivars)
        theta = args[4]
        omega = args[5]
        u = args.tail(4)
        xdot = args.segment_2(2)
        vscale = vf.SumElems([u[0],u[1],u[2],u[3]],
                             [1,     -1,   1 ,-1])
        
        vdot = vf.StackScalar([vf.cos(theta),vf.sin(theta)])*vscale
        theta_dot=omega
       
        omega_dot= vf.SumElems([u[0],u[1],u[2],u[3]],
                               [alpha, -alpha, -beta ,beta])
        ode = vf.Stack([xdot,vdot
                        ,vf.StackScalar([theta_dot,omega_dot])])
        ##############################################################
        super().__init__(ode,Xvars,Uvars)

    class obj(ast.ScalarFunctional):
        def __init__(self):
            u = Args(4)
            obj = u[0] + u[1] + u[2] + u[3]
            super().__init__(obj)




ode = FreeFlyingRobotODE(.2,.2)

t0 = 0
tf = 12

X0 =[-10,-10,0,0,np.pi/2.0,0, 0]
XF =[0,0,0,0,0,0,tf]

IG = []
ts = np.linspace(0,tf,100)

for t in ts:
    T = np.zeros((11))
    T[0] = -10 + t*10/12.0
    T[1]=T[0]
    T[4] = np.pi/2.0 - t*(np.pi/2.0)/12.0
    T[6]=t
    T[7:11] = np.ones((4))*.5
    IG.append(T)

#ode.vf().SuperTest(IG[0],100000)
phase = ode.phase(Tmodes.LGL7)
phase.EnableVectorization=True
phase.setIntegralMode(oc.BaseIntegral)
phase.setTraj(IG,256)
phase.addBoundaryValue(PhaseRegs.Front,range(0,7),X0)
phase.addBoundaryValue(PhaseRegs.Back ,range(0,7),XF)
phase.addLUVarBounds(PhaseRegs.Path   ,range(7,11),0.00001,1.0,1)
phase.addIntegralObjective(FreeFlyingRobotODE.obj(),range(7,11))
phase.Threads=32
phase.optimizer.OptLSMode = ast.L1
phase.optimizer.MaxLSIters =1
phase.optimizer.PrintLevel =0

t0 = time.perf_counter()
phase.optimize()
tf = time.perf_counter()
print(tf-t0)
TrajConv = phase.returnTraj()

def plot(T):
    IGT = np.array(T).T
    
    plt.plot(IGT[0],IGT[1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid("True")
    plt.show()
    plt.plot(IGT[6],IGT[7]-IGT[8])
    plt.plot(IGT[6],IGT[9]-IGT[10])
    plt.show()


plot(TrajConv)



