import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

vf = ast.VectorFunctions
oc = ast.OptimalControl
sol = ast.Solvers
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

class FreeFlyingRobotODE(oc.ode_x_u.ode):
    def __init__(self,alpha,beta):
        
        Xvars = 6
        Uvars = 4
        Ivars = Xvars + 1 + Uvars
        ############################################################
        args = oc.ODEArguments(6,4)
        theta = args[4]
        omega = args[5]
        u     =  args.UVec()
        xdot = args.segment2(2)
        vscale = vf.SumElems([u[0],u[1],u[2],u[3]],
                             [1,     -1,   1 ,-1])
        
        vdot = vf.Stack([vf.cos(theta),vf.sin(theta)])*vscale
        
        theta_dot=omega
       
        omega_dot= vf.SumElems([u[0],u[1],u[2],u[3]],
                               [alpha, -alpha, -beta ,beta])
        ode = vf.Stack([xdot,vdot,vf.StackScalar([theta_dot,omega_dot])])
        ##############################################################
        super().__init__(ode,Xvars,Uvars)

    class obj(vf.ScalarFunction):
        def __init__(self):
            u = Args(4)
            obj = u[0] + u[1] + u[2] + u[3]
            super().__init__(obj)




ode = FreeFlyingRobotODE(.2,.2)

t0 = 0
tf = 12

X0 =np.array([-10,-10,0,0,np.pi/2.0,0, 0])
XF =np.array([0,0,0,0,0,0,tf])

IG = []
ts = np.linspace(0,tf,100)

for t in ts:
    T = np.zeros((11))
    T[0:7] = X0[0:7] + ((t-t0)/(tf-t0))*( XF[0:7]- X0[0:7])
    T[7:11] = np.ones((4))*.50
    IG.append(T)

ode.vf().SuperTest(IG[0],1000000)
input("s")

phase = ode.phase(Tmodes.LGL3,IG,256)
phase.addBoundaryValue(PhaseRegs.Front,range(0,7),X0)
phase.addBoundaryValue(PhaseRegs.Back ,range(0,7),XF)
phase.addLUVarBounds(PhaseRegs.Path   ,range(7,11),0.00000,1.0,1)
phase.addIntegralObjective(FreeFlyingRobotODE.obj(),range(7,11))
phase.optimizer.BoundFraction = .995
phase.optimizer.PrintLevel=0
phase.optimizer.OptLSMode = sol.LineSearchModes.L1
phase.optimizer.MaxLSIters =1
phase.optimizer.KKTtol =1.0e-6
phase.optimizer.QPThreads =8
phase.Threads=8
phase.optimize()
#phase.refineTrajManual(150)
#phase.optimize()

TrajConv = phase.returnTraj()

##########################################################
IGT = np.array(TrajConv).T

plt.plot(IGT[0],IGT[1])
plt.xlabel("X")
plt.ylabel("Y")
plt.grid("True")
plt.show()
plt.plot(IGT[6],IGT[7]-IGT[8])
plt.plot(IGT[6],IGT[9]-IGT[10])
plt.show()
##########################################################




