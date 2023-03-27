import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import time

norm = np.linalg.norm
vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
solvs     = ast.Solvers


class Model(oc.ODEBase):
    def __init__(self,a,w3):
        Xvars = 7
        Uvars = 3
        Ivars = Xvars + 1 + Uvars
        ############################################################
        args = oc.ODEArguments(4,2)
        
        w1,w2 = args.head(2).tolist()
        x1,x2 = args.segment(2,2).tolist()
        u1,u2 = args.tail(2).tolist()
        
        w1dot = a*w2*w3 + u1
        w2dot =-a*w3*w1 + u2
        
        x1dot =  w3*x2 + w2*x1*x2 + w1*(1+x1**2 -x2**2)/2
        x2dot = -w3*x1 + w1*x1*x2 + w2*(1+x2**2 -x1**2)/2
        
        ode= vf.stack(w1dot,w2dot,x1dot,x2dot)
        
        ##############################################################
        super().__init__(ode,4,2)
        
        


a = .5
tfig = 4

ode = Model(a,0)

X0 = np.zeros(7)
X0[0]=-.45
X0[1]=-1.1
X0[2]=.1
X0[3]=-.1
X0[4]=0
X0[5]=.0

XF = np.zeros((7))
XF[4]=tfig

ts = np.linspace(0,1,100)

IG = []
for t in ts:
    XI = X0*(1-t) + XF*t
    IG.append(XI)
    
    
phase = ode.phase("LGL5",IG,100)
phase.setControlMode("BlockConstant")
phase.addBoundaryValue("Front",range(0,5),X0[0:5])
phase.addBoundaryValue("Back",range(0,4),XF[0:4])
phase.addLUVarBounds("Path",[5,6],-1,1,1)

phase.addDeltaTimeObjective(1)
phase.setAdaptiveMesh(True)
phase.optimizer.PrintLevel = 2
phase.DetectControlSwitches = True
phase.ForceOneMeshIter = True
phase.MeshTol=1.0e-7
#phase.optimizer.set_OptLSMode("L1")
phase.Threads = 8
phase.optimizer.QPThreads = 8
phase.optimizer.KKTtol=1.0e-10
phase.optimizer.deltaH=1.0e-9
phase.optimizer.QPOrderingMode = solvs.QPOrderingModes.MINDEG
phase.optimizer.BoundFraction = .997
phase.AbsSwitchTol=.1
phase.setThreads(1,1)
phase.optimizer.CNRMode = True

phase.solve_optimize()

Traj = phase.returnTraj()

TT = np.array(Traj).T

TT[4]=TT[4]/TT[4][-1]

plt.plot(TT[4],TT[5])
plt.plot(TT[4],TT[6])

plt.show()
