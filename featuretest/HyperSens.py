import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes = oc.ControlModes

'''
Hyper-Sensitive Problem
https://openmdao.github.io/dymos/examples/hypersensitive/hypersensitive.html
'''

class HyperSens(oc.ode_x_u.ode):
    def __init__(self):
        ############################################################
        args  = oc.ODEArguments(2,1)
        
        x0 = args.XVar(0)
        u   =args.UVar(0)
        
        x0dot = -(x0) + u
        jdot  = (u**2 +x0**2)
        
        ode = vf.stack(x0dot,jdot)
        ##############################################################
        super().__init__(ode,2,1)


xt0 = 1.5
xtf = 1.0
tf  = 10000.0

n   = 160

TrajIG =[[0.0,0,t,0] for t in np.linspace(0,tf,3*n)]

ode= HyperSens()

phase = ode.phase(Tmodes.LGL7,TrajIG,n)
#phase.integrator.setStepSizes(.1,.001,1000)
#phase.setControlMode("HighestOrderSpline")
phase.setControlMode("NoSpline")
phase.addBoundaryValue(PhaseRegs.Front,range(0,3),[xt0,0,0])
phase.addBoundaryValue(PhaseRegs.Back ,[0,2],[xtf,tf])
phase.addLUVarBound("Path",0,-50,50)
phase.addLUVarBound("Path",1,-50,50)
phase.addLUVarBound("Path",3,-50,50)

phase.addDeltaVarObjective(1,1.0)

phase.optimizer.OptLSMode = ast.Solvers.LineSearchModes.L1
phase.optimizer.SoeLSMode = ast.Solvers.LineSearchModes.L1

phase.optimizer.MaxLSIters = 2
phase.optimizer.PrintLevel = 1
phase.setThreads(8,8)
#phase.MeshIncFactor = 1.4
phase.AdaptiveMesh=True
phase.optimizer.set_QPOrderingMode("MINDEG")
phase.optimizer.QPPivotPerturb =6
phase.optimizer.MaxIters=100
phase.optimizer.EContol=1.0e-7
phase.MeshErrorEstimator='integrator'

import time

t00 = time.perf_counter()
phase.optimize_solve()    
tff = time.perf_counter()

print(tff-t00)

plt.plot(phase.MeshTimes,phase.MeshDistInt,marker='o')
plt.show()
TT = np.array(phase.returnTraj()).T


plt.plot(TT[2],TT[0])
#plt.plot(TT[2],TT[1])
plt.plot(TT[2],TT[3])
#plt.xscale("log")
plt.show()

