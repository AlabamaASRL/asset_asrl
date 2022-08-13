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
        
        x0dot = -x0 + u
        jdot  = (u**2 +x0**2)/2.0
        
        ode = vf.stack(x0dot,jdot)
        ##############################################################
        super().__init__(ode,2,1)


xt0 =1.5
xtf =1.0
tf  =20.0

TrajIG =[[xt0*(1-t/tf) + xtf*(t/tf),0,t,.01] for t in np.linspace(0,tf,100)]
ode= HyperSens()

phase = ode.phase(Tmodes.LGL3,TrajIG,128)
phase.addBoundaryValue(PhaseRegs.Front,range(0,3),[xt0,0,0])
phase.addBoundaryValue(PhaseRegs.Back ,[0,2],[xtf,tf])
phase.addDeltaVarObjective(1,1.0)

phase.optimizer.OptLSMode = ast.Solvers.LineSearchModes.L1
phase.optimizer.MaxLSIters = 2
phase.optimizer.PrintLevel = 1
phase.Threads=8
phase.optimize()    

TT = np.array(phase.returnTraj()).T


plt.plot(TT[2],TT[0])
plt.plot(TT[2],TT[1])
plt.plot(TT[2],TT[3])

plt.show()

