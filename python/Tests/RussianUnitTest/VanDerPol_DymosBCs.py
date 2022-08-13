import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags


'''
Vanderpol Osscilator Optimization Problem Taken From 
https://openmdao.github.io/dymos/examples/vanderpol/vanderpol.html
'''

class VanderPol(oc.ode_x_u.ode):
    def __init__(self):
        ############################################################
        args  = oc.ODEArguments(2,1)
        x0    = args[0]
        x1    = args[1]
        u     = args[3]
        
        x0dot = (1.0 - x1**2)*x0 -x1 + u
        x1dot = x0
        ode = vf.stack(x0dot,x1dot)
        ##############################################################
        super().__init__(ode,2,1)

ode = VanderPol()

tf = 15.0

TrajIG = [[1,1,t,0] for t in np.linspace(0,tf,100)]

phase = ode.phase(Tmodes.LGL3,TrajIG,128)
phase.addBoundaryValue(PhaseRegs.Front,range(0,3),[1,1,0])
phase.addLUVarBound(PhaseRegs.Path,3,-0.75,1.0,1.0)
phase.addIntegralObjective(Args(3).squared_norm(),[0,1,3])
phase.addBoundaryValue(PhaseRegs.Back,[0,1,2],[0.0,0.0,tf])
phase.optimizer.PrintLevel=1
phase.optimizer.set_tols(1.0e-8,1.0e-8,1.0e-8)



phase.optimize()
Traj = phase.returnTraj()
T = np.array(Traj).T
plt.plot(T[2],T[0])
plt.plot(T[2],T[1])
plt.plot(T[2],T[3])
plt.show()
