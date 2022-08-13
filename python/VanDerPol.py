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

https://web.casadi.org/docs/#a-simple-test-problem
'''

class VanderPol(oc.ode_2_1.ode):
    def __init__(self):
        ############################################################
        args  = Args(4)
        x0    = args[0]
        x1    = args[1]
        u     = args[3]
        
        x0dot = (1.0 - x1**2)*x0 -x1 + u
        x1dot = x0
        ode = vf.Stack([x0dot,x1dot])
        ##############################################################
        super().__init__(ode,2,1)

ode = VanderPol()
TrajIG = [[0,1,t,0] for t in np.linspace(0,10,100)]

ode.vf().SuperTest(TrajIG[0],1000000)
input("s")

phase = ode.phase(Tmodes.LGL3,TrajIG,128)
#phase.setControlMode(oc.ControlModes.BlockConstant)
phase.addBoundaryValue(PhaseRegs.Front,range(0,3),[0,1,0])
phase.addLUVarBound(PhaseRegs.Path,3,-1.0,1.0,1.0)
phase.addLowerVarBound(PhaseRegs.Path,0,-0.75,1.0)
phase.addIntegralObjective(Args(3).squared_norm(),[0,1,3])
phase.addBoundaryValue(PhaseRegs.Back,[2],[10])
phase.optimizer.PrintLevel=0
phase.optimizer.QPThreads =8
#phase.optimizer.QPOrderingMode = ast.Solvers.QPOrderingModes.MINDEG
phase.Threads =8

phase.optimize()
Traj = phase.returnTraj()

T = np.array(Traj).T

plt.plot(T[2],T[0])
plt.plot(T[2],T[1])
plt.plot(T[2],T[3])

plt.show()

