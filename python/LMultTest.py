import numpy as np
import asset as ast
import matplotlib.pyplot as plt

norm = np.linalg.norm
vf = ast.VectorFunctions
oc = ast.OptimalControl


Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

def ODE():
    args = Args(3)
    x=args[0]
    u=args[2]
    xdot = .5*x + u
    return xdot

def Obj():
    args = Args(2)
    x=args[0]
    u=args[1]
    return u*u + x*u + 1.25*x*x


ode = oc.ode_x_x.ode(ODE(),1,1)


IG  = [1,0.0,0.0000]
TIG = [[1,t,0.01] for t in np.linspace(0,1,100)]

phase = ode.phase(Tmodes.LGL3)
phase.setTraj(TIG,50)
#phase.setControlMode(oc.BlockConstant)
phase.addBoundaryValue(PhaseRegs.Front,[0,1],[1,0])
phase.addBoundaryValue(PhaseRegs.Back, [1],[1])
phase.addIntegralObjective(Obj(),[0,2])
phase.optimize()
Traj = phase.returnTraj()
CTraj= phase.returnCostateTraj()
T = np.array(Traj).T
CT = np.array(CTraj).T

X = T[0]
t = T[1]
U = T[2]
lstar = 2*np.cosh(1-t)*np.tanh(1-t)/np.cosh(1)
lm = CT[0]

plt.plot(t,(abs(lstar-lm)))
plt.show()

    
    