import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from DerivChecker import FDDerivChecker
import sys
import jedi

ast.PyMain()
vf = ast.VectorFunctions
oc = ast.OptimalControl
sol = ast.Solvers

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
Imodes = oc.IntegralModes

ff = Args(2)

PhaseRegs = oc.PhaseRegionFlags


def Plot(Traj, name, col, ax=plt, linestyle='-'):
    TT = np.array(Traj).T
    ax.plot(TT[0], TT[1], label=name, color=col, linestyle=linestyle)


def Scatter(State, name, col, ax=plt):
    ax.scatter(State[0], State[1], label=name, c=col)


def ThrottlePlot(Traj, name, col, ax=plt):
    TT = np.array(Traj).T
    ax.plot(TT[6], (TT[7] ** 2 + TT[8] ** 2 + TT[9] ** 2) ** .5, label=name, color=col)


def ThrottlePlot2(Traj, name, col, ax=plt):
    TT = np.array(Traj).T
    ax.plot(TT[6], TT[7], label=name + " x", color=col)
    ax.plot(TT[6], TT[8], label=name + " y", color=col)
    ax.plot(TT[6], TT[9], label=name + " z", color=col)
    ax.plot(TT[6], (TT[7] ** 2 + TT[8] ** 2 + TT[9] ** 2) ** .5, label=name + " |mag|", color=col)


class LTModel(oc.ode_x_u_p.ode):
    def __init__(self, mu, ltacc):
        Xvars = 6
        Uvars = 3
        Pvars = 1
        ############################################################
        args = oc.ODEArguments(Xvars, Uvars, Pvars)
        r = args.head3()
        v = args.segment3(3)
        u = args.UVec().head3()
        p = args.PVec()[0]
        g = r.normalized_power3() * (-mu)
        thrust = (u *p)*ltacc
        acc = g + thrust
        ode = vf.Stack([v, acc])
        #############################################################
        super().__init__(ode, Xvars, Uvars,Pvars)

    class massobj(vf.ScalarFunction):
        def __init__(self, scale):
            u = Args(3)
            super().__init__(u.norm() * scale)

    class powerobj(vf.ScalarFunction):
        def __init__(self, scale):
            u = Args(3)
            super().__init__(u.norm().squared() * scale)


mu = 1
acc = .02
ode = LTModel(mu, acc)

r0 = 1.0
v0 = np.sqrt(mu / r0)

rf = 2.0
vF = np.sqrt(mu / rf)

X0 = np.zeros((7))
X0[0] = r0
X0[4] = v0

Xf = np.zeros((6))
Xf[0] = rf
Xf[4] = vF

XIG = np.zeros((11))
XIG[0:7] = X0
XIG[7] = .99
XIG[10] = 0.6

integ = ode.integrator(.01, Args(3).normalized(), [3, 4, 5])
TrajIG = integ.integrate_dense(XIG, 6.4 * np.pi, 100)





phase = ode.phase(Tmodes.LGL3, TrajIG, 256)
phase.addBoundaryValue(PhaseRegs.ODEParams,[0],[.3])
phase.addBoundaryValue(PhaseRegs.Front, range(0, 7), X0)
phase.addLUNormBound(PhaseRegs.Path, [7, 8, 9], .001, 1.0, 1.0)

phase.solve()


###########################################
Plot(TrajIG, "Initial Guess", 'blue')
Plot(phase.returnTraj(), "Reslove", 'red')

Scatter(X0, "X0", 'black')
Scatter(Xf, "XF", 'red')
plt.grid(True)
plt.axis("Equal")
plt.show()
###########################################



