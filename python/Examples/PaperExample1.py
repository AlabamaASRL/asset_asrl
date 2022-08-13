import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf  = ast.VectorFunctions
oc  = ast.OptimalControl
sol = ast.Solvers

Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
Cmodes    = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags

class LTModel(oc.ode_x_u.ode):
    def __init__(self, mu, ltacc):
        Xvars = 6
        Uvars = 3
        ####################################
        args = Args(Xvars + Uvars + 1)
        r = args.head3()
        v = args.segment3(3)
        u = args.tail3()
        g = r.normalized_power3() * (-mu)
        thrust = u * ltacc
        acc = g + thrust
        ode = vf.stack([v, acc])
        ####################################
        super().__init__(ode, Xvars, Uvars)
        
mu = 1
acc = .02

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

XIG = np.zeros((10))
XIG[0:7] = X0


ode = LTModel(1, .02)
integ = ode.integrator(.01, Args(3).normalized() * .8, [3, 4, 5])
TrajIG = integ.integrate_dense(XIG, 6.4 * np.pi, 100)

phase = ode.phase(Tmodes.LGL3, TrajIG, 256)
phase.addBoundaryValue(PhaseRegs.Front, range(0, 7), X0)
phase.addLUNormBound(PhaseRegs.Path, [7, 8, 9], .001, 1.0, 1.0)
phase.addBoundaryValue(PhaseRegs.Back, range(0, 6), Xf[0:6])

phase.optimizer.PrintLevel = 1
phase.optimizer.deltaH = 1.0e-6

########################################
phase.addDeltaTimeObjective(1.0)
phase.optimize()
TimeOptimal = phase.returnTraj()

phase.removeStateObjective(-1)
########################################
phase.addIntegralObjective(Args(3).squared_norm()/2.0, [7, 8, 9])
phase.optimize()
PowerOptimal = phase.returnTraj()

phase.removeIntegralObjective(-1)
#######################################

phase.addIntegralObjective(Args(3).norm(), [7, 8, 9])
phase.optimize()
MassOptimal = phase.returnTraj()






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





###########################################
Plot(TrajIG, "Initial Guess", 'blue')
Scatter(X0, "X0", 'black')
Scatter(Xf, "XF", 'red')
plt.grid(True)
plt.axis("Equal")
plt.show()
###########################################


###########################################
Plot(TimeOptimal, "TimeOptimal", 'blue')
Plot(MassOptimal, "MassOptimal", 'green')
Plot(PowerOptimal, "PowerOptimal", 'red')
plt.legend()
Scatter(X0, "X0", 'black')
Scatter(Xf, "XF", 'gold')
plt.grid(True)
plt.axis("Equal")
plt.show()

ThrottlePlot(TimeOptimal, "TimeOptimal", 'blue')
ThrottlePlot(MassOptimal, "MassOptimal", 'green')
ThrottlePlot(PowerOptimal, "MassOptimal", 'red')
plt.grid(True)
plt.show()
###########################################
