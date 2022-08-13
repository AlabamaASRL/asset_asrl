import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf = ast.VectorFunctions
oc = ast.OptimalControl
sol = ast.Solvers

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
Imodes = oc.IntegralModes
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


class LTModel(oc.ode_x_u.ode):
    def __init__(self, mu, ltacc):
        Xvars = 6
        Uvars = 3
        ############################################################
        args = oc.ODEArguments(Xvars, Uvars)
        r = args.head3()
        v = args.segment3(3)
        u = args.tail3()
        g = r.normalized_power3() * (-mu)
        thrust = u * ltacc
        acc = g + thrust
        ode = vf.stack([v, acc])
        #############################################################
        super().__init__(ode, Xvars, Uvars)

    class massobj(vf.ScalarFunction):
        def __init__(self, scale):
            u = Args(3)
            super().__init__(u.norm() * scale)
            
    class powerobj(vf.ScalarFunction):
        def __init__(self, scale):
            u = Args(3)
            super().__init__(u.norm().squared() * scale)


class LTModelSemiDirect(oc.ode_x_u.ode):
    def __init__(self, mu, ltacc):
        Xvars = 12
        Uvars = 1
        Ivars = Xvars + 1 + Uvars
        ############################################################
        args = Args(Ivars)

        r = args.head3()
        v = args.segment3(3)
        lr = args.segment3(6)
        lv = args.segment3(9)
        u = args[13]

        udir = lv.normalized()
        g = r.normalized_power3() * (-mu)
        thrust = (udir * u) * (-ltacc)
        acc = g + thrust

        lrdot = ((-3.0) * r.normalized_power5() * r.dot(lv) + lv * r.inverse_cubed_norm())
        lvdot = -lr

        ode = vf.stack([v, acc, lrdot, lvdot])
        ##############################################################
        super().__init__(ode, Xvars, Uvars)

    class massobj(vf.ScalarFunction):
        def __init__(self, scale):
            u = Args(1)[0]
            super().__init__(u * scale)

    class powerobj(vf.ScalarFunction):
        def __init__(self, scale):
            u = Args(1)[0]
            super().__init__(u.squared_norm() * scale)


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

XIG = np.zeros((10))
XIG[0:7] = X0

integ = ode.integrator(.01, Args(3).normalized() * .8, [3, 4, 5])
TrajIG = integ.integrate_dense(XIG, 6.4 * np.pi, 100)

phase = ode.phase('LGL3', TrajIG, 256)
#phase.setControlMode("BlockConstant")
phase.addBoundaryValue(PhaseRegs.Front, range(0, 7), X0)
phase.addLowerSquaredNormBound("Path",[7,8,9], 0.000001, 1.0)
phase.addUpperNormBound("Path",[7,8,9],  1, 1.0)


phase.addBoundaryValue(PhaseRegs.Back, range(0, 6), Xf[0:6])
phase.optimizer.PrintLevel = 0
phase.optimizer.BoundFraction = .995
phase.optimizer.set_OptLSMode("L1")
phase.optimizer.MaxLSIters = 2

phase.optimizer.deltaH = 1.0e-6
########################################
phase.addDeltaTimeObjective(1.0)


phase.optimize()
phase.optimizer.MaxLSIters = 0

TimeOptimal = phase.returnTraj()
TimeCostates = phase.returnCostateTraj()

phase.removeStateObjective(-1)
########################################
phase.addIntegralObjective(LTModel.powerobj(0.5), [7, 8, 9])
phase.optimize()

PowerOptimal = phase.returnTraj()
PowerCostates = phase.returnCostateTraj()
phase.removeIntegralObjective(-1)
#######################################


phase.addIntegralObjective(LTModel.massobj(1.0), [7, 8, 9])
phase.optimize()

MassOptimal = phase.returnTraj()
MassCostates = phase.returnCostateTraj()

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

MT = np.array(MassOptimal).T
CT = np.array(MassCostates).T
U = (MT[7] ** 2 + MT[8] ** 2 + MT[9] ** 2) ** .5

plt.plot(MT[6], MT[7] / U, color='red', label='Ux(t)')
plt.plot(MT[6], MT[8] / U, color='blue', label='Uy(t)')

N = (CT[3] ** 2 + CT[4] ** 2 + CT[5] ** 2) ** .5
#N=N/N
plt.plot(CT[6], -CT[3] / N, color='red', label='-Px(t)', linestyle='--', marker='.')
plt.plot(CT[6], -CT[4] / N, color='blue', label='-Py(t)', linestyle='--', marker='.')

plt.plot(CT[6], (N * acc - 1) * 10, color='green', label='S(t)')
plt.plot(MT[6], U, color='black', label='|U(t)|', zorder=10)

plt.legend()
plt.xlabel("t (ND)")
plt.grid(True)
plt.show()
###########################################

###########################################

MT = np.array(PowerOptimal).T
CT = np.array(PowerCostates[0:-1:4]).T
U = (MT[7] ** 2 + MT[8] ** 2 + MT[9] ** 2) ** .5

plt.plot(MT[6], MT[7], color='red', label='Ux(t)')
plt.plot(MT[6], MT[8], color='blue', label='Uy(t)')

plt.plot(CT[6], -CT[3] * acc, color='red', label='-Px(t)', linestyle='--', marker='.')
plt.plot(CT[6], -CT[4] * acc, color='blue', label='-Py(t)', linestyle='--', marker='.')

plt.plot(MT[6], U, color='black', label='|U(t)|', zorder=10)

plt.legend()
plt.xlabel("t (ND)")
plt.grid(True)
plt.show()
###########################################
'''
IdTraj = []

for i in range(0, len(MassOptimal)):
    XI = np.zeros((14))
    XI[0:6] = MassOptimal[i][0:6]
    XI[6:9] = MassCostates[i][0:3] * acc  
    XI[9:12] = MassCostates[i][3:6] * acc  
    XI[12] = MassOptimal[i][6]
    XI[13] = np.linalg.norm(MassOptimal[i][7:10]) * .99
    IdTraj.append(XI)

iode = LTModelSemiDirect(mu, acc)

iphase = iode.phase(Tmodes.LGL7)
iphase.setTraj(IdTraj, 512)
iphase.setControlMode(Cmodes.BlockConstant)
iphase.addBoundaryValue(PhaseRegs.Front, range(0, 6), X0[0:6])
iphase.addBoundaryValue(PhaseRegs.Front, [12], [0.0])
iphase.addBoundaryValue(PhaseRegs.Back, range(0, 6), Xf[0:6])
iphase.addBoundaryValue(PhaseRegs.Back, [12], [MassOptimal[-1][6]])

iphase.addLUVarBound(PhaseRegs.Path, 13, .0001, 1.0, 1.0)
iphase.addIntegralObjective(LTModelSemiDirect.massobj(1), [13])
iphase.optimizer.KKTtol = 1.0e-6
iphase.optimizer.deltaH = 1.0e-6
iphase.optimizer.OptLSMode = sol.LineSearchModes.L1
iphase.optimizer.MaxLSIters = 2
iphase.solve_optimize()

TT = np.array(iphase.returnTraj()).T
N = (TT[9] ** 2 + TT[10] ** 2 + TT[11] ** 2) ** .5

plt.plot(TT[12], TT[13], color='black', label='U(t)')
plt.plot(TT[12], -TT[9] / N, color='red', label='-Px(t)')
plt.plot(TT[12], -TT[10] / N, color='blue', label='-Py(t)')

plt.xlabel("t (ND)")
plt.grid(True)
plt.show()
'''