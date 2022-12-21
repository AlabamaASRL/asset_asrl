import asset as ast
import spiceypy as spice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import MKgSecConstants as c
################################################################################
# Setup
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
TModes = oc.TranscriptionModes
spice.furnsh("standard.html")

###############################################################################


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


def GetEphemTraj(body, startDay, endDay, numstep, LU=1.0, TU=1.0,
                 Frame='ECLIPJ2000', Center='SOLAR SYSTEM BARYCENTER'):
    startET = spice.str2et(startDay)
    endET = spice.str2et(endDay)
    times = [startET + (endET - startET)*x/numstep for x in range(numstep)]

    states = []
    t0 = times[0]
    for t in times:
        X = np.zeros((7))
        X[0:6] = spice.spkezr(body, t, Frame, 'NONE', Center)[0]
        X[0:3] *= 1000.0/LU
        X[3:6] *= 1000.0*TU/LU
        X[6] = (t-t0)/TU
        states.append(X)
    return states


def VinfMatchCon(tab):
    X = Args(8)
    v0 = X.head3()
    t0 = X[3]

    v1 = X.tail(4).head3()
    t1 = X.tail(4)[3]

    BodyV = oc.InterpFunction(tab, range(3, 6)).vf()
    vInfPlus = (BodyV.eval(t0)-v0).norm()
    vInfMinus = (BodyV.eval(t1)-v1).norm()
    return (vInfPlus-vInfMinus)


def RendCon(tab):
    XT = Args(7)
    x = XT.head(6)
    t = XT[6]
    fun = oc.InterpFunction(tab, range(0, 6)).vf()
    return fun.eval(t) - x


def PosCon(tab):
    XT = Args(4)
    x = XT.head(3)
    t = XT[3]
    fun = oc.InterpFunction(tab, range(0, 3)).vf()
    return fun.eval(t) - x


def VinfFunc(tab):
    XT = Args(4)
    x = XT.head(3)
    t = XT[3]
    fun = oc.InterpFunction(tab, range(3, 6)).vf()
    return fun.eval(t) - x


def FlybyAngleBound(tab, mubod, minrad):
    X = Args(8)
    v0 = X.head3()
    t0 = X[3]

    v1 = X.tail(4).head3()
    t1 = X.tail(4)[3]

    BodyV = oc.InterpFunction(tab, range(3, 6)).vf()
    v0dir = (BodyV.eval(t0)-v0).normalized()
    v1dir = (BodyV.eval(t1)-v1).normalized()

    vInf2 = (BodyV.eval(t0)-v0).squared_norm()

    delta = vf.arccos(vf.dot(v0dir, v1dir))
    deltaMax = 2*vf.arcsin(mubod/(mubod + minrad*vInf2))
    return delta - deltaMax


if __name__ == "__main__":

    lstar = c.AU
    vstar = np.sqrt(c.MuSun/lstar)
    tstar = np.sqrt(lstar**3/c.MuSun)
    astar = c.MuSun/lstar**2

    engacc = (1./1800)/astar

    print(engacc)

    data_start    = 'July 1, 2022'
    utc_start     = 'Aug 1, 2022 '
    utc_MarsFlyBy = 'May 15, 2023'
    utc_arrival   = 'Nov 1, 2025 '
    data_end      = 'Jan 1, 2028 '

    EarthDat = GetEphemTraj("EARTH", data_start, data_end,
                            9000, LU=lstar, TU=tstar, Center="SUN")
    MarsDat  = GetEphemTraj("Mars", data_start, data_end,
                           9000, LU=lstar, TU=tstar, Center="SUN")
    PsycheDat = GetEphemTraj("PSYCHE", data_start,
                             data_end, 9000, LU=lstar, TU=tstar, Center="SUN")

    t0_nd = spice.utc2et(data_start)/tstar
    ts_ig = spice.utc2et(utc_start)/tstar - t0_nd
    tm_ig = spice.utc2et(utc_MarsFlyBy)/tstar - t0_nd
    tp_ig = spice.utc2et(utc_arrival)/tstar - t0_nd
    te = spice.utc2et(data_end)/tstar - t0_nd

    EarthTab = oc.LGLInterpTable(6, EarthDat, 9000)
    MarsTab = oc.LGLInterpTable(6, MarsDat, 9000)
    PsycheTab = oc.LGLInterpTable(6, PsycheDat, 9000)

    ED = np.array(EarthDat).T
    MD = np.array(MarsDat).T
    PD = np.array(PsycheDat).T

    EarthStart = EarthTab.Interpolate(ts_ig)
    MarsFB = MarsTab.Interpolate(tm_ig)
    PsycheArr = PsycheTab.Interpolate(tp_ig)

    ode = LTModel(1.0, engacc)
    integ1 = ode.integrator(.001, Args(3).normalized()*0.35, [3, 4, 5])
    integ2 = ode.integrator(.001, Args(3).normalized()*0.16, [3, 4, 5])

    E0 = np.zeros((10))
    E0[0:7] = EarthStart
    E0[3:6] += (EarthStart[3:6]/np.linalg.norm(EarthStart[3:6]))*1000.0/vstar

    M0 = np.zeros((10))
    M0[0:7] = MarsFB
    M0[3:6] += (MarsFB[3:6]/np.linalg.norm(MarsFB[3:6]))*10.0/vstar

    EarthMarsIG = integ1.integrate_dense(E0, tm_ig, 300)
    MarsPsycheIG = integ2.integrate_dense(M0, tp_ig, 300)

    EMIG = np.array(EarthMarsIG).T
    MPIG = np.array(MarsPsycheIG).T

    plt.plot(EMIG[0], EMIG[1], color='k')
    plt.plot(MPIG[0], MPIG[1], color='k')

    plt.plot(ED[0], ED[1], color='green')
    plt.plot(MD[0], MD[1], color='red')
    plt.plot(PD[0], PD[1], color='blue')

    plt.scatter(EarthStart[0], EarthStart[1], color='green')
    plt.scatter(MarsFB[0], MarsFB[1], color='red')
    plt.scatter(PsycheArr[0], PsycheArr[1], color='blue')

    plt.show()

    phase1 = ode.phase(TModes.LGL3, EarthMarsIG, 128)

    phase1.addEqualCon(PhaseRegs.Front, PosCon(EarthTab), [0, 1, 2, 6])
    phase1.addEqualCon(PhaseRegs.Back, PosCon(MarsTab),  [0, 1, 2, 6])
    phase1.addLowerVarBound(PhaseRegs.Front, 6, 0.0, 1.0)

    phase1.addUpperFuncBound(PhaseRegs.Front, VinfFunc(EarthTab).squared_norm(),
                             [3, 4, 5, 6], (2000.0/vstar)**2, 10.0)
    phase1.addLUNormBound(PhaseRegs.Path, [7, 8, 9], .001, 1.0, 1.0)

    phase2 = ode.phase(TModes.LGL3, MarsPsycheIG, 256)

    phase2.addEqualCon(PhaseRegs.Back, RendCon(PsycheTab),  range(0, 7))
    phase2.addLUNormBound(PhaseRegs.Path, [7, 8, 9], .001, 1.0, 1.0)
    phase2.addUpperVarBound(PhaseRegs.Back, 6, te - 24.0*3600*745/tstar, 1.0)

    r_Mars_Psyche = (3389.5 + 500)*1000.0/lstar
    mu_Val_Mars = c.MuMars/c.MuSun

    ocp = oc.OptimalControlProblem()

    ocp.addPhase(phase1)
    ocp.addPhase(phase2)

    
    ocp.addForwardLinkEqualCon(phase1, phase2, [0, 1, 2, 6])

    
    #ocp.addLinkEqualCon(VinfMatchCon(MarsTab),"BackToFront", [[phase1, phase2]], [3, 4, 5, 6])

    ocp.addLinkEqualCon(VinfMatchCon(MarsTab),phase1,'Back',range(3,7),phase2,'Front',range(3,7))


    FB = FlybyAngleBound(MarsTab, mu_Val_Mars, r_Mars_Psyche)
    reg   = "BackToFront"
    reg   = ["Back","Front"]
    ptl   = [[phase1, phase2]]
    ptl   = [[0, 1]]
    indxs = [3, 4, 5, 6]
    xtvs  = [indxs,indxs]
    empty = [[],[]]
    lpvs  = [[]]

    #ocp.addLinkInequalCon(FB,reg, ptl, indxs)
    #ocp.addLinkInequalCon(FB,reg, ptl, xtvs,[],[],[])

    ocp.addLinkInequalCon(FB,phase1,'Back',[3, 4, 5, 6],phase2,'Front',[3, 4, 5, 6])


    ocp.optimizer.OptLSMode = ast.Solvers.LineSearchModes.L1
    ocp.optimizer.MaxLSIters = 1
    ocp.optimizer.MaxAccIters = 100
    ocp.optimizer.deltaH = 1.0e-7
    ocp.optimizer.decrH = .1
    ocp.optimizer.BoundFraction = .997
    ocp.optimizer.PrintLevel = 1

    phase1.setControlMode(oc.ControlModes.BlockConstant)
    phase2.setControlMode(oc.ControlModes.BlockConstant)

    t0 = time.perf_counter()
    ###########################################
    phase1.addDeltaTimeObjective(1.0)
    phase2.addDeltaTimeObjective(1.0)
    ocp.solve_optimize()

    EarthMarsT = phase1.returnTraj()
    MarsPsycheT = phase2.returnTraj()

    phase1.removeStateObjective(-1)
    phase2.removeStateObjective(-1)
    ##########################################

    phase1.addIntegralObjective(Args(3).squared_norm(), [7, 8, 9])
    phase2.addIntegralObjective(Args(3).squared_norm(), [7, 8, 9])
    ocp.optimize()

    EarthMarsP = phase1.returnTraj()
    MarsPsycheP = phase2.returnTraj()

    phase1.removeIntegralObjective(-1)
    phase2.removeIntegralObjective(-1)
    ############################################

    phase1.addIntegralObjective(Args(3).norm(), [7, 8, 9])
    phase2.addIntegralObjective(Args(3).norm(), [7, 8, 9])
    ocp.optimize()

    EarthMarsM = phase1.returnTraj()
    MarsPsycheM = phase2.returnTraj()

    phase1.removeIntegralObjective(-1)
    phase2.removeIntegralObjective(-1)

    #######################################################
    tf = time.perf_counter()

    print((tf-t0))
    EarthMars = phase1.returnTraj()
    MarsPsyche = phase2.returnTraj()

    EMT = np.array(EarthMarsT).T
    MPT = np.array(MarsPsycheT).T
    EMP = np.array(EarthMarsP).T
    MPP = np.array(MarsPsycheP).T
    EMM = np.array(EarthMarsM).T
    MPM = np.array(MarsPsycheM).T

    plt.plot(EMT[0], EMT[1], color='k', label='Time-Optimal')
    plt.plot(MPT[0], MPT[1], color='k')

    plt.plot(EMP[0], EMP[1], color='k',
             linestyle='dashed', label='Power-Optimal')
    plt.plot(MPP[0], MPP[1], color='k', linestyle='dashed')

    plt.plot(EMM[0], EMM[1], color='k',
             linestyle='dotted', label='Mass-Optimal')
    plt.plot(MPM[0], MPM[1], color='k', linestyle='dotted')

    plt.plot(ED[0], ED[1], color='green')
    plt.plot(MD[0], MD[1], color='red')
    plt.plot(PD[0], PD[1], color='blue')

    print((MarsPsyche[0]-EarthMars[-1])[3:6]*vstar)

    plt.scatter(EarthMarsT[0][0], EarthMarsT[0][1],
                color='green', label='Time-Optimal', zorder=10)
    plt.scatter(MarsPsycheT[0][0], MarsPsycheT[0][1], color='red', zorder=10)
    plt.scatter(MarsPsycheT[-1][0], MarsPsycheT[-1]
                [1], color='blue', zorder=10)

    plt.scatter(EarthMarsP[0][0], EarthMarsP[0][1], color='green',
                marker="*", label='Power-Optimal', zorder=10)
    plt.scatter(MarsPsycheP[0][0], MarsPsycheP[0][1],
                color='red', marker="*", zorder=10)
    plt.scatter(MarsPsycheP[-1][0], MarsPsycheP[-1][1],
                color='blue', marker="*", zorder=10)

    plt.scatter(EarthMarsM[0][0], EarthMarsM[0][1], color='green',
                marker="s", label='Mass-Optimal', zorder=10)
    plt.scatter(MarsPsycheM[0][0], MarsPsycheM[0][1],
                color='red', marker="s", zorder=10)
    plt.scatter(MarsPsycheM[-1][0], MarsPsycheM[-1][1],
                color='blue', marker="s", zorder=10)

    plt.scatter(0, 0, color='gold', marker='o')
    plt.legend()
    plt.grid(True)
    plt.axis("Equal")
    plt.xlabel("X (AU)")
    plt.ylabel("Y (AU)")

    plt.show()
    #########################################################
    EMTN = (EMT[7]**2 + EMT[8]**2 + EMT[9]**2)**.5
    MPTN = (MPT[7]**2 + MPT[8]**2 + MPT[9]**2)**.5
    plt.plot(EMT[6], EMTN, color='r', label='Time-Optimal')
    plt.plot(MPT[6], MPTN, color='r')

    plt.plot([EMT[6][-1], EMT[6][-1]], [0, 1], color='r', linestyle='--')

    EMTN = (EMP[7]**2 + EMP[8]**2 + EMP[9]**2)**.5
    MPPN = (MPP[7]**2 + MPP[8]**2 + MPP[9]**2)**.5
    plt.plot(EMP[6], EMTN, color='g', label='Power-Optimal')
    plt.plot(MPP[6], MPPN, color='g')
    plt.plot([EMP[6][-1], EMP[6][-1]], [0, 1], color='g', linestyle='--')

    EMMN = (EMM[7]**2 + EMM[8]**2 + EMM[9]**2)**.5
    MPMN = (MPM[7]**2 + MPM[8]**2 + MPM[9]**2)**.5
    plt.plot(EMM[6], EMMN, color='b', label='Mass-Optimal')
    plt.plot(MPM[6], MPMN, color='b')
    plt.plot([EMM[6][-1], EMM[6][-1]], [0, 1], color='b', linestyle='--')

    plt.xlabel("Time (ND)")
    plt.ylabel("|U|")
    plt.legend()
    plt.grid(True)
    plt.show()

    #####################################################
