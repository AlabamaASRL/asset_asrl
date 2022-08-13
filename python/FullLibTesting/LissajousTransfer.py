import numpy as np
import asset_asrl as ast
import asset_asrl.Astro as Astro
import asset_asrl.Astro.Constants as c
import asset_asrl.Astro.Date as date
from asset_asrl.Astro.FramePlot import plt,CRPlot,TBPlot,colpal
from asset_asrl.Astro.AstroModels import TwoBody,TwoBodyFrame,NBody,NBodyFrame
from asset_asrl.Astro.AstroModels import EPPR,EPPRFrame,CR3BP,CR3BPFrame
import spiceypy as sp
import time

from asset_asrl.OptimalControl import ODEBase

vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
sp.furnsh('BasicKernel.txt')


JD0 = date.date_to_jd(2023, 6, 14)
JDF = date.date_to_jd(2029, 6, 14)

SpiceFrame = "J2000"
N = 9000

Lstar = c.RadiusEarth

frame = EPPRFrame("SUN", c.MuSun, "Earth", c.MuEarth, c.AU, JD0, JDF)
frame.AddSpiceBodies(["MOON","VENUS","JUPITER BARYCENTER","SATURN BARYCENTER"])
frame.Add_P2_J2Effect(c.J2Earth,c.RadiusEarth)

ode = EPPR(frame,Enable_J2 = True)
integ = ode.integrator(c.pi/5000)
integ.Adaptive=True
integ.setAbsTol(1.0e-12)

JDImap  = 2460585.029363
IMAPECI = np.array([6104.326,2403.702,66.367,-3.254397,8.119686,6.657586])*1000.0
ImapIG = frame.Vector_to_Frame(IMAPECI, JDImap)

###########################################

ImapEDep = integ.integrate_dense(ImapIG,      ImapIG[6] + 1*c.day/frame.tstar)
ImapP0   = integ.integrate_dense(ImapEDep[-1],ImapIG[6] + 60*c.day/frame.tstar)
ImapP1   = integ.integrate_dense(ImapP0[-1],  ImapIG[6] + 110*c.day/frame.tstar)

#########################################

Liss = frame.GenL1Lissajous(.00065, .0007, 180, 270, 5, 1000,t0=ImapP1[-1][6])


LissF = Liss[-1]
ImapP2 = Liss[0:500]
ImapP3 = Liss[500:-1]

###########################################
ocp  = oc.OptimalControlProblem()


phase0 = ode.phase("LGL3",ImapP0[1:-1],128)
phase0.addBoundaryValue("Front",range(0,6),ImapP0[0][0:6])
phase0.addBoundaryValue("Front",[6],[ImapP0[0][6]])

phase1 = ode.phase("LGL3",ImapP1[1:-1],128)
phase2 = ode.phase("LGL3",ImapP2[1:-1],128)
phase3 = ode.phase("LGL3",ImapP3[1:-1],128)


phase3.addBoundaryValue("Back",range(0,3),LissF[0:3])
phase3.addLUFuncBound("Back",(Args(3)-LissF[3:6]).norm(),range(3,6),.0001/frame.vstar,10/frame.vstar)
phase3.addLUVarBound("Back",6,LissF[6]-c.day/frame.tstar,LissF[6]+c.day/frame.tstar)

phase0.addLowerDeltaTimeBound(10*c.day/frame.tstar)
phase1.addLowerDeltaTimeBound(50*c.day/frame.tstar)
phase2.addLowerDeltaTimeBound(10*c.day/frame.tstar)
phase3.addLowerDeltaTimeBound(10*c.day/frame.tstar)

ocp.addPhase(phase0)
ocp.addPhase(phase1)
ocp.addPhase(phase2)
ocp.addPhase(phase3)

ocp.addForwardLinkEqualCon(phase0,phase3,[0,1,2,6])

def DVObj():
    V1V2 = Args(6)
    V1 =V1V2.head3()
    V2 =V1V2.tail3()
    return (V2-V1).norm()

def DVCon():
    V1V2 = Args(9)
    V1 =V1V2.head3()
    V2 = V1V2.segment3(3)
    DV =V1V2.tail3()
    return (V1 + DV - V2)


ocp.setLinkParams(1*np.ones((9))/frame.vstar)
ocp.addLinkEqualCon(DVCon(),"BackToFront",
                [[0,1],[1,2],[2,3]],
                [[3,4,5],[3,4,5]],
                [],[],
                [range(0,3),range(3,6),range(6,9)])

ocp.addLinkParamObjective(Args(3).norm()*30,[range(0,3),range(3,6),range(6,9)])
ocp.addLinkParamInequalCon(.001/frame.vstar - Args(3).norm(),[range(0,3),range(3,6),range(6,9)])

ocp.optimizer.set_OptLSMode("AUGLANG")

ocp.optimizer.MaxLSIters = 4
ocp.optimizer.MaxAccIters = 100
ocp.optimizer.deltaH = 1.0e-6
ocp.optimizer.BoundFraction = .995


ocp.optimize()


ImapP0C = phase0.returnTraj()
ImapP1C = phase1.returnTraj()
ImapP2C = phase2.returnTraj()
ImapP3C = phase3.returnTraj()


##########################################

cols = colpal('plasma',4)


plot1 = CRPlot(ode)
plot1.addTraj(ImapEDep,"Ballistic",'k')
plot1.addTraj(ImapP0,"Phase0 Init Guess",cols[0])
plot1.addTraj(ImapP1,"Phase1 Init Guess",cols[1])
plot1.addTraj(ImapP2,"Phase2 Init Guess",cols[2])
plot1.addTraj(ImapP3,"Phase3 Init Guess",cols[3])


plot2 = CRPlot(ode)
plot2.addTraj(ImapEDep,"Ballistic",'k')
plot2.addTraj(ImapP0C,"Phase0 Converged",cols[0])
plot2.addTraj(ImapP1C,"Phase1 Converged",cols[1])
plot2.addTraj(ImapP2C,"Phase2 Converged",cols[2])
plot2.addTraj(ImapP3C,"Phase3 Converged",cols[3])


fig = plt.figure()
ax1 = fig.add_subplot(121,projection='3d')
ax2 = fig.add_subplot(122,projection='3d')

plot1.Plot3dAx(ax1,bbox="L1P2",pois=['L1',"P2"])
plot2.Plot3dAx(ax2,bbox="L1P2",pois=['L1',"P2"])

plt.show()