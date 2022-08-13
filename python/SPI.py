from AstroModels import TwoBody,TwoBody_SolarSail,SolarSail
from AstroConstraints import RendezvousConstraint,CosAlpha
from FramePlot import CRPlot,TBPlot,plt
import MKgSecConstants as c
import numpy as np
import asset as ast


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags

beta = .06
rcrank = .4


twobody = TwoBody(c.MuSun,c.AU)
SailModel = SolarSail(beta,False)
sail = TwoBody_SolarSail(c.MuSun,c.AU,SailModel = SailModel)

integ =twobody.integrator(c.pi/1000)
integ.Adaptive=True

def RetroFunc():
    args = Args(6)
    rhat = args.head3().normalized()
    vhat = args.tail3().normalized()
    return (rhat -.75*vhat).normalized()


sinteg =sail.integrator(c.pi/1000,RetroFunc(),range(0,6))
sinteg.Adaptive=True



IG =np.zeros((10))
IG[0]=1
IG[4]=1
IG[7]=1

IGCrank=np.zeros((7))
IGCrank[0]=rcrank
IGCrank[4]=1/np.sqrt(rcrank)


earth = integ.integrate_dense(IG[0:7],7.0,1000)
SpiralDown = sinteg.integrate_dense(IG,13.5,1000)
crankstart = integ.integrate_dense(IGCrank,7.0,1000)

##############################################################################
phase = sail.phase(Tmodes.LGL5,SpiralDown,800)
#phase.setControlMode(oc.ControlModes.BlockConstant)
phase.addBoundaryValue(PhaseRegs.Front, range(0,7), IG[0:7])
phase.addLUNormBound(PhaseRegs.Path,[7,8,9],.8,1.2)
MaxAlpha = 70.0
CosMaxAlpha =np.cos(np.deg2rad(MaxAlpha))
phase.addLowerFuncBound(PhaseRegs.Path,CosAlpha(),[0,1,2,7,8,9],CosMaxAlpha,10.0)


def VCirc(r):
    args = Args(6)
    rvec = args.head3()
    vvec = args.tail3()
    
    f1 = rvec.norm() + [-r]
    f2 = vvec.norm() + [-np.sqrt(1/r)]
    f3 = vf.dot(rvec.normalized(),vvec.normalized())
    f4 = rvec[2]
    f5 = vvec[2]
    return vf.Stack([f1,f2,f3,f4,f5])


phase.addEqualCon(PhaseRegs.Back,VCirc(rcrank),range(0,6))
phase.addDeltaTimeObjective(1)

phase.optimizer.OptLSMode = ast.LineSearchModes.L1
phase.optimizer.MaxLSIters = 3
phase.Threads = 18
phase.optimizer.QPThreads=6
phase.optimizer.MaxAccIters=190

phase.optimizer.incrH = 8
phase.optimizer.decrH = .33
phase.optimizer.deltaH = 1.0e-6
phase.optimizer.BoundFraction = .995

phase.solve_optimize()

SpiralDown = phase.returnTraj()




##############################################################################
DeltaT = 24.0
Tinc = SpiralDown[-1][6] + DeltaT

def RetroFunc():
    args = Args(6)
    rhat = args.head3().normalized()
    vhat = args.tail3().normalized()
    h = vf.cross(rhat,vhat).normalized()
    return (rhat -.75*h).normalized()

sinteg =sail.integrator(c.pi/7000,RetroFunc(),range(0,6))
sinteg.Adaptive=True
Crank = sinteg.integrate_dense(SpiralDown[-1],Tinc,1000)



cphase = sail.phase(Tmodes.LGL5,Crank,400)
#phase.setControlMode(oc.ControlModes.BlockConstant)
cphase.addBoundaryValue(PhaseRegs.Front, range(0,7), Crank[0][0:7])
cphase.addLUNormBound(PhaseRegs.Path,[7,8,9],.5,1.5)
cphase.addLUNormBound(PhaseRegs.Path,[0,1,2],.35,.45)

MaxAlpha = 70.0
CosMaxAlpha =np.cos(np.deg2rad(MaxAlpha))
cphase.addLowerFuncBound(PhaseRegs.Path,CosAlpha(),[0,1,2,7,8,9],CosMaxAlpha,10.0)

def IncObj():
    args = Args(6)
    rvec = args.head3()
    vvec = args.tail3()
    z = args.Constant([0,0,1])
    h = vf.cross(rvec,vvec).normalized()
    cosi=vf.dot(h,z)
    return (cosi*1.0)

cphase.addStateObjective(PhaseRegs.Back,IncObj(),range(0,6))

def VCirc2(r):
    args = Args(6)
    rvec = args.head3()
    vvec = args.tail3()
    
    f1 = rvec.norm() + [-r]
    f2 = vvec.norm() + [-np.sqrt(1/r)]
    f3 = vf.dot(rvec.normalized(),vvec.normalized())
    return vf.Stack([f1,f2,f3])

cphase.addEqualCon(PhaseRegs.Back,VCirc2(rcrank),range(0,6))
cphase.addDeltaTimeEqualCon(DeltaT)

cphase.optimizer.OptLSMode = ast.LineSearchModes.L1
cphase.optimizer.MaxLSIters = 1
cphase.Threads = 18
cphase.optimizer.QPThreads=6
cphase.optimizer.MaxAccIters=490
cphase.optimizer.MaxIters=4550

cphase.optimizer.incrH = 8
cphase.optimizer.decrH = .3
cphase.optimizer.deltaH = 1.0e-6
cphase.optimizer.BoundFraction = .995
#cphase.transcribe(True,True)
cphase.solve_optimize()
cphase.refineTrajManual(800)
cphase.solve()

Crank = cphase.returnTraj()

##############################################################################






plot= TBPlot(twobody,"Sun")
plot.addTraj(earth,"Earth",'g')
plot.addTraj(SpiralDown,"Spiral Down",'b')
#plot.addTraj(crankstart,"Crank Radius",'r')
plot.addTraj(Crank,"Crank Orbit",'r')

plot.Plot3d(pois=['P1'],bbox='One',legend=True)

