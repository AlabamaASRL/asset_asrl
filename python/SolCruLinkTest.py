from AstroModels import EPPRFrame,EPPR,EPPR_SolarSail,SolarSail,EPPR_LT,LowThrustAcc
from AstroConstraints import RendezvousConstraint,CosAlpha
from FramePlot import CRPlot,TBPlot,plt
import MKgSecConstants as c
import numpy as np
import asset as ast
from DerivChecker import FDDerivChecker
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

vf = ast.VectorFunctions
oc = ast.OptimalControl


ast.SoftwareInfo()

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
LinkRegs = oc.LinkFlags
sol = ast.Solvers



JDImap = 2460585.02936299
JD0 = JDImap - 3.5
JDF = JD0 + 3.0*365.0   
N = 4000

SpiceFrame = 'J2000'
EFrame = EPPRFrame("SUN",c.MuSun,"EARTH",c.MuEarth,c.AU,JD0,JDF,N=N,SpiceFrame=SpiceFrame)
Bodies = ["MOON","JUPITER BARYCENTER","VENUS"]
EFrame.AddSpiceBodies(Bodies,N=2000)
EFrame.Add_P2_J2Effect()


eppr  = EPPR(EFrame,Enable_J2=True)

beta = 0.02
SailModel = SolarSail(beta,False)
sail = EPPR_SolarSail(EFrame,SailModel = SailModel)


epinteg =eppr.integrator(c.pi/50000)
epinteg.Adaptive=True

sinteg =sail.integrator(c.pi/7000,(Args(3)).normalized(),range(0,3))
sinteg.Adaptive=True



Day = c.day/EFrame.tstar

Imap  = EFrame.Copernicus_to_Frame("COP.csv",center="P2")
IG = Imap[0]

Tdep = 116.5*Day
Ts = 197*Day
ImapNom = epinteg.integrate_dense(IG,IG[6] + Tdep)
IT = epinteg.integrate_dense(IG,IG[6] + 190*Day,100000)

ITab = oc.LGLInterpTable(6,IT,len(IT)*4)

IG = np.zeros((10))
IG[0:7]=ImapNom[-1]
IG[7]=1.0




SolCru = sinteg.integrate_dense(IG,IG[6] + Ts)

#############################################################
SORB = sail.CR3BP_ZeroAlpha.GenSubL1Lissajous(.0020,.000,180,0,1,500,0)
ophase = sail.CR3BP_ZeroAlpha.phase(Tmodes.LGL3)
ophase.setTraj(SORB,600)
ophase.addBoundaryValue(PhaseRegs.Front,[1,3,5,6],[0,0,0,0])
ophase.addBoundaryValue(PhaseRegs.Front,[2],[-.0010])
ophase.addBoundaryValue(PhaseRegs.Back,[1,3,5],[0,0,0])
ophase.optimizer.EContol = 1.0e-12
ophase.optimizer.PrintLevel = 0
ophase.optimizer.QPThreads = 1

ophase.solve()
SORB = ophase.returnTraj()
OTab = oc.LGLInterpTable(6,SORB,len(SORB)*3)
OTab.makePeriodic()
################################################################    


N = 512
np.set_printoptions(precision  = 6, linewidth = 400)
print(SolCru[0])
FDDerivChecker(sail.vf(),SolCru[0])
np.set_printoptions(precision  = 6, linewidth = 400)

import time
t0=time.perf_counter()
sail.vf().rpt(SolCru[0],100000)
tf=time.perf_counter()
print(tf-t0)
sail.vf().SuperTest(SolCru[0],100000)
ast.PyMain()
input("s")

Cm = Cmodes.BlockConstant
Tm = Tmodes.LGL5
sphase = sail.phase(Tm,SolCru,N)
sphase.addEqualCon(PhaseRegs.Front,RendezvousConstraint(ITab,range(0,6)),range(0,7))
sphase.setControlMode(Cm)
P=False
#sphase.EnableHessianSparsity = P
sphase.setStaticParams([0.3])
MaxAlpha = 17.0

CosMaxAlpha =np.cos(np.deg2rad(MaxAlpha))
sphase.addLowerFuncBound(PhaseRegs.Path,CosAlpha(),[0,1,2,7,8,9],CosMaxAlpha,10.0)
sphase.addLUNormBound(PhaseRegs.Path,[7,8,9],.8,1.2,.1)
#sphase.addValueObjective(PhaseRegs.Back,6,1.0)
sphase.addDeltaTimeObjective(1.0)
sphase.addEqualCon(PhaseRegs.Back,RendezvousConstraint(OTab,range(0,6)),range(0,6),[],[0])
sphase.Threads=16
sphase.optimizer.QPThreads =8
sphase.optimizer.QPOrderingMode = sol.QPOrderingModes.MINDEG

#sphase.enable_vectorization(False)

sphase.optimizer.incrH = 8
sphase.optimizer.decrH = .33
sphase.optimizer.deltaH = 1.0e-6
sphase.optimizer.OptLSMode = sol.LineSearchModes.L1
sphase.optimizer.PrintLevel = 0
sphase.optimizer.MaxLSIters = 1
sphase.optimizer.QPOrderingMode = sol.QPOrderingModes.METIS

#sphase.test_threads()

t0=time.perf_counter()

sphase.solve_optimize()


tf=time.perf_counter()

t0=time.perf_counter()
sphase.test_threads(16,16,10000)
tf=time.perf_counter()
print("16:",tf-t0)


#sail.vf().SuperTest(SolCru[0],100000)

SolCru=sphase.returnTraj()



#############################################################################









    
