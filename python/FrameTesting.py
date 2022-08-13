from AstroModels import EPPRFrame,EPPR,EPPR_SolarSail,SolarSail
import MKgSecConstants as c
from FramePlot import CRPlot,TBPlot,plt
import numpy as np
import asset as ast
from AstroConstraints import RendezvousConstraint,CosAlpha


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags

JDImap = 2460585.02936299
JD0 = JDImap - 3.5
JDF = JD0 + 5.0*365.0   
N = 10000

SpiceFrame = 'J2000'

Bodies = ["MOON","JUPITER BARYCENTER","VENUS","SATURN BARYCENTER","MARS BARYCENTER","URANUS BARYCENTER","NEPTUNE BARYCENTER"]

EFrame = EPPRFrame("SUN",c.MuSun,"EARTH",c.MuEarth,c.AU,JD0,JDF,N=N,SpiceFrame=SpiceFrame)
EFrame.AddSpiceBodies(Bodies,N=6000)
EFrame.Add_P2_J2Effect()





eppr  = EPPR(EFrame,Enable_J2=True)

beta = 0.02
SailModel = SolarSail(.02,False)
sail = EPPR_SolarSail(EFrame,SailModel = SailModel)




epinteg =eppr.integrator(c.pi/50000)
epinteg.MaxStepSize*=1000
epinteg.Adaptive=True

sinteg =sail.integrator(c.pi/7000,(Args(3)).normalized(),range(0,3))
sinteg.MaxStepSize*=1000
sinteg.Adaptive=True



Day = c.day/EFrame.tstar

Imap  = EFrame.Copernicus_to_Frame("COP.csv",center="P2")
IG = Imap[0]

Tdep = 117.1*Day
Ts = 197*Day
ImapNom = epinteg.integrate_dense(IG,IG[6] + Tdep)
IT = epinteg.integrate_dense(IG,IG[6] + 250*Day)

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
ophase.addBoundaryValue(PhaseRegs.Front,[2],[-.0011])
ophase.addBoundaryValue(PhaseRegs.Back,[1,3,5],[0,0,0])
ophase.optimizer.EContol = 1.0e-12
ophase.solve()
SORB = ophase.returnTraj()
OTab = oc.LGLInterpTable(6,SORB,len(SORB)*3)
OTab.makePeriodic()
################################################################    


    
    
    
sphase = sail.phase(Tmodes.LGL7,SolCru,400)
sphase.addEqualCon(PhaseRegs.Front,RendezvousConstraint(ITab,range(0,6)),range(0,7))
sphase.setStaticParams([0.3])
MaxAlpha = 35.0
CosMaxAlpha =np.cos(np.deg2rad(MaxAlpha))
sphase.addLowerFuncBound(PhaseRegs.Path,CosAlpha(),[0,1,2,7,8,9],CosMaxAlpha,10.0)
sphase.addValueObjective(PhaseRegs.Back,6,1.0)
sphase.addEqualCon(PhaseRegs.Back,RendezvousConstraint(OTab,range(0,6)),range(0,6),[],[0])
sphase.Threads=12

sphase.optimizer.incrH = 8
sphase.optimizer.decrH = .1
sphase.optimizer.deltaH = 1.0e-6
sphase.optimizer.OptLSMode = ast.LineSearchModes.L1
sphase.optimizer.MaxLSIters =2
sphase.optimizer.BoundFraction = .99


sphase.optimize()
SolCru=sphase.returnTraj()



plot= CRPlot(EFrame)
plot.addTraj(Imap,"IMAP-Nominal",'k')
plot.addTraj(IT,"IMAP-NominalR",'b')
plot.addTraj(SORB,"iig",'k')

plot.addTraj(SolCru,"SolCru",'r')
plot.addPoint(sail.SubL1,"Sub-L1",'purple',marker='.')

plot.Plot3d(pois=['L1','P2','L2'],bbox='L1P2',legend=False)









