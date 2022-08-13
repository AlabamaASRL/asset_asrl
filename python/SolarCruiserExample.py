from AstroModels import EPPRFrame,EPPR,EPPR_SolarSail,SolarSail,EPPR_LT,LowThrustAcc,CR3BP,CR3BPFrame
from AstroConstraints import RendezvousConstraint,CosAlpha
from FramePlot import CRPlot,TBPlot,plt
import MKgSecConstants as c
import numpy as np
import asset as ast


print((1.01979e17/(111.3))*1653*2/c.MuSun)


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags



def normalized(x): return x/np.linalg.norm(x)

JDImap = 2460585.02936299
JD0 = JDImap - 3.5
JDF = JD0 + 3.0*365.0   
N = 4000

SpiceFrame = 'J2000'
EFrame = EPPRFrame("SUN",c.MuSun,"EARTH",c.MuEarth,c.AU,JD0,JDF,N=N,SpiceFrame=SpiceFrame)
Bodies = ["MOON","JUPITER BARYCENTER","VENUS"]
EFrame.AddSpiceBodies(Bodies,N=4000)
EFrame.Add_P2_J2Effect()


eppr  = EPPR(EFrame,Enable_J2=False)




beta = 0.02
SailModel = SolarSail(beta,False)
sail = EPPR_SolarSail(EFrame,SailModel = SailModel)


epinteg =eppr.integrator(c.pi/50000)
epinteg.Adaptive=True

sinteg =sail.integrator(c.pi/7000,(Args(3).normalized() + Args(3).Constant([0,-.4,0])).normalized(),range(0,3))
sinteg.Adaptive=True
cr3bp = CR3BP(c.MuSun, c.MuEarth, c.AU)

integ =cr3bp.integrator(c.pi/50000)
#integ.Adaptive=True


Day = c.day/EFrame.tstar

Imap  = EFrame.Copernicus_to_Frame("COP.csv",center="P2")
IG = Imap[0]

Tdep = 34.5*Day
Ts = 150*Day
ImapNom = epinteg.integrate_dense(IG,IG[6] + Tdep)
IT = epinteg.integrate_dense(IG,IG[6] + 190*Day,100000)
IGG = np.copy(IG)
IGG[3:6]+=normalized(IG[3:6])*16.45/cr3bp.vstar
ImapNom3 = integ.integrate_dense(IGG,IGG[6] + Tdep*1.5)

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
ophase.optimizer.PrintLevel = 1

ophase.solve()
SORB = ophase.returnTraj()
OTab = oc.LGLInterpTable(6,SORB,len(SORB)*3)
OTab.makePeriodic()
################################################################    


sphase = sail.phase(Tmodes.LGL3,SolCru,512)
sphase.addEqualCon(PhaseRegs.Front,RendezvousConstraint(ITab,range(0,6)),range(0,7))
sphase.setControlMode(Cmodes.BlockConstant)
sphase.setStaticParams([0.9])
MaxAlpha = 20.0
CosMaxAlpha =np.cos(np.deg2rad(MaxAlpha))
sphase.addLowerFuncBound(PhaseRegs.Path,CosAlpha(),[0,1,2,7,8,9],CosMaxAlpha,10.0)
sphase.addLUNormBound(PhaseRegs.Path,[7,8,9],.75,1.5)

sphase.addValueObjective(PhaseRegs.Back,6,1.0)
#sphase.addIntegralObjective(-(CosAlpha()**2),[0,1,2,7,8,9])
#sphase.addUpperDeltaTimeBound(3.2,1.0)

sphase.addEqualCon(PhaseRegs.Back,RendezvousConstraint(OTab,range(0,6)),range(0,6),[],[0])
sphase.Threads=16
sphase.optimizer.incrH = 8
sphase.optimizer.decrH = .33
sphase.optimizer.deltaH = 1.0e-6
sphase.optimizer.OptLSMode = ast.Solvers.LineSearchModes.L1
sphase.optimizer.PrintLevel = 1




#sphase.optimizer.BoundFraction = .995


sphase.solve_optimize()
SolCru=sphase.returnTraj()

for S in SolCru:
    S[7:10]*=.9


Tab =oc.LGLInterpTable(sail,6,3,Tmodes.LGL3,SolCru,len(SolCru))


SailModel = SolarSail(beta*.99999,False)
sail = EPPR_SolarSail(EFrame,SailModel = SailModel)

sinteg =sail.integrator(c.pi/7000,Tab,[7,8,9])
sinteg.Adaptive=True
#sinteg.ModifyInitialState=False
Test1 = np.copy(SolCru[0])
Test1[7:10] = [1,0,0]

SolCru2 = sinteg.integrate_dense(Test1,SolCru[-1][6])

print(SolCru2[-1][6]*EFrame.tstar/c.day - 3.5)
print(SolCru2[0][6]*EFrame.tstar/c.day - 3.5)

#############################################################################





es =[]
ts=[]
for S in SolCru:
    X = [S[0],S[1],S[2],S[7],S[8],S[9]]
    ts.append(S[6])
    es.append(np.arccos(CosAlpha().compute(X))*c.rtd)

plt.plot(ts,es)
plt.show()

plot= CRPlot(EFrame)
plot.addTraj(Imap,"IMAP-Nominal",'k')
plot.addTraj(IT,"IMAP-NominalR",'b')
plot.addTraj(SORB,"iig",'k')
plot.addTraj(ImapNom3,"IMAP-Nominale",'r')


plot.addTraj(SolCru,"SolCru",'r')
plot.addTraj(SolCru2,"SolCru2",'pink')

plot.addPoint(sail.SubL1,"Sub-L1",'purple',marker='.')
plot.addPoint(OTab.Interpolate(sphase.returnStaticParams()[0])[0:3],"Sub-L1",'purple',marker='.')

plot.Plot3d(pois=['L1','P2','L2'],bbox='L1P2',legend=False)


ers=[]
ts=[]
for T in Imap:
    t= T[6]
    X=T[0:3]
    ers.append(np.linalg.norm(ITab.Interpolate(t)[0:3]-X)*EFrame.lstar/1000.0)
    ts.append(t*EFrame.tstar/c.day)
    
plt.plot(ts,ers)
plt.ylabel("Error Km")
plt.xlabel("Days")

plt.show()  
    




