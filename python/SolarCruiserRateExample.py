from AstroModels import EPPRFrame,EPPR,EPPR_SolarSail,SolarSail,EPPR_LT,LowThrustAcc,CR3BP,CR3BPFrame
from AstroConstraints import RendezvousConstraint,CosAlpha
from FramePlot import CRPlot,TBPlot,plt
import MKgSecConstants as c
import numpy as np
import asset as ast
from DerivChecker import FDDerivChecker



vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags



def normalized(x): return x/np.linalg.norm(x)

JDImap = 2460585.02936299
JD0 = JDImap - 5.5
JDF = JD0 + 3.0*365.0   
N = 9000

SpiceFrame = 'J2000'
EFrame = EPPRFrame("SUN",c.MuSun,"EARTH",c.MuEarth,c.AU,JD0,JDF,N=N,SpiceFrame=SpiceFrame)
Bodies = ["MOON","JUPITER BARYCENTER","VENUS","MARS BARYCENTER","SATURN BARYCENTER",
          "MERCURY","URANUS BARYCENTER","NEPTUNE BARYCENTER"]

#Bodies = ["MOON","JUPITER BARYCENTER","VENUS"]

EFrame.AddSpiceBodies(Bodies,N=4000)
EFrame.Add_P2_J2Effect()


eppr  = EPPR(EFrame,Enable_J2=True)


beta = 0.02
SailModel = SolarSail(beta,False)
sail = EPPR_SolarSail(EFrame,SailModel = SailModel)


epinteg =eppr.integrator(c.pi/90000)
epinteg.setAbsTol(1.0e-13)
epinteg.Adaptive=True

sinteg =sail.integrator(c.pi/7000,(Args(3).normalized() + Args(3).Constant([0,-.4,0])).normalized(),range(0,3))
sinteg.Adaptive=True
cr3bp = CR3BP(c.MuSun, c.MuEarth, c.AU)



Day = c.day/EFrame.tstar

Imap  = EFrame.Copernicus_to_Frame("COP.csv",center="P2")
IG = Imap[0]

Tdep = 100*Day
Ts = 150*Day
ImapNom = epinteg.integrate_dense(IG,IG[6] + Tdep)

ISTATE = Imap[1500]

State = epinteg.integrate(ISTATE,Imap[-1][6])

print((State[0:3]-Imap[-1][0:3])*EFrame.lstar/1000)

input("S")


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
ophase.optimizer.PrintLevel = 1
ophase.optimizer.CNRMode = True
ophase.optimizer.QPThreads = 1
ophase.Threads=1
ophase.solve()
SORB = ophase.returnTraj()
OTab = oc.LGLInterpTable(6,SORB,len(SORB)*3)
OTab.makePeriodic()



################################################################    


sphase = sail.phase(Tmodes.LGL3,SolCru,512)
sphase.addEqualCon(PhaseRegs.Front,RendezvousConstraint(ITab,range(0,6)),range(0,7))
#sphase.setControlMode(Cmodes.BlockConstant)
sphase.setStaticParams([1.9])
MaxAlpha = 19.0
CosMaxAlpha =np.cos(np.deg2rad(MaxAlpha))
sphase.addLowerFuncBound(PhaseRegs.Path,CosAlpha(),[0,1,2,7,8,9],CosMaxAlpha,10.0)


##############################################################################
####################### Critical Section !!###################################
maxslewrate = 4.0 # Deg/Day

def EPPRRateFunc(EFrame):
    tu1_tu2 = Args(8)
    t1 = tu1_tu2[0]
    u1 = tu1_tu2.segment3(1)
    t2 = tu1_tu2[4]
    u2 = tu1_tu2.tail3()
    
    h = t2 - t1
    tm = (t1+t2)/2.0   
    
    # Sense of normal vetor rotation in rotatiing frame
    omegahat = u1.cross(u2).normalized()
    # slew rate mag in rotating frame
    omegadot = vf.arccos(u1.normalized().dot(u2.normalized()))/h  
    
    # add in rotation of frame from EFrame to get real slew rate
    omega = (omegahat*omegadot) + EFrame.WFunc.eval(tm)  
    degpday = (180/np.pi)*(c.day/EFrame.tstar)              
    
    # Take norm
    #return tu1_tu2.norm()
    return omega.norm()*degpday


    

sphase.addUpperFuncBound(PhaseRegs.PairWisePath,EPPRRateFunc(EFrame),[6,7,8,9],maxslewrate,1.0)

################################################################################    


sphase.addLUNormBound(PhaseRegs.Path,[7,8,9],.75,1.5)

sphase.addValueObjective(PhaseRegs.Back,6,1.0)
#sphase.addStateObjective(PhaseRegs.Back,(Args(1)[0]*1.0)*1.0,[6])


sphase.addEqualCon(PhaseRegs.Back,RendezvousConstraint(OTab,range(0,6)),range(0,6),[],[0])
sphase.Threads=16
sphase.optimizer.incrH = 8
sphase.optimizer.decrH = .33
sphase.optimizer.deltaH = 1.0e-6
sphase.optimizer.CNRMode = False
sphase.optimizer.QPThreads = 1
#sphase.enable_vectorization(False)
sphase.optimizer.OptLSMode = ast.Solvers.LineSearchModes.L1
sphase.optimizer.PrintLevel = 0
#sphase.solve_optimize()



SolCru = sphase.returnTraj()
#############################################################################



fig,axs = plt.subplots(2,1)


es =[]
ts=[]
for S in SolCru:
    X = [S[0],S[1],S[2],S[7],S[8],S[9]]
    ts.append(S[6]*EFrame.tstar/c.day)
    es.append(np.arccos(CosAlpha().compute(X))*c.rtd)

RateFunc =  EPPRRateFunc(EFrame)
Rs=[]
tss =[]
for i in range(0,len(SolCru)-1):
    
    X = np.zeros((8))
    X[0:4] = SolCru[i][6:10]
    X[4:8] = SolCru[i+1][6:10]
    tss.append((X[0]/2+X[4]/2)*EFrame.tstar/c.day)
    Rs.append(RateFunc.compute(X)[0])
    

axs[0].plot(ts,es)
axs[0].set_ylabel("Incidence Angle")

axs[1].plot(tss,Rs)
axs[1].set_xlabel("t(Day)")
axs[1].set_ylabel("Slew Rate (Deg/day)")

plt.show()

plot= CRPlot(EFrame)
plot.addTraj(IT,"IMAP-Nominal",'b')
plot.addTraj(SORB,"Orbit",'k')


plot.addTraj(SolCru,"SolCru",'r')
plot.addPoint(ISTATE,"Fd",'red',marker='.')

plot.addPoint(State,"F",'red',marker='.')

plot.addPoint(sail.SubL1,"Sub-L1",'purple',marker='.')
plot.addPoint(OTab.Interpolate(sphase.returnStaticParams()[0])[0:3],"Sub-L1",'purple',marker='.')

plot.Plot3d(pois=['L1','P2','L2'],bbox='L1P2',legend=False)







