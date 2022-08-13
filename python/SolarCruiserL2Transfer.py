import asset as ast
import numpy as np

import MKgSecConstants as c
import Date as dt
from CR3BPModels import CR3BPFrame,CR3BP,CR3BP_SolarSail,CR3BP_SolarSail_ZeroAlpha
from EPPRModels import EPPRFrame,EPPR
from CRPlot import CRPlot,plt,colpal

#dt.datetime_to_jd(2025,1,1)
norm = np.linalg.norm
def normalize(x): return np.copy(x)/norm(x)
    
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags


JDImap = 2460585.02936299
JD0 = JDImap - 3.5
JDF = JD0 + 4.0*365.0   

Rearth  = 6378.136*1000.0
J2earth = .001082629
     
N = 3000

Frame = EPPRFrame("SUN",c.MuSun,"EARTH",c.MuEarth,c.AU,JD0,JDF,N=N)
 
Frame.AddSpiceBody("MOON",c.MuMoon)
Frame.AddSpiceBody("JUPITER BARYCENTER",c.MuJupiterBarycenter)
Frame.AddSpiceBody("VENUS",c.MuVenus)
Frame.AddSpiceBody("SATURN BARYCENTER",c.MuSaturnBarycenter)
#Frame.AddSpiceBody("MARS BARYCENTER",c.MuMars)
#Frame.AddSpiceBody("URANUS BARYCENTER",c.MuUranusBarycenter)
#Frame.AddSpiceBody("NEPTUNE BARYCENTER",c.MuNeptune)
Frame.Add_P2_J2Effect(J2earth,Rearth)
Frame.Test("MOON",500)

input("s")

eppr = EPPR(Frame,Enable_J2=True)
sail = CR3BP_SolarSail_ZeroAlpha()

LTAcc = (0.4*(1.0e-3)*4.0*np.cos(20.0*c.dtr)/114.0)/7.5
LTAcc/= Frame.astar
args = Args(7)
ode = oc.ode_6.ode(eppr.vf() - (args.segment3(3).normalized()*LTAcc).padded_upper(3))

print("s")
integ = eppr.integrator(c.pi/50000)
integ.MaxStepSize*=100
integ.Adaptive=True

sinteg = sail.integrator(c.pi/50000)
sinteg.MaxStepSize*=100
sinteg.Adaptive=True


Imap  = Frame.Copernicus_to_EPPR("COP.csv",center="P2")

IG = Imap[0]
ImapNom = integ.integrate_dense(IG,Imap[-1][6],5000)


integ2 = oc.ode_6.integrator(ode,c.pi/50000)
integ2.MaxStepSize*=100
integ2.Adaptive=True

Vs = np.linspace(-1.2,-1.45,30)
Trajs =[]
KTrajs = []
TF =3.1
DT = 1.80
for V in Vs:
    IGC = np.copy(IG)
    t = IGC[6]
    IGC[3:6]+= normalize(IGC[3:6])*V*Frame.M_S(t)
    Trajs.append(integ.integrate_dense(IGC,TF))
    KTrajs.append(sinteg.integrate_dense(Trajs[-1][-1],TF+DT))


IGF = np.copy(IG)
IGF = np.copy(IG)
t0 = IGF[6]
tf = 5.20
V = -2.33

IGF[3:6]+= normalize(IGF[3:6])*V*Frame.M_S(t0)
L2Traj = integ.integrate_dense(IGF,tf)

TranTab = oc.LGLInterpTable(6,L2Traj,len(L2Traj)*3)


t0s = [3.81]
tfs = 6.5
tfs =5.9
STrajs=[]
for t0 in t0s:
    SIG = TranTab.Interpolate(t0)
    STrajs.append(sinteg.integrate_dense(SIG,tfs))



SORB  = sail.GenSubL2Lissajous(.0015,.0,0,0,2,600,0)

ophase = sail.phase(Tmodes.LGL3)
ophase.setTraj(SORB,600)
ophase.addBoundaryValue(PhaseRegs.Front,[1,3,5,6],[0,0,0,0])
ophase.addBoundaryValue(PhaseRegs.Front,[2],[-.00089])

ophase.addBoundaryValue(PhaseRegs.Back,[1,3,5],[0,0,0])
ophase.optimizer.EContol = 1.0e-12

ophase.solve()
SORB = ophase.returnTraj()



def JC(X):
    beta = sail.SailModel.Normalbeta
    r = norm(X[0:3] - sail.P2)
    d = norm(X[0:3] - sail.P1)
    x = X[0]
    y = X[1]
    V = norm(X[3:6])
    
    J = (x**2 + y**2)/2.0 + (1-sail.mu)*(1-beta)/d + sail.mu/r - V*V/2
    return J*2


    
    
Dat = np.array([[X[6],JC(X)] for X in SORB]).T
plt.plot(Dat[0],Dat[1])
cols  = colpal("plasma",len(Trajs))
for i,Traj in enumerate(Trajs):
    Dat = np.array([[X[6],JC(X)] for X in Traj]).T
    plt.plot(Dat[0],Dat[1],color=cols[i])

plt.show()




OTab = oc.LGLInterpTable(6,SORB,len(SORB)*3)
OTab.makePeriodic()

SIGA = STrajs[0]

SailIG = []
for S in SIGA:
    X = np.zeros((10))
    X[0:7] = S[0:7]
    X[7] =.99
    SailIG.append(X)
    

##################################################
sailode = CR3BP_SolarSail()
sphase = sailode.phase(Tmodes.LGL7)
sphase.setTraj(SailIG,100)
#sphase.setControlMode(oc.ControlModes.HighestOrderSpline)

def RendFun(tab):
    args = Args(7)
    x = args.head(6)
    t = args[6]
    fun = oc.InterpFunction(tab,range(0,6)).vf()
    return fun.eval(t) - x

sphase.addEqualCon(PhaseRegs.Front,RendFun(TranTab),range(0,7))
sphase.addLUVarBound(PhaseRegs.Front,6,3.5,4.5,1.0)
sphase.addLUVarBound(PhaseRegs.Back,6,5.3,6.7,1.0)

def CosAlpha():
    args = Args(6)
    rhat = args.head3().normalized()
    nhat = args.tail3().normalized()
    return vf.dot(rhat,nhat)

MaxAlpha = 15.0
CosMaxAlpha =np.cos(np.deg2rad(MaxAlpha))
sphase.addLowerFuncBound(PhaseRegs.Path,CosAlpha(),[0,1,2,7,8,9],CosMaxAlpha,10.0)

sphase.addLUNormBound(PhaseRegs.Path,[7,8,9],.8,1.2)


InsIG = 1.3
sphase.setStaticParams([InsIG])
sphase.addEqualCon(PhaseRegs.Back, RendFun(OTab),range(0,6),[],[0])
#sphase.addDeltaTimeObjective(0.001)
#sphase.addIntegralObjective(-.01*(CosAlpha()),[0,1,2,7,8,9])
sphase.addValueObjective(PhaseRegs.Back,6,.0001)

sphase.optimizer.incrH = 8
sphase.optimizer.decrH = .1
sphase.optimizer.deltaH = 1.0e-7
sphase.optimizer.OptLSMode = ast.LineSearchModes.L1
sphase.optimizer.MaxLSIters =1
sphase.optimizer.BoundFraction = .99
sphase.optimizer.EContol = 1.0e-9
sphase.optimizer.MaxIters = 75

#sphase.solve()
sphase.optimize()
sphase.solve()

SailTraj = sphase.returnTraj()
InsT = sphase.returnStaticParams()[0]



ts =[]
ca =[]
for S in SailTraj:
    ts.append(S[6])
    CA = np.dot(normalize(S[0:3]),normalize(S[7:10]))
    ca.append(np.rad2deg(np.arccos(CA)))


    
plt.plot(ts,ca)
plt.show()

plot= CRPlot(Frame)

plot.addTraj(Imap,"IMAP-Nominal",'k')
#plot.addTraj(ImapNom,"ImapA",'k')

L2Traj = integ.integrate_dense(IGF,SailTraj[0][6])
SORB   =  sinteg.integrate_dense(SailTraj[-1][0:7],SailTraj[-1][6] + 2.5)

tf = SailTraj[-1][6]*Frame.tstar/c.day
plot.addTraj(L2Traj,"IMAP-Underburn",'b')
plot.addTraj(SailTraj,"Sail-Transfer",'purple')

plot.addTrajSeq(KTrajs)
plot.addTraj(SORB,"Sub-L2 Halo Orbit",'r')

plot.addTraj(Frame.AltBodyGTables["MOON"].InterpRange(1000,0.0,6.0),"Moon",'grey')
plot.addPoint(sail.SubL2,"Sub-L2",'red',marker ='.')
plot.addPoint(OTab.Interpolate(InsT),"Halo Insertion, TOF:" + str(int(tf)) +"Days",'gold')

#plot.addSphere("g","Earth",Frame.P2,r = Frame.P2_Rad)
plot.Plot3d(pois=['L1','P2','L2'],bbox='L1P2L2',legend=False)




