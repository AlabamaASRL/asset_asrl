import asset as ast
import numpy as np

import MKgSecConstants as c
import Date as dt
from CR3BPModels import CR3BPFrame,CR3BP,CR3BP_SolarSail,CR3BP_SolarSail_ZeroAlpha
from EPPRModels import EPPRFrame,EPPR,EPPR_LT
from CRPlot import CRPlot,plt

#dt.datetime_to_jd(2025,1,1)
norm = np.linalg.norm
def normalize(x): return np.copy(x)/norm(x)
    
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags

def LTTrade(Frame,LTAcc,TrajIG,IState,Tab,N = 1100):
    
    ode = EPPR_LT(Frame,LTacc=LTAcc, Enable_J2 = True)
    print(ode.NDLTacc)
    phase = ode.phase(Tmodes.LGL5)
    phase.setTraj(TrajIG,N)
    phase.addBoundaryValue(PhaseRegs.Front,range(0,7),IState[0:7])
    phase.addLUNormBound(PhaseRegs.Path,[7,8,9],.0000001,1.0,10.0)
    #phase.setControlMode(Cmodes.BlockConstant)
    def RendFun(tab):
        args = Args(4)
        x = args.head(3)
        t = args[3]
        fun = oc.InterpFunction(tab,range(0,3)).vf()
        return (fun.eval(t) - x)
    def VinfFunc(tab):
        args = Args(4)
        v = args.head(3)
        t = args[3]
        fun = oc.InterpFunction(tab,[3,4,5]).vf()
        dV = fun.eval(t) - v
        return vf.dot(dV,dV)
    
    #phase.setStaticParams([TrajIG[-1][6]])
    phase.addEqualCon(PhaseRegs.Back,RendFun(Tab),[0,1,2,6])
    #phase.addStateObjective(oc.StateObjective(VinfFunc(Tab),PhaseRegs.Back,[3,4,5,6]))
    phase.addBoundaryValue(PhaseRegs.Back,[6],[TrajIG[-1][6]])

    phase.addUpperFuncBound(PhaseRegs.Back,VinfFunc(Tab), [3,4,5,6],0.1**2/Frame.vstar**2,10.0)

    phase.Threads = 18
    phase.optimizer.QPThreads =6
    #phase.addLowerDeltaTimeBound(.01,1.0)
    phase.addIntegralObjective(.1*(Args(3).norm()**6),[7,8,9])
    #phase.addDeltaTimeObjective(1.0)
    phase.optimizer.OptLSMode = ast.LineSearchModes.L1
    phase.optimizer.MaxLSIters =2
    phase.optimizer.BoundFraction = .993
    phase.optimizer.KKTtol  = 1.0e-6
    phase.optimizer.Bartol  = 1.0e-2
    phase.optimizer.EContol = 1.0e-8
    phase.optimizer.BoundPush = 1.0e-3
    phase.optimizer.deltaH = 1.0e-6
    #phase.optimizer.OptBarMode = ast.BarrierModes.PROBE

    phase.optimizer.incrH = 8
    phase.optimizer.decrH = .1



    #phase.solve()
    phase.optimize()
    
    T= phase.returnTraj()
    print(T[0][0:6]-IState[0:6])
    print(Tab.Interpolate(T[-1][6])[0:6]-T[-1][0:6])

    return T




JDImap = 2460585.02936299
JD0 = JDImap - 3.5
JDF = JD0 + 3.0*365.0   

Rearth  = 6378.136*1000.0
J2earth = .001082629
     
N = 2000

Frame = EPPRFrame("SUN",c.MuSun,"EARTH",c.MuEarth,c.AU,JD0,JDF,N=N)
Frame.AddSpiceBody("MOON",c.MuMoon)
Frame.AddSpiceBody("JUPITER BARYCENTER",c.MuJupiterBarycenter)
Frame.AddSpiceBody("VENUS",c.MuVenus)
#Frame.AddSpiceBody("SATURN BARYCENTER",c.MuSaturnBarycenter)
#Frame.AddSpiceBody("MARS BARYCENTER",c.MuMars)
Frame.Add_P2_J2Effect(J2earth,Rearth)


eppr = EPPR(Frame,Enable_J2=True)
LTAcc = (0.4*(1.0e-3)*4.0*np.cos(20.0*c.dtr)/114.0)/1.2




print("s")
integ = eppr.integrator(c.pi/50000)
integ.MaxStepSize*=100
integ.Adaptive=True




Imap  = Frame.Copernicus_to_EPPR("COP.csv",center="P2")
IG = Imap[0]



IGF = np.copy(IG)
IGF = np.copy(IG)
t0 = IGF[6]
tf = 4.4
V = -2.32

IGF[3:6]+= normalize(IGF[3:6])*V*Frame.M_S(t0)
L2Traj = integ.integrate_dense(IGF,tf,50000)
#TranTab = oc.LGLInterpTable(6,L2Traj,len(L2Traj)*3)
TranTab = oc.LGLInterpTable(eppr.vf(),6,0,Tmodes.LGL3,L2Traj,len(L2Traj)*2)


VDisps = np.linspace(-2.0,-2.6,9)
LTStart = 1.5*c.day/Frame.tstar
Tks=[]
Istates =[]
for V in VDisps:
    IGC = np.copy(IG)
    t = IGC[6]
    IGC[3:6]+= normalize(IGC[3:6])*V*Frame.M_S(t)
    Istates.append(integ.integrate_dense(IGC,t+LTStart)[-1])
    Tks.append(integ.integrate_dense(IGC,tf))
    
     

from DerivChecker import FDDerivChecker
FDDerivChecker(eppr.vf(),Istates[0])

input("s")    


TOFIG = 2.9

BtrajIG = TranTab.InterpRange(1000,Istates[0][6],Istates[0][6]+TOFIG)
LTIG  =[]

for T in BtrajIG:
    X = np.zeros((10))
    X[0:7] = T[0:7]
    X[7:10] =normalize(X[0:3])*.01
    LTIG.append(X)
    
    
LTTrajs = []

for i,I in enumerate(Istates):
    TrajLT = LTTrade(Frame,LTAcc,LTIG,I,TranTab)
    
    ode = EPPR_LT(Frame,LTacc=LTAcc, Enable_J2 = True)
    TTab = oc.LGLInterpTable(ode.vf(),6,3,Tmodes.LGL3,TrajLT,len(TrajLT)*2)

    intt = ode.integrator(c.pi/38000,oc.InterpFunction(TTab,[7,8,9]),[6])
    
    #Tks[i] = integ.integrate_dense(TrajLT[0][0:7],TrajLT[-1][6])
    LTIG = np.copy(TrajLT)
    LTTrajs.append(TrajLT)
    TT = np.array(TrajLT).T
    plt.plot(TT[6],(TT[7]**2 + TT[8]**2 + TT[9]**2)**.5)
    
plt.show()

plot= CRPlot(Frame)

plot.addTraj(L2Traj,"IMAP-Divert",'k')
plot.addTrajSeq(LTTrajs)
#plot.addTrajSeq(Tks)

plot.addPoint(Istates[0],"Sub-L2",'red',marker ='.')

plot.Plot2d(pois=['L1','P2','L2'],bbox='L1P2L2',legend=False)




