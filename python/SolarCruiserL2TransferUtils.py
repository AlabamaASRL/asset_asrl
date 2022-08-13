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
Frame.Add_P2_J2Effect(J2earth,Rearth)

eppr = EPPR(Frame,Enable_J2=True)
sail = CR3BP_SolarSail_ZeroAlpha()

integ = eppr.integrator(c.pi/50000)
integ.MaxStepSize*=100
integ.Adaptive=True

sinteg = sail.integrator(c.pi/50000)
sinteg.MaxStepSize*=100
sinteg.Adaptive=True


Imap  = Frame.Copernicus_to_EPPR("COP.csv",center="P2")
IG = Imap[0]


def TransferOpt(ode,TrajIG,InsIG,IGTable,Orbtable,N=500,Solve=True):
    sphase = ode.phase(Tmodes.LGL7)
    sphase.setTraj(TrajIG,N)
    #sphase.setControlMode(oc.ControlModes.HighestOrderSpline)
    
    def RendFun(tab):
        args = Args(7)
        x = args.head(6)
        t = args[6]
        fun = oc.InterpFunction(tab,range(0,6)).vf()
        return fun.eval(t) - x
    
    sphase.addEqualCon(PhaseRegs.Front,RendFun(IGTable),range(0,7))
    sphase.addLUVarBound(PhaseRegs.Front,6,3.5,4.5,1.0)
    sphase.addLUVarBound(PhaseRegs.Back,6,5.3,6.7,1.0)
    
    def CosAlpha():
        args = Args(6)
        rhat = args.head3().normalized()
        nhat = args.tail3().normalized()
        return vf.dot(rhat,nhat)
    
    MaxAlpha = 85.0
    CosMaxAlpha =np.cos(np.deg2rad(MaxAlpha))
    sphase.addLowerFuncBound(PhaseRegs.Path,CosAlpha(),[0,1,2,7,8,9],CosMaxAlpha,10.0)
    sphase.addLUNormBound(PhaseRegs.Path,[7,8,9],.8,1.2)
    
    
    sphase.setStaticParams([InsIG])
    sphase.addEqualCon(PhaseRegs.Back, RendFun(Orbtable),range(0,6),[],[0])
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
    if(Solve==True):sphase.optimize()
    
    SailTraj = sphase.returnTraj()
    
    return SailTraj,sphase.returnStaticParams()[0]

def TransferType1(IG,VdI,t0s,tfs,OrbitAmp,InsIG,N=500,Solve=True):
    IGF = np.copy(IG)
    t0 = IGF[6]
    tf = 8.00
    V = VdI
    
    IGF[3:6]+= normalize(IGF[3:6])*V*Frame.M_S(t0)
    L2Traj = integ.integrate_dense(IGF,tf)
    TranTab = oc.LGLInterpTable(6,L2Traj,len(L2Traj)*3)
    
    
    SIG = TranTab.Interpolate(t0s)
    STraj=sinteg.integrate_dense(SIG,tfs)
    SailIG = []
    for S in STraj:
        X = np.zeros((10))
        X[0:7] = S[0:7]
        X[7] =.99
        SailIG.append(X)
    
    
    SORB  = sail.GenSubL2Lissajous(.0015,.0,0,0,2,600,0)
    ophase = sail.phase(Tmodes.LGL3)
    ophase.setTraj(SORB,600)
    ophase.addBoundaryValue(PhaseRegs.Front,[1,3,5,6],[0,0,0,0])
    ophase.addBoundaryValue(PhaseRegs.Front,[2],[OrbitAmp])
    ophase.addBoundaryValue(PhaseRegs.Back,[1,3,5],[0,0,0])
    ophase.optimizer.EContol = 1.0e-12
    ophase.solve()
    SORB = ophase.returnTraj()
    OTab = oc.LGLInterpTable(6,SORB,len(SORB)*3)
    OTab.makePeriodic()
    
    ode = CR3BP_SolarSail()

    STraj,Ins = TransferOpt(ode,SailIG,InsIG,TranTab,OTab,N=N,Solve=Solve)
    
    BTraj = integ.integrate_dense(IGF,STraj[0][6])

    
    return BTraj,STraj,OTab,SORB,Ins


BTraj,SailTraj,OTab,SORB,Ins = TransferType1(IG,VdI=-2.3,t0s=4.9,tfs=6.4,OrbitAmp=-.00089,InsIG=1.3,Solve=True)
BTraj,SailTraj,OTab,SORB,Ins = TransferType1(IG,VdI=-2.33,t0s=3.81,tfs=5.9,OrbitAmp=-.00089,InsIG=1.3,Solve=True)





plot= CRPlot(Frame)

plot.addTraj(Imap,"IMAP-Nominal",'k')
tf = SailTraj[-1][6]*Frame.tstar/c.day
plot.addTraj(BTraj,"IMAP-Underburn",'b')
#plot.addTraj(BTraj1,"IMAP-Underburn1",'cyan')


plot.addTraj(SailTraj,"Sail-Transfer",'purple')
#plot.addTraj(SailTraj1,"Sail-Transfer1",'g')

plot.addTraj(SORB,"Sub-L2 Halo Orbit",'r')
plot.addTraj(Frame.AltBodyGTables["MOON"].InterpRange(1000,0.0,6.0),"Moon",'grey')
plot.addPoint(sail.SubL2,"Sub-L2",'red',marker ='.')
plot.addPoint(OTab.Interpolate(Ins),"Halo Insertion, TOF:" + str(int(tf)) +"Days",'gold')
plot.Plot3d(pois=['L1','P2','L2'],bbox='L1P2L2',legend=True)
plot.Plot2d(pois=['L1','P2','L2'],bbox='L1P2L2',legend=True)

