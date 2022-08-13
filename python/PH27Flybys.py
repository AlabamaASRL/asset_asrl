import asset as ast
import spiceypy as spice
import matplotlib.pyplot as plt
import matplotlib.axes as axes
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
spice.furnsh("BasicKernel.txt")

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

def GetEphemTraj(body,startDay,endDay,numstep,LU=1.0,TU=1.0,\
    Frame='ECLIPJ2000',Center='SOLAR SYSTEM BARYCENTER'):
    startET = spice.str2et(startDay)
    endET   = spice.str2et(endDay)
    times = [startET + (endET - startET)*x/numstep for x in range(numstep)]
    
    states=[]
    t0 = times[0]
    for t in times:
        X = np.zeros((7))
        X[0:6] = spice.spkezr(body,t,Frame,'NONE',Center)[0]
        X[0:3] *= 1000.0/LU # Convert to m
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
    
    BodyV =  oc.InterpFunction(tab,range(3,6)).vf()
    vInfPlus=(BodyV.eval(t0)-v0).norm()
    vInfMinus=(BodyV.eval(t1)-v1).norm()
    return (vInfPlus-vInfMinus)
    
def RendCon(tab):
    XT = Args(7)
    x = XT.head(6)
    t = XT[6]
    fun = oc.InterpFunction(tab,range(0,6)).vf()
    return fun.eval(t) - x

def PosCon(tab):
    XT = Args(4)
    x = XT.head(3)
    t = XT[3]
    fun = oc.InterpFunction(tab,range(0,3)).vf()
    return fun.eval(t) - x
def VinfFunc(tab):
    XT = Args(4)
    x = XT.head(3)
    t = XT[3]
    fun = oc.InterpFunction(tab,range(3,6)).vf()
    return fun.eval(t) - x

def FlybyAngleBound(tab,mubod,minrad):
    X = Args(8)
    v0 = X.head3()
    t0 = X[3]
    
    v1 = X.tail(4).head3()
    t1 = X.tail(4)[3]
    
    BodyV =  oc.InterpFunction(tab,range(3,6)).vf()
    v0dir =(BodyV.eval(t0)-v0).normalized()
    v1dir =(BodyV.eval(t1)-v1).normalized()
    
    vInf2  =(BodyV.eval(t0)-v0).squared_norm()
    
    delta      = vf.arccos(vf.dot(v0dir,v1dir))
    deltaMax   = np.pi - 2*vf.arccos(1.0/(1.0 + minrad*vInf2/mubod)) 
    return   (delta - deltaMax)


def FlybyAngleBoundTest(tab,mubod,minrad):
    X = Args(8)
    v0 = X.head3()
    t0 = X[3]
    
    v1 = X.tail(4).head3()
    t1 = X.tail(4)[3]
    
    BodyV =  oc.InterpFunction(tab,range(3,6)).vf()
    v0dir =(BodyV.eval(t0)-v0).normalized()
    v1dir =(BodyV.eval(t1)-v1).normalized()
    
    vInf2  =(BodyV.eval(t0)-v0).squared_norm()
    
    delta      = vf.arccos(vf.dot(v0dir,v1dir))
    deltaMax   = np.pi - 2*vf.arccos(mubod/(mubod + minrad*vInf2)) 
    return   (deltaMax)

def DoEverything(ODE, IGs, BodyData, StartTable, EndTable, DepVinf, NumSegs = 256):
    ocp = oc.OptimalControlProblem()
    for i,IG in enumerate(IGs):
        ocp.addPhase(ODE.phase(TModes.LGL3, IG, NumSegs))
        ocp.Phase(-1).addLUNormBound(PhaseRegs.Path,[7,8,9],.01,1.0)
        
        
    ocp.addForwardLinkEqualCon(0, len(IGs)-1, [0,1,2,6])
    ocp.Phase(0).addLowerVarBound(PhaseRegs.Front,6,0.0,1.0)
    ocp.Phase(0).addEqualCon(PhaseRegs.Front, PosCon(StartTable),[0,1,2,6]) # Start at Earth
    
    ocp.Phase(0).addUpperFuncBound(PhaseRegs.Front,VinfFunc(StartTable).squared_norm(),
                              [3,4,5,6],(DepVinf/vstar)**2,1.0)
    
    #ocp.Phase(-1).addEqualCon(PhaseRegs.Back, PosCon(EndTable),[0,1,2,6]) # Start at Earth
    #ocp.Phase(-1).addStateObjective(PhaseRegs.Back, VinfFunc(EndTable).squared_norm()*10.0,[3,4,5,6]) # Start at Earth
    
    def Inc():
        X = Args(6)
        h = X.head3().cross(X.tail3()).normalized()
        return h[2]**2
    ocp.Phase(-1).addStateObjective(PhaseRegs.Back,Inc(),[0,1,2,3,4,5]) # Start at Earth
    ocp.Phase(0).addUpperVarBound(PhaseRegs.Back,6,ocp.Phase(-1).returnTraj()[-1][6]*1.05,1.0)

    print(len(IGs),len(BodyData))
    for i,Data in enumerate(BodyData):
        Table = Data[0]
        MuValue = Data[2]
        BodyRad = Data[1]
        
        ocp.Phase(i).addEqualCon(PhaseRegs.Back, PosCon(Table),[0,1,2,6]) # Finish at Venus
        ocp.addLinkEqualCon(VinfMatchCon(Table),oc.LinkFlags.BackToFront,[[i,i + 1]],[3,4,5,6])
        ocp.addLinkInequalCon(FlybyAngleBound(Table,MuValue,BodyRad),oc.LinkFlags.BackToFront,[[i,i+1]],[3,4,5,6])
        
    ocp.optimizer.OptLSMode =ast.Solvers.LineSearchModes.L1
    
    #ocp.transcribe(True,True)
    ocp.solve_optimize()
    
    for i,Data in enumerate(BodyData):
        T1 = ocp.Phase(i).returnTraj()
        T2 = ocp.Phase(i+1).returnTraj()
        Table = Data[0]
        MuValue = Data[2]
        BodyRad = Data[1]
        
        X = np.zeros((8))
        X[0:4]= T1[-1][3:7]
        X[4:8]= T2[0][3:7]
        F = FlybyAngleBoundTest(Table,MuValue,BodyRad)
        F2 = FlybyAngleBound(Table,MuValue,BodyRad)
        
        
        
        
    return [ocp.Phase(i).returnTraj() for i in range(len(IGs))]
        
    

    

if __name__ == "__main__":
 
    lstar     = c.AU
    vstar     = np.sqrt(c.MuSun/lstar)
    tstar     = np.sqrt(lstar**3/c.MuSun)
    astar     = c.MuSun/lstar**2
      
    engacc = (1./1800)/astar
    
    print(engacc)
    
    LoadIG = np.load("IG.npy", allow_pickle = True)
    startdayJD = LoadIG[0]

    YEAR = 365*24*3600/tstar
    
    data_start    = 'Nov 8, 2030'
    data_end      = 'Jan 1, 2049'
    datapoints = 15000
    
    EarthDat  = GetEphemTraj("EARTH",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    VenusDat   = GetEphemTraj("Venus",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    PH27Dat = GetEphemTraj("54186922",data_start,data_end,datapoints,LU=lstar,TU=tstar,Center="SUN")
    
    t0 = spice.utc2et(data_start)
    te = spice.utc2et(data_end)-t0
    
    EarthTab  = oc.LGLInterpTable(6,EarthDat,datapoints)
    VenusTab   = oc.LGLInterpTable(6,VenusDat,datapoints)
    PH27Tab = oc.LGLInterpTable(6,PH27Dat,datapoints)
    
    ED =  np.array(EarthDat).T
    VD =  np.array(VenusDat).T
    PD =  np.array(PH27Dat).T
    
    
    
    
    Trajs =[]
    
    for i, traj in enumerate(LoadIG[1:]):
        Traj=[]
        for j, state in enumerate(traj):
            X= np.zeros((10))
            X[0:7]=state
            X[6]+=2*YEAR
            n = state[3:6]/np.linalg.norm(state[3:6])
            X[7:10]= n*.3
            Traj.append(X)
        Trajs.append(Traj)
        
    
    Stuff ={}
    Stuff["Earth"] = [EarthTab,(c.RadiusEarth + 300.0*1000.0)/lstar, c.MuEarth*(tstar**2)/(lstar**3)]
    Stuff["Venus"] = [VenusTab,(c.RadiusVenus + 300.0*1000.0)/lstar, c.MuVenus*(tstar**2)/(lstar**3)]

    Seq =["Venus", "Earth", "Venus", "Venus", "Venus", "Venus"]
    
    BodyData =[Stuff[S] for S in Seq]
    
    StartTable = EarthTab
    EndTable  = PH27Tab
    
    ode = LTModel(1,.015)

    Trajs2 = DoEverything(ode, Trajs, BodyData, StartTable, EndTable, 3000.0, NumSegs = 196)
    
    times = []
    for i,traj in enumerate(Trajs2):
        times.append(traj[-1][6])
    
    fig = plt.figure(figsize = (9, 9))
    axes = fig.add_subplot(projection = "3d")
    axes.plot(ED[0]*lstar/c.AU,ED[1]*lstar/c.AU,ED[2]*lstar/c.AU,color ='g')
    axes.plot(VD[0]*lstar/c.AU,VD[1]*lstar/c.AU,VD[2]*lstar/c.AU,color = 'r')
    axes.plot(PD[0]*lstar/c.AU,PD[1]*lstar/c.AU,PD[2]*lstar/c.AU,color = 'k')
    T = []
    
    
    fig2 = plt.figure(figsize = (9, 9))
    axes2 = fig2.add_subplot()
    
    for traj in Trajs2:
        TT = np.array(traj).T
        T.append(TT)
        axes.plot(TT[0], TT[1], TT[2])
        axes2.plot(TT[6],(TT[7]**2 +TT[8]**2+TT[9]**2 )**.5)
        
    for i,seq in enumerate(Seq):
        state = Stuff[seq][0].Interpolate(times[i])
        axes.scatter(state[0], state[1], state[2])

    axes.scatter(0,0,0,color= 'y')
    axes.set_xlabel('X (AU)')
    axes.set_ylabel('Y (AU)')
    axes.set(xlim=(-1, 1), ylim = (-1, 1), zlim = (-1, 1))
    #axes.set_aspect('equal')
    #axes.grid(True)
    
    
    plt.show()

    breakpoint()

   
   
   
   


   
   
    
    
    
