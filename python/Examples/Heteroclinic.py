import asset_asrl as ast
import numpy as np
import matplotlib.pyplot as plt

from asset_asrl.Astro.AstroModels import CR3BP
from asset_asrl.Astro.FramePlot import CRPlot

import asset_asrl.Astro.Constants as c
import asset as astt

vf   = ast.VectorFunctions
oc   = ast.OptimalControl
sol  = ast.Solvers
Args = vf.Arguments

def normalize(x):
    return np.array(x)/np.linalg.norm(x)

def JacobiFunc(mu):
    
    args = Args(6)
    
    r = args.head3()
    v = args.segment3(3)

    p1loc = np.array([-mu,0,0])
    p2loc = np.array([1.0-mu,0,0])
    
    gt1 = (r-p1loc).inverse_norm()*(1.0-mu)
    gt2 = (r-p2loc).inverse_norm()*(mu)
    
    return r.head2().squared_norm() + (gt1 + gt2)*2 - v.squared_norm()

    
def MakeOrbit(IG,C,n=100):
    ode = CR3BP(c.MuEarth,c.MuMoon,c.LD)
    
    phase = ode.phase("LGL5",IG,300)
    phase.addBoundaryValue("Front",[1,2,3,5,6],[0,0,0,0,0])
    phase.addBoundaryValue("Back",[1,3],[0,0])
    
    eq= JacobiFunc(ode.mu)-C
    phase.addEqualCon("Front",eq,range(0,6))
    phase.optimizer.EContol=1.0e-12
    phase.solve()
    
    return phase.returnTraj()
    

    
def GetManifold(OrbitIn,h,T,Stable=True):
    ode = CR3BP(c.MuEarth,c.MuMoon,c.LD)
    
    integ =ode.integrator("DOPRI87",.001)
    
    integ.setAbsTol(1.0e-13)
    Period = OrbitIn[-1][6]

    integ.vf()
    
    Orbit = integ.integrate_dense(OrbitIn[0],Period,400)
    
    times = [O[6]+Period for O in Orbit]
    Results = integ.integrate_stm_parallel(Orbit,times,16)
    

    EigIGs = []
   
    
    
    for i,Result in enumerate(Results):
        
        Xf,Jac = Result
        
        vals,vecs = np.linalg.eig(Jac[0:6,0:6])
        
        iv = 0
        if(not Stable):iv = 1
        
        Vec = np.real(vecs[0])
        
        Xp = np.copy(Orbit[i])
        Xp[6]=0
        Xp[0:3]+=normalize(Vec[0:3])*h
        
        EigIGs.append(Xp)
        
        Xp = np.copy(Orbit[i])
        Xp[6]=0
        Xp[0:3]-=normalize(Vec[0:3])*h
        
        
        Vec = np.real(vecs[1])
        
        Xp = np.copy(Orbit[i])
        Xp[6]=0
        Xp[0:3]+=normalize(Vec[0:3])*h
        
        EigIGs.append(Xp)
        
        Xp = np.copy(Orbit[i])
        Xp[6]=0
        Xp[0:3]-=normalize(Vec[0:3])*h
        
        EigIGs.append(Xp)
        
    
    def CrossMoon():
        X = Args(7)
        return X[0]-(1-ode.mu)
    
    def Cull():
        X = Args(7)
        alt = (X-[1-ode.mu,0,0]).head3().norm()-.015
        y = (X[1]-.15)*(X[1]+.15)
        return alt*y
    
    ts = np.ones((len(EigIGs)))*T
    if(not Stable):ts = -ts
    events = [(CrossMoon(),0,1),
              (Cull(),0,1)]
    print("S")
    import time
    
    t00=time.perf_counter()
    Results = integ.integrate_dense_parallel(EigIGs,ts,events,16)
    tff=time.perf_counter()
    
    print(tff-t00)
    
    t00=time.perf_counter()
    Results8 = integ.integrate_parallel(EigIGs,ts,events,16)
    tff=time.perf_counter()
    print(tff-t00)
    
    Manifolds=[]
    for Result in Results:
        Traj,eventlocs = Result
        if(len(eventlocs[0])==1 and len(eventlocs[1])==0):
            Traj.pop()
            Traj.append(eventlocs[0][0])
            Manifolds.append(Traj)
        
        
    
    
    return Manifolds
        
    
        
        
        
        
        
        
        
        
        
        
    
def GetBest(Orbs1,Orbs2):
    distij =[]
    for i in range(0,len(Orbs1)):
        for j in range(0,len(Orbs2)):
            dist = np.linalg.norm(Orbs1[i][-1][0:6]-Orbs2[j][-1][0:6])
            distij.append([dist,i,j])
    distij.sort(key = lambda x:x[0])
    
    return Orbs1[distij[0][1]],Orbs2[distij[0][2]]
            
import sys    

def Func(Traj):
    
    tab = oc.LGLInterpTable(6,Traj,10000)
    
    func = oc.InterpFunction(tab,range(0,6))
    
    #del(tab)
    
    return func
    
    
import gc   

if __name__ == "__main__":
    print(gc.get_stats())

    ode = CR3BP(c.MuEarth,c.MuMoon,c.LD)
    
    IGL1 = ode.GenL1Lissajous(.03,0,180,0,1,100)
    CL1 = MakeOrbit(IGL1,3.15)
    IGL2 = ode.GenL2Lissajous(.03,0,  0,0,1,100)
    CL2 = MakeOrbit(IGL2,3.15)
    
    MansL1 = GetManifold(CL1,1e-5,9.0)
    MansL2 = GetManifold(CL2,1e-5,12.0,False)
    
    func = Func(CL1)
    

    import time
    t00=time.perf_counter()
    O1,O2 = GetBest(MansL1,MansL2)
    tff=time.perf_counter()
    
    print(tff-t00)
    print(gc.get_stats())

    print(gc.collect())
    print(gc.get_stats())
    print(func.compute([CL1[20][6]]))

    
    plot = CRPlot(ode,'Earth','Moon','green','grey')
    
    #plot.addTraj(IGL1,"L1IG",'b',linestyle='dashed')
    plot.addTrajSeq(MansL1,header='L1',colp='plasma')
    plot.addTrajSeq(MansL2,header='L2',colp='viridis')

    plot.addTraj(CL1,"L1C",'b',linestyle='solid')
    plot.addTraj(O1,"O1",'b',linestyle='solid')

    plot.addTraj(CL2,"L2C",'r',linestyle='solid')
    plot.addTraj(O2,"O2",'r',linestyle='solid')

    

    fig = plt.figure()
    ax = plt.subplot()
    
    plot.Plot2dAx(ax,bbox="L1P2L2",pois=["L1","L2","P2"],plegend=True)
    ax.grid(True)
    plt.show()
    