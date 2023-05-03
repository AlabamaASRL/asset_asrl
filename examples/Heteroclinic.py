import asset_asrl as ast
import numpy as np
import matplotlib.pyplot as plt

from asset_asrl.Astro.AstroModels import CR3BP
from asset_asrl.Astro.FramePlot import CRPlot

import asset_asrl.Astro.Constants as c

'''
Find a heteroclinic connection between two orbit families in the EM-CR3BP.

Orbits derived from Amanda Haapala's dissertation

https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Dissertations/2014_Haapala.pdf

'''


vf   = ast.VectorFunctions
oc   = ast.OptimalControl
sol  = ast.Solvers
Args = vf.Arguments

def normalize(x):
    return np.array(x)/np.linalg.norm(x)



def JacobiFunc(mu):
    r,v = Args(6).tolist([(0,3),(3,3)])
    
    p1loc = np.array([-mu,0,0])
    p2loc = np.array([1.0-mu,0,0])
    
    gt1 = (r-p1loc).inverse_norm()*(1.0-mu)
    gt2 = (r-p2loc).inverse_norm()*(mu)

    return r.head2().squared_norm() + (gt1 + gt2)*2 - v.squared_norm()

    
def MakeOrbit(ode, OrbitIG,Jconst,nsegs=100):
    phase = ode.phase("LGL5",OrbitIG,nsegs)
    
    #Enforce planar perpendicular crossing
    phase.addBoundaryValue("First",[1,2,3,5,6],[0,0,0,0,0])
    phase.addBoundaryValue("Last",[1,3],[0,0])

    # Specifying Jacobi constant, not amplitude of orbit
    phase.addEqualCon("First",JacobiFunc(ode.mu)-Jconst,range(0,6))
    phase.optimizer.set_EContol(1.0e-12)
    phase.optimizer.PrintLevel=1
    phase.solve()  # Solve the orbit
    
    return phase.returnTraj()
    

    
def GetManifold(ode,OrbitIn,dx,dt,nman=50,Stable=True):
    integ =ode.integrator("DOPRI87",.01)
    integ.setAbsTol(1.0e-13)
    
    Period = OrbitIn[-1][6]
    Orbit = integ.integrate_dense(OrbitIn[0],Period,nman)
    
    times = [O[6]+Period for O in Orbit]
    ncores = 16
    StmResults = integ.integrate_stm_parallel(Orbit,times,ncores)
    
    EigIGs = []
   
    for i,Result in enumerate(StmResults):
        
        Xf,Jac = Result
        
        vals,vecs = np.linalg.eig(Jac[0:6,0:6])
        vecs = vecs.T
        
        idxs = list(range(0,6))
        idxs.sort(key = lambda x:np.abs(vals[x]))
       
        if(Stable):
            Vec = np.real(vecs[idxs[0]])
        else:
            Vec = np.real(vecs[idxs[-1]])

        
        Xp = np.copy(Orbit[i])
        Xp[0:3]+=normalize(Vec[0:3])*dx
        
        EigIGs.append(Xp)
        
        Xp = np.copy(Orbit[i])
        Xp[0:3]-=normalize(Vec[0:3])*dx
        
        EigIGs.append(Xp)
        
    if Stable:
        dt = -dt
    ts = [IG[6] + dt for IG in EigIGs]

    # Event for detecting crossing of Moon's X position
    def CrossMoon(ode):
        X = Args(7)
        return X[0]-(1-ode.mu)
    # Event for detecting departure from lunar SOI and close encounters
    def Cull(ode):
        X = Args(7)
        alt = (X.head3()-ode.P2).norm()-.015
        y = (X[1]-.15)*(X[1]+.15)
        return alt*y
    
    events = [(CrossMoon(ode),0,1),(Cull(ode),0,1)]
   
    Results = integ.integrate_dense_parallel(EigIGs,ts,events,ncores)
    
    Manifolds=[]
    for Result in Results:
        Traj,eventlocs = Result
        # Accept those that crossed moon and werent culled
        if(len(eventlocs[0])==1 and len(eventlocs[1])==0):
            Traj.pop()
            Traj.append(eventlocs[0][0])
            Manifolds.append(Traj)
        
    return Manifolds
        
    
        
    
def FindClosestConnection(Orbs1,Orbs2):
    distij =[]
    for i in range(0,len(Orbs1)):
        for j in range(0,len(Orbs2)):
            dist = np.linalg.norm(Orbs1[i][-1][0:6]-Orbs2[j][-1][0:6])
            distij.append([dist,i,j])
    distij.sort(key = lambda x:x[0])
    
    return Orbs1[distij[0][1]],Orbs2[distij[0][2]]
            
def MakeHeteroclinic(ode,Man1,Man2,L1Orbit,L2Orbit):
    OrbitTab1 = oc.LGLInterpTable(L1Orbit)
    OrbitTab1.makePeriodic()

    OrbitTab2 = oc.LGLInterpTable(L2Orbit)
    OrbitTab2.makePeriodic()
    
    #Enforces that position should lie somewhere along orbit
    def PosCon(OrbitTab):
        PosFunc = oc.InterpFunction(OrbitTab,range(0,3)).vf()
        Rt = Args(4)
        R = Rt.head(3)
        t = Rt[3]
        return R - PosFunc(t)
    
    #Objective is squared velocity difference with orbit
    def DVObj(OrbitTab):
        VelFunc = oc.InterpFunction(OrbitTab,range(3,6)).vf()
        Vt = Args(4)
        V = Vt.head(3)
        t = Vt[3]
        return (V - VelFunc(t)).squared_norm()
    
    phase1 = ode.phase("LGL7",Man1[1::],50)
    
    phase1.addLowerVarBound('Front',6,-L1Orbit[-1][6])
    phase1.addUpperVarBound('Front',6,2*L1Orbit[-1][6])

    # Departing from Orbit1
    phase1.addEqualCon("First",PosCon(OrbitTab1),[0,1,2,6])
    phase1.addStateObjective("First",DVObj(OrbitTab1),[3,4,5,6])

    phase2 = ode.phase("LGL7",Man2[0:-1],50)
    # Arriving at Orbit2
    phase2.addEqualCon("Last",PosCon(OrbitTab2),[0,1,2,6])
    phase2.addStateObjective("Last",DVObj(OrbitTab2),[3,4,5,6])
    
    phase1.addLowerVarBound('Last',6,-L2Orbit[-1][6])
    phase1.addUpperVarBound('Last',6,2*L2Orbit[-1][6])

    
    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase1)
    ocp.addPhase(phase2)

    #Enforce continuity in position and velocity between phases
    ocp.addForwardLinkEqualCon(phase1,phase2,range(0,6))
    ocp.setAdaptiveMesh()
    
    ocp.optimizer.set_EContol(1.0e-9)
    ocp.optimizer.set_OptLSMode("L1")
    ocp.optimize()
    
    Traj1 = phase1.returnTraj()
    Traj2 = phase2.returnTraj()
        
    DV1 = np.linalg.norm(Traj1[0][3:6]-OrbitTab1(Traj1[0][6])[3:6])
    DV2 = np.linalg.norm(Traj2[-1][3:6]-OrbitTab2(Traj2[-1][6])[3:6])
    print("Total DV:",(DV1+DV2)*ode.vstar)
    
    return Traj1,Traj2
    
    
    

if __name__ == "__main__":
    
    
    Jconst = 3.15    # Jacobi of target orbits
    dx     = 1.0e-5  # Manifold Perturbation
    dt     = 12.0    # Manifold Propagation Time
    nman   = 100     # # of Manifold trajectories
    nsegs  = 100     # Segments per orbit
    
    
    ode = CR3BP(c.MuEarth,c.MuMoon,c.LD)
    
    L1OrbitIG = ode.GenL1Lissajous(.03,0,180,0,1,100)
    L2OrbitIG = ode.GenL2Lissajous(.03,0,  0,0,1,100)
    
    L1Orbit = MakeOrbit(ode,L1OrbitIG,Jconst,nsegs)
    L2Orbit = MakeOrbit(ode,L2OrbitIG,Jconst,nsegs)
    
    
    
    
    UnstableL1 = GetManifold(ode,L1Orbit,dx,dt,nman,False)
    StableL2   = GetManifold(ode,L2Orbit,dx,dt,nman,True)
    
    Traj1IG,Traj2IG = FindClosestConnection(UnstableL1,StableL2)
    
    Traj2IG.reverse()
    
    Traj1,Traj2 = MakeHeteroclinic(ode,Traj1IG,Traj2IG,L1Orbit,L2Orbit)
    
    
    #########################################################
    plot = CRPlot(ode,'Earth','Moon','green','grey')
    
    plot.addTraj(L1OrbitIG,"L1 Initial Guess",'b',linestyle='dashed')
    plot.addTraj(L1Orbit,"L1 Converged",'b',linestyle='solid')

    plot.addTraj(L2OrbitIG,"L2 Initial Guess",'r',linestyle='dashed')
    plot.addTraj(L2Orbit,"L2 Converged",'r',linestyle='solid')

    fig = plt.figure()
    ax = plt.subplot()
    
    plot.Plot2dAx(ax,bbox="L1P2L2",pois=["L1","L2","P2"],plegend=True,legend=True)
    ax.grid(True)
    #plt.show()
    
    
    ##########################################################
    plot = CRPlot(ode,'Earth','Moon','green','grey')
    
    plot.addTrajSeq(UnstableL1,header='L1',colp='Blues')
    plot.addTrajSeq(StableL2,header='L2',colp='Reds')

    plot.addTraj(L1Orbit,"L1 Orbit",'b',linestyle='solid')

    plot.addTraj(L2Orbit,"L2 Orbit",'r',linestyle='solid')

    fig = plt.figure()
    ax = plt.subplot()
    
    plot.Plot2dAx(ax,bbox="L1P2L2",pois=["L1","L2","P2"],plegend=True)
    ax.grid(True)
    #plt.show()
    
    ##########################################################
    plot = CRPlot(ode,'Earth','Moon','green','grey')
    
    
    plot.addTraj(L1Orbit,"L1 Orbit",'b',linestyle='solid')
    plot.addTraj(Traj1,"O1",'b',linestyle='solid')

    plot.addTraj(L2Orbit,"L2 Orbit",'r',linestyle='solid')
    plot.addTraj(Traj2,"O2",'r',linestyle='solid')

    fig = plt.figure()
    ax = plt.subplot()
    
    plot.Plot2dAx(ax,bbox="L1P2L2",pois=["L1","L2","P2"],plegend=True)
    ax.grid(True)
    plt.show()
    