import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt

from asset_asrl.OptimalControl.MeshErrorPlots import PhaseMeshErrorPlot

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

'''
Source for problem formulation
https://arxiv.org/pdf/1905.11895.pdf
'''
import time

class ODE(oc.ODEBase):
    def __init__(self,L):
        XVars =6
        UVars =3
        ####################################
        XtU = oc.ODEArguments(XVars,UVars)
        
        y1,y2,y3,y4,y5,y6 = XtU.XVec().tolist()
        u1,u2,u3    = XtU.UVec().tolist()
        
        Ip =((L-y1)**3 - y1**3)/3.0
        It = Ip*(vf.sin(y5)**2)
        
        y1dot = y2
        y2dot = u1/L
        y3dot = y4
        y4dot = u2/It
        y5dot = y6
        y6dot = u3/Ip
        
       
        
        ode = vf.stack([y1dot,y2dot,y3dot,y4dot,y5dot,y6dot])
       
        super().__init__(ode,XVars,UVars)


def Jump(Traj):
    
    TT = np.array(Traj).T
    
    U0 = TT[7]
    
    U0N = ( U0 - min(U0))/(.01+max(U0) - min(U0))
    
    JF = []
    ts = []
    t = TT[6]
    for i in range(1,len(U0N)-1):
        tt = t[i]
        
        cj0 = 2/( ( t[i-1] - t[i])*( t[i-1] - t[i+1]))
        cj1 = 2/( ( t[i] - t[i-1])*( t[i] - t[i+1]))
        cj2 = 2/( ( t[i+1] - t[i])*( t[i+1] - t[i-1]))
        
        q =  cj2

        r = ( U0N[i-1]*cj0 +U0N[i]*cj1 +U0N[i+1]*cj2   )/q
        JF.append( abs(r))
        ts.append(tt)
        
        
    
    
    tstmp = np.linspace(ts[0],ts[-1],640)
    #tstmp = ts[0:-1]
    t0 = time.perf_counter()
    jt = oc.jump_function(ts,U0N[1:-1],tstmp,5)
    jy1 = oc.jump_function_mmod(ts,U0N[1:-1],tstmp,[3,4,5,6,7,8,9])
    tf = time.perf_counter()
    print(tf-t0)
    
    plt.plot(TT[6],U0N)
    plt.plot(ts,JF)
    #plt.plot(tstmp,abs(jt))
    plt.plot(tstmp,jy1)

    
    
    
    plt.show()
    
    
    
if __name__ == "__main__":
    
    
    ts = np.linspace(0,2,101)
    ys = [ 1 if t<1 else -1 for t in ts ]
    
    js = oc.jump_function_mmod(ts,ys,ts[0:-1],[3,4,5])
    plt.plot(ts,ys)
    plt.plot(ts[0:-1],js)
    
    plt.show()
    
    
    L = 5
    tf = 1
    
    Traj = []
    
    ts = np.linspace(0,tf,200)
    
    for t in ts:
        X = np.zeros((10))
        X[0] = 4.5
        X[2] = 2*np.pi*(t/tf)/3
        X[4] = np.pi/4
        X[6] = t
        
        Traj.append(X)
        
        
        
    ode = ODE(L)
    phase = ode.phase("LGL7",Traj,128)
    phase.setControlMode("NoSpline")
    #phase.setControlMode("BlockConstant")

    phase.addBoundaryValue("Front",range(0,7),Traj[0][0:7])
    phase.addBoundaryValue("Back",range(0,6),Traj[-1][0:6])
    
    
    phase.addLUVarBounds("Path",[7,8,9],-1,1.,1.0)
    phase.addDeltaTimeObjective(1.0)
    
    #phase.setAdaptiveMesh(True)
    phase.optimizer.set_QPOrderingMode("MINDEG")
    phase.optimizer.PrintLevel =1
    
    
    t0 = time.perf_counter()
    phase.solve_optimize()
    tf = time.perf_counter()
    
    print(tf-t0)
    
    
    Jump(phase.returnTraj())
    
    PhaseMeshErrorPlot(phase,show=True)

    T = np.array(phase.returnTraj()).T
    
    
    plt.plot(T[6],T[7],marker='o')
    plt.plot(T[6],T[8])
    plt.plot(T[6],T[9])
    plt.show()
    

    
        
    
        
        
