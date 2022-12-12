import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import seaborn as sns


norm      = np.linalg.norm
vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
solvs     = ast.Solvers


class Chain(oc.ode_x_u.ode):
    def __init__(self):
        args = oc.ODEArguments(1,1)
        super().__init__(args[2],1,1)
def Energy():
    x,u = Args(2).tolist()
    return x*vf.sqrt(1+u**2)
def Length():
    u, = Args(1).tolist()
    return vf.sqrt(1+u**2)
        

def GetIG(a,b,ts):
    IG = []

    for t in ts:
        if(b>a):tm = .25
        else:tm = .75
        x = 2*abs(b-a)*t*(t-2*tm) + a
        u = 2*abs(b-a)*( t*2.0 - 2*tm ) 
        IG.append([x,t,u])
        
    return IG
    
def Job(a,b,n,L):
    
    ts = np.linspace(0,1,n)
    IG = GetIG(a,b,ts)
    
    ode = Chain()
    
    phase = ode.phase("LGL3",IG,n)
    phase.setStaticParams([L])
    
    phase.addBoundaryValue("Front",[0,1],[a,0])
    phase.addBoundaryValue("Back", [0,1],[b,1])
    phase.addBoundaryValue("StaticParams", [0],[L])
    
    phase.addUpperVarBound("Path",0,max(a,b)+.001)
    
    phase.addIntegralObjective(Energy(),[0,2])
    phase.addIntegralParamFunction(Length(),[2],0)
    
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.MaxLSIters = 2
    phase.optimizer.PrintLevel= 1
    phase.JetJobMode = ast.Solvers.JetJobModes.SolveOptimize
    
    return phase



        
        
        
a = 1
b = 3
L = 4
n = 200


Ls = np.linspace(2.25,8,1000)
cols = sns.color_palette("plasma",len(Ls))


JArgs = [(a,b,n,L) for L in Ls]

Res = solvs.Jet.map(Job,JArgs,16)

for i,res in enumerate(Res):
    
    Traj = res.returnTraj()
    
    TT = np.array(Traj).T
    
    plt.plot(TT[1],TT[0],color = cols[i])
    
    
    

plt.grid(True)
plt.show()





