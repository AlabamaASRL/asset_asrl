import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes    = oc.ControlModes



#https://arxiv.org/pdf/2003.00142.pdf#page=15&zoom=100,65,877

lstar = 1
tstar = 1

vstar = lstar/tstar
astar = vstar/tstar

class ODE(oc.ode_x_u.ode):
    
    def __init__(self,la,lb):
        args = oc.ODEArguments(4,2)
        
        x,y,psi,v = args.XVec().tolist()
        
        acc,alpha = args.UVec().tolist()
        
        beta = vf.arctan((la/(la+lb))*vf.tan(alpha))
        
        
        xdot = v*vf.cos(psi+beta)
        ydot = v*vf.sin(psi+beta)
        psidot = v*vf.sin(beta)/lb
        vdot = acc
        
        ode = vf.stack(xdot,ydot,psidot,vdot)
        super().__init__(ode,4,2)
        
        
def PathCon(xobs,yobs, aobs,m):
    x,y = Args(2).tolist()

    denom = aobs+m
    ellips = ((x-xobs)/denom)**2 + ((y-yobs)/denom)**2
    
    return 1 - ellips


aobs = 5 /lstar
xobs = 0
yobs = 50
m = 2.5

la = 1.58/lstar
lb = 1.72/lstar



accbound = 2/astar
vlbound = 5/vstar
vubound = 29/vstar
        

x0   =0
y0   =0
psi0 = np.pi/2
v0   = 15 /vstar
alpha0 = 0

xf = 0
yf = 100/lstar



tfIG =yf/v0


TrajIG = []

for t in np.linspace(0,tfIG,100):
    X = np.zeros(7)
    X[0]=x0+5.1
    X[1]=yf*t/tfIG
    X[2]=psi0
    X[3]=v0
    X[4]=t
    X[5]=accbound/2
    X[6]=0.01
    TrajIG.append(X)
    
    

ode  = ODE(la,lb)


phase = ode.phase("Trapezoidal",TrajIG,100)
phase.setControlMode("BlockConstant")
phase.addBoundaryValue("Front",[0,1,2,3,4],[x0,y0,psi0,v0,0])

phase.addLUVarBound("Path",5,-accbound,accbound)
phase.addLUVarBound("Path",6,-np.pi/6,np.pi/6)
phase.addInequalCon("Path",PathCon(xobs,yobs,aobs,m),[0,1])
phase.addBoundaryValue("Back",[0,1],[xf,yf])
phase.addDeltaTimeObjective(1.0)
phase.Threads  = 4
phase.optimizer.QPThreads=4
#phase.optimizer.set_QPOrderingMode("MINDEG")
phase.EnableHessianSparsity = True
phase.optimizer.PrintLevel=1
#phase.optimizer.set_OptLSMode("AUGLANG")
phase.optimize()

TrajF = phase.returnTraj()



TT = np.array(TrajF).T


plt.plot(TT[1],TT[0])

plt.grid(True)

plt.show()

plt.plot(TT[4],TT[6])
plt.show()
print(tfIG)


        
