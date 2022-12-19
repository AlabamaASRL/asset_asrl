import asset_asrl as ast
import matplotlib.pyplot as plt
import numpy as np

################################################################################
# Setup
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments

###############################################################################


class Model(oc.ODEBase):
    def __init__(self):
        Xvars = 2
        Uvars = 1
        ############################################################
        args = oc.ODEArguments(Xvars, Uvars)
        x = args.XVec()[0]
        v = args.XVec()[1]
        u = args.UVec()[0]
        ode = vf.stack([v, u])
        #############################################################
        super().__init__(ode, Xvars, Uvars)



n  = 100
ts = np.linspace(0,1,n)
vs = np.linspace(1,-1,n)

IG = [[.0,v,t,0] for t,v in zip(ts,vs)]

ode = Model()

phase = ode.phase("LGL5",IG,65)
#phase.setControlMode("BlockConstant")
phase.addBoundaryValue("Front",range(0,3),[0,1,0])
phase.addUpperVarBound("Path",0,1/9)
phase.addIntegralObjective((Args(1)[0]**2)/2,[3])
phase.addBoundaryValue("Back",range(0,3) ,[0,-1,1])
phase.optimizer.set_OptLSMode("L1")
#phase.optimizer.set_QPOrderingMode("MINDEG")
phase.optimizer.set_KKTtol(1.0e-10)
phase.optimizer.PrintLevel=0
phase.Threads = 2
phase.optimizer.QPThreads=2
phase.optimize()

Traj = phase.returnTraj()
ii=0
for i in range(0,len(Traj)-1):
    t1 =Traj[i][2]
    u1 = Traj[i][3]
    
    t2 =Traj[i+1][2]
    u2 = Traj[i+1][3]
    
    h = t2-t1
    
    ii += (u1**2+u2**2)*h/4
    
print(ii)

TT = np.array(phase.returnTraj()).T

#############################################################################
fig,axs = plt.subplots(3,1)

axs[0].plot(TT[2],TT[0])
axs[1].plot(TT[2],TT[1])
axs[2].plot(TT[2],TT[3])

axs[0].set_ylabel(r"$x$")
axs[1].set_ylabel(r"$v$")
axs[2].set_ylabel(r"$u$")
axs[2].set_xlabel(r"$t$")


for i in range(0,3):
    axs[i].grid(True)
plt.show()

#############################################################################
