import asset as ast
import matplotlib.pyplot as plt
import numpy as np

################################################################################
# Setup
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
TModes = oc.TranscriptionModes

###############################################################################


class Model(oc.ode_x_u.ode):
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

IG = [[.0,v,t,-1] for t,v in zip(ts,vs)]

ode = Model()

phase = ode.phase(TModes.LGL3,IG,256)
phase.addBoundaryValue(PhaseRegs.Front,range(0,3),[0,1,0])
phase.addUpperVarBound(PhaseRegs.Path,0,1/9)
phase.addIntegralObjective((Args(1)[0]**2)/2,[3])
phase.addBoundaryValue(PhaseRegs.Back,range(0,3) ,[0,-1,1])
    
phase.optimizer.PrintLevel=1
phase.optimize()


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
