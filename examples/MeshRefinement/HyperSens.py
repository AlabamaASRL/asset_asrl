import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
from asset_asrl.OptimalControl.MeshErrorPlots import PhaseMeshErrorPlot

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes = oc.ControlModes

'''
Hyper-Sensitive Problem
Classic mesh refinement benchamrk problem from Rao and company
https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.2114

'''

class HyperSens(oc.ODEBase):
    def __init__(self):
        ############################################################
        args  = oc.ODEArguments(2,1)
        
        x0 = args.XVar(0)
        u   =args.UVar(0)
        
        x0dot = -(x0)**3 + u
        jdot  = (u**2 +x0**2)/2.0
        
        ode = vf.stack(x0dot,jdot)
        ##############################################################
        super().__init__(ode,2,1)



if __name__ == "__main__":

    xt0 = 1.5
    xtf = 1.0
    tf  = 10000.0
    
    nsegs   = 300
    
    TrajIG =[[0.0,0,t,0] for t in np.linspace(0,tf,100)]
    
    ode= HyperSens()
    
    phase = ode.phase("LGL7",TrajIG,nsegs)
    # Disable splined controls
    phase.setControlMode("NoSpline")
    phase.addBoundaryValue(PhaseRegs.Front,range(0,3),[xt0,0,0])
    phase.addBoundaryValue(PhaseRegs.Back ,[0,2],[xtf,tf])
    phase.addLUVarBound("Path",0,-50,50)
    phase.addLUVarBound("Path",1,-50,50)
    phase.addLUVarBound("Path",3,-50,50)
    phase.addDeltaVarObjective(1,1.0)
    
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_SoeLSMode("L1")
    phase.optimizer.set_MaxLSIters(2)
    phase.optimizer.PrintLevel = 2
    phase.setThreads(1,1)
    phase.AdaptiveMesh=True
    
    # Method to estijate error in mesh
    # Use the polynomial differenccing scheme of deboor,russell and grebow
    phase.MeshErrorEstimator='deboor'
    #Use the phases integrator
    #phase.MeshErrorEstimator='integrator'
    
    # Set Error tolerance on mesh: defaults to 1.0e-6
    phase.MeshTol = 1.0e-7
    
    # Make sure to set optimizer Econtol to be the same as or smaller than MeshTol
    phase.optimizer.set_EContol(1.0e-7)

    
    # Maximum multiple by which the # of segments can be increased between iterations
    # This defaults to 10, which is far too agressive for this problem (but still works)
    phase.MeshIncFactor=2.0
    # Minimum multiple by which the # of segments can be reduced between iterations
    phase.MeshRedFactor=.5


    ## MINDEG Ordering MUCH more stable for this problem
    phase.optimizer.set_QPOrderingMode("MINDEG")
    phase.optimizer.QPPivotPerturb =6
    
    phase.optimize_solve()    
   
    
    PhaseMeshErrorPlot(phase,show=True)
    
    TT = np.array(phase.returnTraj()).T
    
    
    plt.plot(TT[2],TT[0])
    plt.plot(TT[2],TT[3])
    plt.show()

