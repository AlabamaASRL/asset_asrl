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
Classic hypersensitive mesh refinement benchmark problem from Rao and company
https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.2114

'''

class HyperSens(oc.ODEBase):
    def __init__(self):
        ############################################################
        args  = oc.ODEArguments(2,1)
        
        x0 = args.XVar(0)
        u   =args.UVar(0)
        
        x0dot = -(x0) + u
        jdot  = (u**2 +x0**2)/2.0
        
        ode = vf.stack(x0dot,jdot)
        ##############################################################
        super().__init__(ode,2,1)



if __name__ == "__main__":

    xt0 = 1.5
    xtf = 1.0
    tf  = 10000.0  # lower values of final time make this problem easier
    
    
    TrajIG =[[0.0,0,t,0] for t in np.linspace(0,tf,1000)]
    
    ode= HyperSens()
    
    '''
    For tf  = 10000.0 we need at least 80 evenly spaced LGL5 or LGL7 segments on the initial mesh to approximate
    hypersensiteve behavior and not blow up, mesh refinement will take care of the rest, we'll start with 100
    to be safe
    '''
    nsegs   = 100

    phase = ode.phase("LGL7",TrajIG,nsegs)
    
    # Disabling splined controls seems to work best for this problem, but splines still work
    #phase.setControlMode("NoSpline")
    
    # Boundary Conditions
    phase.addBoundaryValue(PhaseRegs.Front,range(0,3),[xt0,0,0])
    phase.addBoundaryValue(PhaseRegs.Back ,[0,2],[xtf,tf])
    
    # Some loose bounds on variables
    phase.addLUVarBound("Path",0,-50,50)
    phase.addLUVarBound("Path",1,-50,50)
    phase.addLUVarBound("Path",3,-50,50)
    
    # Objecyive is final value of second state variable
    phase.addDeltaVarObjective(1,1.0)
    
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_SoeLSMode("L1")
    phase.optimizer.set_MaxLSIters(2)
    phase.optimizer.PrintLevel = 2
   
    '''
    For tf=10000.0 this problem is so sensitve that the static pivoting order 
    produced by the default METIS method is structurally singular despite
    the problem being well posed.
    However, MINDEG Ordering is stable for this problem problem
    Comment it out , enable console printing, and watch the number of peturbed pivots (PPS)
     and flying red colors to see what im talking about
    '''
    phase.optimizer.set_QPOrderingMode("MINDEG")
    
    
    #########################################
    #### The New Adaptive Mesh Interface ####
    #########################################
    
    '''
    Enable auto mesh refinement.It is disabled by default. When disabled, everything
    behaves as it did before. When enabled, after solving the first mesh, we utilize an adaptive
    mesh stragegy that calcuates the nuymber and spacing segments to meet as specified toelrance
    '''
    phase.setAdaptiveMesh(True)
    
    # Set Error tolerance on mesh: defaults to 1.0e-6
    phase.MeshTol = 1.0e-7
    
    # Set Max number of mesh iterations: defaults to 5
    phase.MaxMeshIters = 5
    
    # Make sure to set optimizer Econtol to be the same as or smaller than MeshTol
    phase.optimizer.set_EContol(1.0e-7)
    
    '''
    Specify the method used to estimate the error in each segment of the current trajectory
    '''
    
    '''
    Use the polynomial differenccing scheme of deboor,russell and grebow
    '''
    phase.MeshErrorEstimator='deboor'
    
    '''
    Or use a scheme leveraging an explicit integrator that we came up with, 
    set the phases integrator tolerances and step sizes appropraitely for good performance
    '''
    #phase.MeshErrorEstimator='integrator'
    
    
    '''
    # Specify which type of error must be less than MeshTol for the problem to be converged
    '''
    ## 'max' (default) will make sure that the max error in any of the segments is less than MeshTol
    phase.MeshErrorCriteria = 'max'
    
    ##'avg' will make sure the average error accross all segments is less than MeshTol
    #phase.MeshErrorCriteria = 'avg'
    
    ##'geometric' will make sure the geometric mean of the avg and max errors is less than MeshTol
    #phase.MeshErrorCriteria = 'geometric'
    
    ## 'endtoend' will reininegrate the entire control history and make sure the max error between
    ## the collocated and integrated final states is less than MeshTol
    #phase.MeshErrorCriteria = 'endtoend'
    

    '''
    Maximum multiple by which the # of segments can be increased between iterations
    this defaults to 10, which is too agressive for this problem (but still works).
    '''
    phase.MeshIncFactor=5.0
    
    '''
    Minimum multiple by which the # of segments can be reduced between iterations , Defaults to 0.7
    '''
    phase.MeshRedFactor=.25
    
    '''
    Factor by which we exagerate the error in each segment when calculating the needed number
    of points in the next mesh. This helps makes the nect iterate more likley to meet the tolerance but can 
    overprovison the # of segments, When using endtoend convergance criteria and the low order methods, be
    very agressive with this parameter
    '''
    phase.MeshErrFactor = 30.0  #defaults to 10
    
    phase.MaxMeshIters = 10

    '''
    As before, flag returned indicates whether the last call to psipot converged or not, it does
    not indicate whether the mesh is accurate/converged. Atm, that is checked by reading MeshConverged field.
    '''
    flag = phase.optimize_solve()    
   
    if(phase.MeshConverged):
        print("Fly it")
    else:
        print("Dont fly it")
        
    #######################################################
    TT = np.array(phase.returnTraj()).T
    
    fig,axs = plt.subplots(1,3)
    
    for ax in axs:
        ax.grid(True)
        ax.plot(TT[2],TT[0],label='x')
        ax.plot(TT[2],TT[3],label='u')
        ax.set_xlabel("t")
    
    
    axs[0].legend()
    axs[1].set_xlim([-.1,10])
    axs[2].set_xlim([tf-10,tf+.1])

    plt.show()
    ###############################################################
    '''
    Will produce 3 plots showing the evolution of the error in the mesh between iterations
    The first shows the estimated error as calculated by MeshErrorEstimator as a function
    of non-dimensional time along the trajectory.
    
    The second plot shows the error distribution function, which shows where the areas where
    more segments will be placed.
    
    The third plot is the normalized integral of the distribtion function, which is used to generate the mesh
    spacing for the next iteration (the dots on each curve). 
    '''
    PhaseMeshErrorPlot(phase,show=True)
    
