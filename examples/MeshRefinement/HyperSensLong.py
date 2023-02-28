import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
from asset_asrl.OptimalControl.MeshErrorPlots import PhaseMeshErrorPlot

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments


'''
Hyper-Sensitive Problem
Classic hypersensitive mesh refinement benchmark problem from Rao and company
https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.2114

There is also a slightly modified cubed version that they use 
'''

class HyperSens(oc.ODEBase):
    def __init__(self,cubed=False):
        ################################
        XtU  = oc.ODEArguments(1,1)
        x    = XtU.XVar(0)
        u    = XtU.UVar(0)
    
        xdot = -(x) + u
        if(cubed):
            xdot = -(x)**3 + u
       
        ################################
        super().__init__(xdot,1,1)



if __name__ == "__main__":

    xt0 = 1.5
    xtf = 1.0
    tf  = 10000.0  # smaller tf makes it easier to solve
    
    cubed = False
    ode= HyperSens(cubed)
    
    '''
    Both of these initial guesses work equally well for the original non-cubed problem.
    Second one works much better for the cubed version
    '''
    ## Lerp boundary conditions
    TrajIG =[[xt0*(1-t/tf) + xtf*(t/tf),t,0] for t in np.linspace(0,tf,1000)]
    ## All Zeros
    #TrajIG =[[0.0,t,0] for t in np.linspace(0,tf,1000)]
    
    '''
    Need minimum of 3 segments on initial mesh to get original to converge.
    Need minumum of about 16 segments on initial mesh for cubed version to converge
    '''
    nsegs   = 50
    phase = ode.phase("LGL7",TrajIG,nsegs)  # LGL7 recommended for cubed version
    
    # A knob to turn
    phase.setControlMode("NoSpline")
    
    # Boundary Conditions
    phase.addBoundaryValue("First",[0,1],[xt0,0])
    phase.addBoundaryValue("Last" ,[0,1],[xtf,tf])
    
    #Objective
    phase.addIntegralObjective(Args(2).squared_norm()/2,[0,2])
    
    # Some loose bounds on variables
    phase.addLUVarBound("Path",0,-50,50)
    phase.addLUVarBound("Path",2,-50,50)
        
    
    # Enable line searches
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_SoeLSMode("L1")
    phase.optimizer.set_MaxLSIters(2)
    
    phase.optimizer.PrintLevel = 2
    phase.setThreads(1,1)
   
    '''
    For tf=10000.0 this problem is so sensitve that the static pivoting order 
    produced by the default METIS method is structurally singular despite
    the problem being well posed.However, MINDEG Ordering is stable for this problem
    Comment it out , enable console printing, and watch the number of peturbed pivots (PPS)
    and flying red colors to see what im talking about
    '''
    phase.optimizer.set_QPOrderingMode("MINDEG")
    
    ###########################################################################
    ############# The New Adaptive Mesh Interface #############################
    ###########################################################################
    
    '''
    Enable auto mesh refinement.It is disabled by default. When disabled, everything
    behaves as it did before. When enabled, after solving the first mesh, we utilize an adaptive
    stragegy that calcuates the number and spacing segments to meet a specified toelrance
    '''
    phase.setAdaptiveMesh(True)
    
    ## Set Error tolerance on mesh: 
    phase.setMeshTol(1.0e-7) #default = 1.0e-6
    ## Make sure to set optimizer Econtol to be the same as or smaller than MeshTol
    phase.optimizer.set_EContol(1.0e-7)
    
    ## Set Max number of mesh iterations: 
    phase.setMaxMeshIters(10) #default = 10
    
    
    
    '''
    Specify the method used to estimate the error in each segment of the current trajectory
    '''
    ##Use the polynomial differencing scheme of deboor
    phase.setMeshErrorEstimator('deboor')  #default
    
    ##Use the phase's explicit integrator, set the phases integrator tolerances and step sizes appropraitely for good performance
    #phase.setMeshErrorEstimator('integrator')
    
    
    '''
    # Specify which type of error must be less than MeshTol for the problem to be converged
    '''
    ##'max' will make sure that the max error in any of the segments is less than MeshTol
    phase.setMeshErrorCriteria('max')  # default
    
    ##'avg' will make sure the average error accross all segments is less than MeshTol
    #phase.setMeshErrorCriteria('avg')
    
    ## 'endtoend' will reininegrate the entire control history and make sure the max error between
    ## the collocated and integrated final states is less than MeshTol
    #phase.setMeshErrorCriteria('endtoend')
    
    
    '''
    Parameters controlling how quickly/agressively the mesh can change between iterates
    '''
    ## Maximum multiple by which the # of segments can be increased between iterations
    phase.setMeshIncFactor(5.0)  # default = 5
    
    ## Minimum multiple by which the # of segments can be reduced between iterations 
    phase.setMeshRedFactor(.5)  #default = .5
    
    '''
    Factor by which we exagerate the error in each segment when calculating the needed number
    of points in the next mesh. This helps makes the next iterate more likley to meet the tolerance but can 
    overprovison the # of segments, When using endtoend convergance criteria and/or the low order collocation methods, be
    very agressive with this parameter (=100)
    '''
    phase.setMeshErrFactor(10.0)  #default = 10
    

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
    
    fig = plt.figure()
    
    ax0 = plt.subplot(211)
    ax1 = plt.subplot(223)
    ax2 = plt.subplot(224)
    
    axs =[ax0,ax1,ax2]

    for ax in axs:
        ax.grid(True)
        ax.plot(TT[1],TT[0],label='x',marker='o')
        ax.plot(TT[1],TT[2],label='u',marker='o')
        ax.set_xlabel("t")
    
    
    axs[0].legend()
    axs[1].set_xlim([-.5,12])
    axs[2].set_xlim([tf-12,tf+.5])

    plt.show()
    ###############################################################
    
    
    
    '''
    Will produce 3 plots showing the evolution of the error in the mesh between iterations
    The first shows the estimated error as calculated by MeshErrorEstimator as a function
    of non-dimensional time along the trajectory.
    
    The second plot shows the error distribution function, which shows where the areas where
    more segments need to be placed.
    
    The third plot is the normalized integral of the distribtion function, which is used to generate the mesh
    spacing for the next iteration. 
    '''
    PhaseMeshErrorPlot(phase,show=True)