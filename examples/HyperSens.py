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

See MeshRefinement Folder for more in depth version

'''

class HyperSens(oc.ODEBase):
    def __init__(self):
        ################################
        XtU  = oc.ODEArguments(1,1)
        x    = XtU.XVar(0)
        u    = XtU.UVar(0)
        xdot = -(x) + u
        ################################
        super().__init__(xdot,1,1)

if __name__ == "__main__":

    xt0 = 1.5
    xtf = 1.0
    tf  = 10000.0  
    
    ode= HyperSens()
    ## Lerp boundary conditions
    TrajIG =[[xt0*(1-t/tf) + xtf*(t/tf),t,0] for t in np.linspace(0,tf,1000)]
    
    nsegs   = 10
    phase = ode.phase("LGL7",TrajIG,nsegs)  
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
    ## Neccessary for this problem
    phase.optimizer.set_QPOrderingMode("MINDEG")
    phase.optimizer.PrintLevel = 2
    phase.setThreads(1,1)
    
    # Enable Adaptive Mesh
    phase.setAdaptiveMesh(True)
    ## Set Error tolerance on mesh: 
    phase.setMeshTol(1.0e-6) #default = 1.0e-6
    ## Set Max number of mesh iterations: 
    phase.setMaxMeshIters(10) #default = 10
    ## Make sure to set optimizer Econtol to be the same as or smaller than MeshTol
    phase.optimizer.set_EContol(1.0e-7)
    
    flag = phase.optimize_solve() #Recommended to run with post solve enabled    
   
    if(phase.MeshConverged and flag==0):
        print("Success")
    else:
        print("Failure")
        
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
    PhaseMeshErrorPlot(phase,show=True)