import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags


'''
Vanderpol Osscilator Optimization Problem Taken From 
https://openmdao.github.io/dymos/examples/vanderpol/vanderpol.html
'''

class VanderPol(oc.ODEBase):
    def __init__(self):
        ############################################################
        args  = oc.ODEArguments(2,1)
        x0    = args[0]
        x1    = args[1]
        u     = args[3]
        
        x0dot = (1.0 - x1*x1)*x0 -x1 + u
        x1dot = x0
        ode = vf.stack(x0dot,x1dot)
        ##############################################################
        super().__init__(ode,2,1)
        
if __name__ == "__main__":

    import time
    
    
    t00 = time.perf_counter()
    ode = VanderPol()

    tf = 10.0
    
    TrajIG = [[0,0,t,0] for t in np.linspace(0,tf,100)]
    
    
    
    phase = ode.phase('LGL3',TrajIG,800)
    phase.integrator.setStepSizes(.25,.001,3)
    #phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("Front",range(0,3),[0,1,0])
    phase.addLUVarBound("Path",3,-0.75,1.0,1.0)
    phase.addLUVarBound("Path",0,-0.25,1.0,1.0)

    phase.addIntegralObjective(Args(3).squared_norm(),[0,1,3])
    phase.addBoundaryValue("Back",[0,1,2],[0.0,0.0,tf])
    phase.optimizer.PrintLevel=1
    phase.setThreads(8,8)
    phase.optimizer.set_tols(1.0e-8,1.0e-8,1.0e-8)
    #phase.setAdaptiveMesh(True)
    phase.setMeshErrorEstimator("integrator")
    
    tff = time.perf_counter()

    print(tff-t00)
    
    
    
    phase.optimize()
    Traj = phase.returnTraj()
    T = np.array(Traj).T
    plt.plot(T[2],T[0],label=r'$x_0$')
    plt.plot(T[2],T[1],label=r'$x_1$')
    plt.plot(T[2],T[3],label=r'$u$')
    plt.grid(True)
    plt.legend()
    plt.xlabel(r"$t$")
    plt.show()
