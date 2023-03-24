import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

'''
Source for problem formulation
https://arxiv.org/pdf/1905.11895.pdf
'''

class ODE(oc.ODEBase):
    def __init__(self,a1,a2,a3):
        XVars =4
        UVars =2
        ####################################
        XtU = oc.ODEArguments(XVars,UVars)
        
        N1,N2,N3,up = XtU.XVec().tolist()
        u1,u2    = XtU.UVec().tolist()
        
        
        N1dot = -a1*N1 + 2*a3*N3*(1.0-u1)
        N2dot = -a2*N2*u2 + a1*N1
        N3dot = a3*N3 + a2*N2*u2
        
        ode = vf.stack([N1dot,N2dot,N3dot,u1])
       
        super().__init__(ode,XVars,UVars)

    

##############################################

if __name__ == "__main__":
    
    
    a1 = .197
    a2 = .395
    a3 = .107
    
    r1 = 1
    r2 =.5
    r3 =1
    tf = 7
    
    u2min = .7

    ode = ODE(a1,a2,a3)
    integ = ode.integrator(.01)
    
    
    XtU0 = np.zeros((7))
    
    XtU0[0:3] = [38,2.5,3.25]
    XtU0[5] = 1
    XtU0[6] =.71
    
    
    TrajIG =  integ.integrate_dense(XtU0,tf)
    
    phase = ode.phase("LGL3",TrajIG,32)
    phase.addBoundaryValue("Front",range(0,5),XtU0[0:5])
    
    phase.addLUVarBound("Path",5,0,1,1.0)
    phase.addLUVarBound("Path",6,u2min,1,1.0)

    phase.addStateObjective("Back",Args(4).dot([r1,r2,r3,1.0]),range(0,4))
    
    phase.addDeltaTimeEqualCon(tf)
    phase.setThreads(1,1)
    phase.setAdaptiveMesh(True)
    phase.setMeshErrorCriteria("endtoend")
    phase.solve_optimize()
    
    

    Traj = phase.returnTraj()    
    
    
    
    
    T = np.array(Traj).T
    plt.plot(T[4],T[0])
    plt.plot(T[4],T[1])
    plt.plot(T[4],T[2])
    plt.show()
    plt.plot(T[4],T[5])
    plt.plot(T[4],T[6])
    plt.show()