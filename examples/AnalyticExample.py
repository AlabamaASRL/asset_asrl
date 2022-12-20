import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


class AnalyticODE(oc.ODEBase):
    def __init__(self):
        
        XVars =1
        UVars =1
        
        args = oc.ODEArguments(XVars,UVars)
        x = args.XVar(0)
        u = args.UVar(0)
        xdot = .5*x + u
        super().__init__(xdot,XVars,UVars)

    class obj(vf.ScalarFunction):
        def __init__(self):
            args = Args(2)
            x=args[0]
            u=args[1]
            obj = u*u + x*u + 1.25*x**2
            super().__init__(obj)
        

##############################################

if __name__ == "__main__":

    ode = AnalyticODE()
    
    x0 = 1.0
    t0 = 0.0
    
    tf = 1.0
    u0 = .00
    
    nsegs = 100
    
    TrajIG = [[x0,t,u0] for t in np.linspace(t0,tf,100)]
    
    phase = ode.phase("LGL3",TrajIG,nsegs)
    phase.addBoundaryValue("Front",[0,1],[x0,t0])
    phase.addBoundaryValue("Back", [1],  [tf])
    phase.addIntegralObjective(AnalyticODE.obj(),[0,2])
    phase.optimize()
    
    
    Traj = phase.returnTraj()
    CTraj= phase.returnCostateTraj()
    
    ###########################################
    T = np.array(Traj).T
    CT = np.array(CTraj).T
    
    X = T[0]
    t = T[1]
    U = T[2]
    
    ## Collocation Costates
    lcoll = CT[0]

    ### Analytic costates
    lstar = 2*np.cosh(1-t)*np.tanh(1-t)/np.cosh(1)
        
    plt.plot(t,lcoll,label   ='Collocation',marker='o')
    plt.plot(t,lstar,label   ='Analytic')
    plt.grid(True)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$l$")
    plt.show()
    #################################
    
    