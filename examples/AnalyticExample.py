import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

'''
Source for problem formulation
https://www.hindawi.com/journals/aaa/2014/851720/
'''

class ODE(oc.ODEBase):
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

    ode = ODE()
    
    x0 = 1.0
    t0 = 0.0
    
    tf = 1.0
    u0 = .00
    
    nsegs = 20
    
    
    
    TrajIG = [[x0,t,u0] for t in np.linspace(t0,tf,100)]
    
    phase = ode.phase("LGL5",TrajIG,nsegs)
    phase.addBoundaryValue("Front",[0,1],[x0,t0])
    phase.addBoundaryValue("Back", [1],  [tf])
    phase.addIntegralObjective(ODE.obj(),[0,2])
    phase.optimize()
    
    
    Traj = phase.returnTraj()
    CTraj= phase.returnCostateTraj()

    
    ###########################################
    T = np.array(Traj).T
    CT = np.array(CTraj).T
    
    X = T[0]
    t = T[1]
    
    #Collocation control
    U = T[2]
    ## Collocation Costates
    L = CT[0]

    ### Analytic costates
    Lstar = 2*np.cosh(1-t)*np.tanh(1-t)/np.cosh(1)
    ### Analytic control
    Ustar = -(np.tanh(1-t)+.5)*np.cosh(1-t)/np.cosh(1)
    
    
    

    plt.plot(t,L,label   =r'$L$' + '-Collocation',marker='o')
    plt.plot(t,Lstar,label   =r'$L$' +'-Analytic')
    
    plt.plot(t,U,label   =r'$U$' + '-Collocation',marker='o')
    plt.plot(t,Ustar,label   =r'$U$' +'-Analytic')
    plt.legend()
    plt.grid(True)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$L,U$")
    plt.show()
    #################################
    
    
