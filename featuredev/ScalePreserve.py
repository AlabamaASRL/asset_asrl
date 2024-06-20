import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments



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
    u0 = .4
    
    nsegs = 20
    
    
    TrajIG = [[x0,t,u0] for t in np.linspace(t0,tf,100)]
    
    iscale = 3.0
    vscale = 2.0
    
    xstar = 2.11
    tstar = 3.3
    ustar = 5.1
    pstar = 8.2
    
    
    
    phase = ode.phase("LGL5",TrajIG,nsegs)
    phase.setUnits([xstar,tstar,ustar])
    phase.addBoundaryValue("Front",[0,1],[x0,t0])
    phase.addBoundaryValue("Back", [1],  [tf])
    phase.addIntegralObjective(ODE.obj()*iscale,[0,2])
    phase.addValueObjective("Back",0,vscale)
    phase.optimize()
    
    Traj1 = phase.returnTraj()
    
   
    
    phase = ode.phase("LGL5",TrajIG,nsegs)
    phase.setUnits([xstar,tstar,ustar])
    phase.addBoundaryValue("Front",[0,1],[x0,t0])
    phase.addBoundaryValue("Back", [1],  [tf])
    phase.addIntegralObjective(ODE.obj()*iscale,[0,2])
    phase.addValueObjective("Back",0,vscale)

    phase.setAutoScaling(True)
    phase.optimize()
    Traj2 = phase.returnTraj()
    
    
    phase = ode.phase("LGL5",TrajIG,nsegs)
    phase.setUnits([xstar,tstar,ustar])
    phase.setStaticParams([0.0],[pstar])
    
    phase.addBoundaryValue("Front",[0,1],[x0,t0])
    phase.addBoundaryValue("Back", [1],  [tf])
    phase.addIntegralParamFunction(ODE.obj(),[0,2],0)
    phase.addValueObjective("Back",0,vscale)
    phase.addValueObjective("StaticParams",0,iscale)

    phase.setAutoScaling(True)
    phase.optimize()
    Traj3 = phase.returnTraj()

    
    ###########################################
    T1 = np.array(Traj1).T
    T2 = np.array(Traj2).T
    T3 = np.array(Traj3).T

    
    

    plt.plot(T1[1],T1[0],label   ="NoScale",marker='o')
    plt.plot(T2[1],T2[0],label   ="Scale1",marker='*')
    plt.plot(T3[1],T3[0],label   ="Scale2",marker='o')

    plt.legend()
    plt.grid(True)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$L,U$")
    plt.show()
    #################################

