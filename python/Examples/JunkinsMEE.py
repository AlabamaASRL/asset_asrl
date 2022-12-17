import numpy as np
import matplotlib.pyplot as plt
import asset_asrl as ast
from asset_asrl.Astro.Extensions.ThrusterModels import CSIThruster
from asset_asrl.Astro.AstroModels import MEETwoBody_CSI
 
from FramePlot import TBPlot,plt,colpal
import asset_asrl.Astro.Constants as c


##############################################################################
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


def Run2():
   
    
    Isp_dim  = 3000       #S
    Tmag_dim = .32        #N
    tf_dim   = 3534*c.day #s 
    mass_dim = 4000       #kg
   
    
    thruster = CSIThruster(Tmag_dim, Isp_dim, mass_dim)
    ode = MEETwoBody_CSI(c.MuSun,c.AU,thruster)
    
    
    tf = tf_dim/ode.tstar
    integ = ode.integrator(.1)
    
    ## Already Non Dimesonalized
    X0 = np.array([0.99969,-0.00376, 0.01628,-7.702e-6, 6.188e-7, 14.161])
    XF = np.array([1.5536, 0.15303,-0.51994, 0.01618, 0.11814, 46.3302])
    
    Istate = np.zeros((11))
    Istate[0:6]=X0
    Istate[6]=1
    Istate[9]=.5
    
    
    ts = np.linspace(0,tf,500)
    
    Traj = []
    
    for t in ts:
        State = np.zeros((11))
        Xi = X0 + (XF-X0)*t/tf
        State[0:6]=Xi
        State[6]=1
        State[7]=t
        State[9]=.5
        Traj.append(State)
        
        
        
    TrajIG = Traj
    
    phase = ode.phase("LGL3",Traj,192)
    phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("Front",range(0,8),Istate[0:8])
    phase.addLUNormBound("Path",range(8,11),.0001,1,1)
    phase.addBoundaryValue("Back",[7],[tf])
    phase.addBoundaryValue("Back",range(0,6),XF[0:6])
    phase.addValueObjective("Back",6,-1.0)
    
    phase.optimizer.set_OptLSMode("AUGLANG")
    phase.optimizer.set_MaxLSIters(2)
    phase.optimizer.set_MaxIters(300)
    phase.optimizer.set_MaxAccIters(200)
    phase.optimizer.set_BoundFraction(.997)
    phase.optimizer.set_HpertParams(deltaH = 1.0e-7,incrH = 8.,decrH = .33)

    phase.optimizer.PrintLevel = 0
    #phase.optimizer.set_QPOrderingMode("MINDEG")
    
    phase.setThreads(8,8)
    phase.optimize()
   
    Traj = phase.returnTraj()
    
    
    TT = np.array(Traj).T
    
    
    
    plt.plot(TT[7],TT[0])
    

    plt.show()
    
    
    plt.plot(TT[7],TT[8])
    plt.plot(TT[7],TT[9])
    plt.plot(TT[7],TT[10])
    #plt.plot(TT[7],(TT[8]**2+TT[9]**2+TT[10]**2)**.5)

    plt.show()
    
    TrajCart   = ode.MEEToCartesian(Traj)
    TrajCartIG = ode.MEEToCartesian(TrajIG)

    
    plot = TBPlot(ode)
    
    plot.addTraj(TrajCart, "Conv",color='b')
    plot.addTraj(TrajCartIG, "IG",color='r')

    plot.Plot2d(bbox='Two')
            
        
        
        
    
    
    
    
    
    print(tf)
    
    print(Tmag)
    
    print(mdot)
    
    
if __name__ == "__main__":
    
    Run2()
    

        

