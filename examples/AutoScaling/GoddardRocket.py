import numpy as np
import matplotlib.pyplot as plt
import asset_asrl as ast
import asset_asrl.VectorFunctions as vf
import asset_asrl.OptimalControl as oc
from asset_asrl.VectorFunctions import Arguments as Args


'''
Classic Goddard rocket problem with singular arc

Betts, J.T. "Practical methods for Optimal Control and Estimation Using Nonlinear Programming", Cambridge University Press, 2009
See Section 4.14

'''

g0 = 32.2 
W  = 203000

Lstar = 10000.0   ## feet
Tstar = 60.0      ## sec
Mstar = 1         ## slugs
Vstar   = Lstar/Tstar


rho0  =.002378         
h_ref = 23800          
g     = g0             
Tmag  = 200           
c     = 1580.94        
sigma = 5.4915e-5      

m0   = 3
mf   = 1



class GoddardRocket(oc.ODEBase):
    def __init__(self,sigma,c,h_ref,Tmag, g):
        ############################################################
        XtU  = oc.ODEArguments(3,1)
        h,v,m = XtU.XVec().tolist()
        u = XtU.UVar(0)
        
        hdot=v
        vdot = (u*Tmag - sigma*(v**2)*vf.exp(-h/h_ref))/m -g
        mdot = -u*Tmag/c
        
        ode = vf.stack(hdot,vdot,mdot)
        ##############################################################
        Vgroups = {}
        Vgroups[("h","altitude")] = h
        Vgroups[("v","velocity")] = v
        Vgroups[("m","mass")] = m
        Vgroups[("t","time")] = XtU.TVar()
        Vgroups["u"]=u
        
        super().__init__(ode,3,1,Vgroups = Vgroups)
 
def PathCon(sigma,c,h_ref,Tmag, g):
    h,v,m,u = Args(4).tolist()
    t1 = (u*Tmag - sigma*(v**2)*vf.exp(-h/h_ref)) -g*m
    t2 = (m*g/( 1 + 4*(c/v) +2*(c/v)**2 ))*( c*c*(1+v/c)/(h_ref*g) -1.0 -2.0*c/v )
    return t1-t2
       

def Plot(axs,Traj,label=''):
    T = np.array(Traj).T
    
    
    axs[0].plot(T[3],T[0],label=label)
    axs[1].plot(T[3],T[1])
    axs[2].plot(T[3],T[2])
    axs[3].plot(T[3],T[4]*Tmag)
    
    

if __name__ == "__main__":

    def Ulaw():
        m = Args(1)[0]
        return vf.ifelse(m>mf,1,0)
        
    def StopFunc(x): return (x[1]<0)
        
    ode = GoddardRocket(sigma,c,h_ref,Tmag, g)
    
    units = ode.make_units(h=Lstar,v = Vstar,m = Mstar,t = Tstar)
    
    integ = ode.integrator(.01,Ulaw(),"m")
    
    X0 = ode.make_input(h=0,v=0,m=m0,u=1)
    TrajIG = integ.integrate_dense(X0,60,1000,StopFunc)
    
    '''
    Single phase formualtion has a singular arc
    '''
    
    ##############################################################################
    phase = ode.phase("LGL3",TrajIG,128)
    phase.setAutoScaling(True)
    phase.setUnits(units)
    
    phase.addBoundaryValue("Front",["h","v","m","t"],TrajIG[0][0:4])
    phase.addLUVarBound("Path","u",0.0,1.0,1.0)
    phase.addValueObjective("Back","h",-1.0)
    phase.addBoundaryValue("Back",["v","m"],[0,mf])
    phase.optimize()
    Traj = phase.returnTraj()
    ###############################################################################
    
    '''
    Multi phase formualtion defines control using path constraint on middle
    phase.
    '''
    
    n = int(len(TrajIG)/3)
    
    TrajIG1 = TrajIG[0:n]
    TrajIG2 = TrajIG[n:2*n]
    TrajIG3 = TrajIG[2*n:-1]
    
    phase1 = ode.phase("LGL3",TrajIG1,32)
    phase1.addBoundaryValue("Front",["h","v","m","t"],TrajIG[0][0:4])
    phase1.addBoundaryValue("Path","u",1.0)
    
    phase2 = ode.phase("LGL3",TrajIG2,32)
    # PathCon makse Control splines redundant for LGL>3
    phase2.setControlMode("NoSpline") 
    phase2.addLUVarBound("Path","u",0.0,1.0,1.0)
    phase2.addEqualCon("Path",PathCon(sigma,c,h_ref,Tmag, g),["h","v","m","u"])
    
    phase3 = ode.phase("LGL3",TrajIG3,32)
    phase3.addBoundaryValue("Path","u",0)
    phase3.addBoundaryValue("Back",["v","m"],[0,mf])
    phase3.addValueObjective("Back","h",-1.0)
    
    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase1)
    ocp.addPhase(phase2)
    ocp.addPhase(phase3)
    
    ocp.addForwardLinkEqualCon(phase1,phase3,["h","v","m","t"])
    
    phase1.addLowerDeltaTimeBound(0)
    phase2.addLowerDeltaTimeBound(0)
    phase3.addLowerDeltaTimeBound(0)
    
    
    ## Set each phase's units
    phase1.setUnits(units)
    phase2.setUnits(units)
    phase3.setUnits(units)
    
    # Enable autoscaling for ocp and all constituent phases
    ocp.setAutoScaling(True,True)

    ocp.setThreads(8,8)
    ocp.optimize()
    
    ##############################################################################
    
    Traj2 = phase1.returnTraj() + phase2.returnTraj() + phase3.returnTraj()
    
    fig,axs = plt.subplots(4,1)
    
    
    Plot(axs,TrajIG,"Initial Guess")
    Plot(axs,Traj,"Single Phase")
    Plot(axs,Traj2,"Multi Phase")
    
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    axs[3].grid(True)
    
    axs[3].set_xlabel("t (s)")
    
    axs[0].set_ylabel("h (ft)")
    axs[1].set_ylabel("v (ft/s)")
    axs[2].set_ylabel("m (slug)")
    axs[3].set_ylabel("T (lb)")
    
    axs[1].legend()
    plt.show()


