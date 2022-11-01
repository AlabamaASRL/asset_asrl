import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes = oc.ControlModes


g0 = 32.2 
W  = 203000

Lstar = 10000.0   ## feet
Tstar = 60.0      ## sec
Mstar = 1         ## slugs

Vstar   = Lstar/Tstar
Fstar   = Mstar*Lstar/(Tstar**2)
Astar   = Lstar/(Tstar**2)
Rhostar = Mstar/(Lstar**3)
BTUstar = 778.0*Lstar*Fstar
Mustar  = (Lstar**3)/(Tstar**2)
sigmastar = Mstar/Lstar

rho0  =.002378         /Rhostar
h_ref = 23800          /Lstar
g     = g0             /Astar
Tmag  = 200            /Fstar
c     = 1580.94        /Vstar
sigma = 5.4915e-5      /sigmastar

m0   = 3
mf   = 1



class GoddardRocket(oc.ode_x_u.ode):
    def __init__(self,sigma,c,h_ref,Tmag, g):
        ############################################################
        args  = oc.ODEArguments(3,1)
        h,v,m = args.XVec().tolist()
        u = args.UVar(0)
        
        hdot=v
        
        vdot = (u*Tmag - sigma*(v**2)*vf.exp(-h/h_ref))/m -g
        
        mdot = -u*Tmag/c
       
        ode = vf.stack(hdot,vdot,mdot)
        ##############################################################
        super().__init__(ode,3,1)
 
def PathCon(sigma,c,h_ref,Tmag, g):
    h,v,m,u = Args(4).tolist()
    t1 = (u*Tmag - sigma*(v**2)*vf.exp(-h/h_ref)) -g*m
    t2 = (m*g/( 1 + 4*(c/v) +2*(c/v)**2 ))*( c*c*(1+v/c)/(h_ref*g) -1.0 -2.0*c/v )
    return t1-t2
       

def Plot(axs,Traj):
    T = np.array(Traj).T
    
    
    axs[0].plot(T[3],T[0])
    axs[1].plot(T[3],T[1])
    axs[2].plot(T[3],T[2])
    axs[3].plot(T[3],T[4])
    
    

        




def Ulaw():
    m = Args(1)[0]
    return vf.ifelse(m>mf,1,0)
    
def StopFunc(x): return (x[1]<0)
    


    

ode = GoddardRocket(sigma,c,h_ref,Tmag, g)

integ = ode.integrator(.01,Ulaw(),[2])

X0 = np.zeros((5))
X0[0]=0
X0[1]=0
X0[2]=m0
X0[4]=1



TrajIG = integ.integrate_dense(X0,60/Tstar,1000,StopFunc)

##############################################################################
phase = ode.phase("LGL3",TrajIG,128)
phase.addBoundaryValue("Front",range(0,4),TrajIG[0][0:4])
phase.addLUVarBound("Path",4,0.0,1.0,1.0)
phase.addValueObjective("Back",0,-1.0)
phase.addBoundaryValue("Back",[1,2],[0,mf])
phase.optimize()
Traj = phase.returnTraj()
###############################################################################

n = int(len(TrajIG)/3)

TrajIG1 = TrajIG[0:n]
TrajIG2 = TrajIG[n:2*n]
TrajIG3 = TrajIG[2*n:-1]



phase1 = ode.phase("LGL3",TrajIG1,32)
phase1.addBoundaryValue("Front",range(0,4),TrajIG[0][0:4])
phase1.addBoundaryValue("Path",[4],[1])

phase2 = ode.phase("LGL3",TrajIG2,32)
phase2.addLUVarBound("Path",4,0.0,1.0,1.0)
phase2.addEqualCon("Path",PathCon(sigma,c,h_ref,Tmag, g),[0,1,2,4])

phase3 = ode.phase("LGL3",TrajIG3,32)
phase3.addBoundaryValue("Path",[4],[0])
phase3.addBoundaryValue("Back",[1,2],[0,mf])
phase3.addValueObjective("Back",0,-1.0)

ocp = oc.OptimalControlProblem()

ocp.addPhase(phase1)
ocp.addPhase(phase2)
ocp.addPhase(phase3)

ocp.addForwardLinkEqualCon(phase1,phase3,range(0,4))

phase1.addLowerDeltaTimeBound(0)
phase2.addLowerDeltaTimeBound(0)
phase3.addLowerDeltaTimeBound(0)

ocp.Threads=8
ocp.optimizer.PrintLevel=1
ocp.optimize()


Traj2 = phase1.returnTraj() + phase2.returnTraj() + phase3.returnTraj()

fig,axs = plt.subplots(4,1)


Plot(axs,TrajIG)
Plot(axs,Traj)
Plot(axs,Traj2)

plt.show()


