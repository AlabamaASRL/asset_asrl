import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

'''

Minumum time docking with an out of control target taken from:

J. Michael, K. Chudej, M. Gerdts, and J. Panncek, Optimal Rendezvous
Path Planning to an Uncontrolled Tumbling Target, in Proceedings of the IFAC
ACA2013 Conference, W¨urzburg, Germany, Sept. 2013.

I have reformulated the quaternions so that they transform from body to global frame, and
changed the inital conditions to reflect this. Ive also reforumalted thrust controls to be
in the body frame rather than global frame and updated path constraint accordingly. 

I also solve this problem in two ways. In the first (Form1) I let the freely spinning
target be part of the state variables as in the reference. However, since its
motion is uncontrolled we can actually eliminate it from the ODE and turn the
rendezvous constraint into a time dependent boundary condtion using an interpoaltion
table. This problem (Form2) is much smaller and solves faster yielding the same results.

'''
norm      = np.linalg.norm
vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
solvs     = ast.Solvers

##############################################################################
Lstar   =  10.0         ## m
Tstar   =  30.0         ## sec
Mstar   =  10.0         ## kgs
  


Astar   =  Lstar/Tstar**2
Vstar   =  Lstar/Tstar
Rhostar =  Mstar/Lstar**3
Estar   =  Mstar*(Vstar**2)
Mustar  =  (Lstar**3)/(Tstar**2)
Fstar   =  Astar*Mstar
Istar   =  Mstar*Lstar*Lstar

a       = 7071000   /Lstar
mu      = 3.986e14  /Mustar
n       = np.sqrt(mu/a**3)
m       = 100/Mstar

MaxThrust = .1/Fstar
MaxTorque = 1 /(Fstar*Lstar)


Srad  = 1/Lstar
Udvec = np.array([0,1.01,0])/Lstar

Ivec  = np.array([1000,2000,1000]) / (Mstar*Lstar*Lstar)



##############################################################################
class RelDynModel(oc.ODEBase):
    def __init__(self,I1,I2,n,m):
        Xvars = 20
        Uvars = 6
        ############################################################
        args = oc.ODEArguments(Xvars,Uvars)
        
        ## Position Velocity of Servicer
        X = args.XVec().head3()
        V = args.XVec().segment3(3)
        
        ## Body Frame quat and angvel of Servicer
        q = args.XVec().segment(6,4).normalized()
        w = args.XVec().segment3(10)
        
        ## Body Frame quat and angvel of Target
        p   = args.XVec().segment(13,4).normalized()
        phi = args.XVec().segment3(17)
        
        ## Body Frame Thrust and Torque of Servicer
        Thrust = args.UVec().head3()
        Torque = args.UVec().tail3()
        
        Xdot = V
        
        Vdoto = vf.stack([2*n*V[1] + (3*n**2)*X[0],
                        -2*n*V[0],
                        -(n**2)*X[2]]) 
        
        
        Thrust_global = vf.quatRotate(q,Thrust)
        Vdot = Vdoto + Thrust_global/m
        
        qdot  = vf.quatProduct(q,w.padded_lower(1))/2.0
        L1     = w.cwiseProduct(I1)
        wdot  = (L1.cross(w) + Torque).cwiseQuotient(I1)
        
        pdot    = vf.quatProduct(p,phi.padded_lower(1))/2.0
        L2      = phi.cwiseProduct(I2)
        phidot  = (L2.cross(phi)).cwiseQuotient(I2)
        
        ode = vf.stack([Xdot,Vdot,qdot,wdot,pdot,phidot])

        ######################################################################
        super().__init__(ode,Xvars,Uvars)
        
        

##############################################################################
class RelDynModel2(oc.ODEBase):
    def __init__(self,I1,n,m):
        Xvars = 13
        Uvars = 6
        ############################################################
        args = oc.ODEArguments(Xvars,Uvars)
        
        ## Position Velocity of Servicer
        X = args.XVec().head3()
        V = args.XVec().segment3(3)
        
        ## Body Frame quat and angvel of Servicer
        q = args.XVec().segment(6,4).normalized()
        w = args.XVec().segment3(10)
       
        
        ## Body Frame Thrust and Torque of Servicer
        Thrust = args.UVec().head3()
        Torque = args.UVec().tail3()
        
        
        Xdot = V
        Vdoto = vf.stack([2*n*V[1] + (3*n**2)*X[0],
                        -2*n*V[0],
                        -(n**2)*X[2]]) 
        
        
        Thrust_global = vf.quatRotate(q,Thrust)
        Vdot = Vdoto + Thrust_global/m
        
        
        qdot  = vf.quatProduct(q,w.padded_lower(1))/2.0
        L1     = w.cwiseProduct(I1)
        wdot  = (L1.cross(w) + Torque).cwiseQuotient(I1)
        
       
        ode = vf.stack([Xdot,Vdot,qdot,wdot])

        ######################################################################
        super().__init__(ode,Xvars,Uvars)        

##############################################################################
class TorqueFree(oc.ODEBase):
    def __init__(self,I2):
        Xvars = 7
        Uvars = 0
        ############################################################
        args = oc.ODEArguments(Xvars,Uvars)
        
        ## Body Frame quat and angvel of Target
        p   = args.XVec().head(4).normalized()
        phi = args.XVec().segment3(4)
        
        pdot    = vf.quatProduct(p,phi.padded_lower(1))/2.0
        L2      = phi.cwiseProduct(I2)
        phidot  = (L2.cross(phi)).cwiseQuotient(I2)
        
        ode = vf.stack([pdot,phidot])

        ######################################################################
        super().__init__(ode,Xvars,Uvars)

##############################################################################
        
def RendCon(ud):

    X,V,q,w,p,phi = Args(20).tolist( [(0,3),(3,3),(6,4),
                                      (10,3),(13,4),(17,3)] )
    
    q=q.normalized()
    p =p.normalized()
    
    Xdq =  vf.quatRotate(q,ud)
    vdq =  vf.quatRotate(q,w)
    Vdq =  -Xdq.cross(vdq)
    
    Xdp = vf.quatRotate(p,ud)
    vdp = vf.quatRotate(p,phi)
    Vdp = -Xdp.cross(vdp)
    
    return vf.stack([X + Xdq - Xdp, V + Vdq - Vdp])

def RendCon2(ud,tab):
    
    func = oc.InterpFunction(tab,range(0,7)).vf()
    
    X,V,q,w,t = Args(14).tolist( [(0,3),(3,3),(6,4),(10,3),(13,1)] )
    
    return RendCon(ud)(X,V,q,w,func(t))

##############################################################################

    
    
def Animate(Traj):
    import matplotlib.animation as animation

    TT = np.array(Traj).T


    fig = plt.figure()

    ax3 = fig.add_subplot(111, projection='3d')


    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)*Srad
    y = np.sin(u)*np.sin(v)*Srad
    z = np.cos(v)*Srad
    ax3.plot_surface(x, y, z, color="cyan",alpha=0.2)


    ax3.set_xlim3d([-.75, .75])
    ax3.set_ylim3d([-1, .5])
    ax3.set_zlim3d([-.75, .75])

    xhat1, = ax3.plot([],[],[], color='r',marker='o',label=r'$\hat{x}$')
    yhat1, = ax3.plot([],[],[], color='g',marker='o',label=r'$\hat{y}$')
    zhat1, = ax3.plot([],[],[], color='b',marker='o',label=r'$\hat{z}$')


    xhat2, = ax3.plot([],[],[], color='r',marker='o')
    yhat2, = ax3.plot([],[],[], color='g',marker='o')
    zhat2, = ax3.plot([],[],[], color='b',marker='o')

    trace,  = ax3.plot([],[],[])

    def init():
        
        trace.set_data(np.array([]),np.array([]))
        trace.set_3d_properties(np.array([]),'z')
        
        xhat1.set_data(np.array([]),np.array([]))
        xhat1.set_3d_properties(np.array([]),'z')
        
        yhat1.set_data(np.array([]),np.array([]))
        yhat1.set_3d_properties(np.array([]),'z')
        
        zhat1.set_data(np.array([]),np.array([]))
        zhat1.set_3d_properties(np.array([]),'z')
        
        
        xhat2.set_data(np.array([]),np.array([]))
        xhat2.set_3d_properties(np.array([]),'z')
        
        yhat2.set_data(np.array([]),np.array([]))
        yhat2.set_3d_properties(np.array([]),'z')
        
        zhat2.set_data(np.array([]),np.array([]))
        zhat2.set_3d_properties(np.array([]),'z')
        
        
        return trace,xhat1,yhat1,zhat1,xhat2,yhat2,zhat2

    def animate(i):
        
        trace.set_data(TT[0][0:i],TT[1][0:i])
        trace.set_3d_properties(TT[2][0:i],'z')
        
        vecs = R.from_quat(Traj[i][6:10]).as_matrix().T*Srad
        
        x,y,z = Traj[i][0:3]
        
        xhat1.set_data(np.array([x,x+vecs[0][0]]),np.array([y,y+vecs[0][1]]))
        xhat1.set_3d_properties(np.array([z,z+vecs[0][2]]),'z')
        
        yhat1.set_data(np.array([x,x+vecs[1][0]]),np.array([y,y+vecs[1][1]]))
        yhat1.set_3d_properties(np.array([z,z+vecs[1][2]]),'z')
        
        zhat1.set_data(np.array([x,x+vecs[2][0]]),np.array([y,y+vecs[2][1]]))
        zhat1.set_3d_properties(np.array([z,z+vecs[2][2]]),'z')
        
        
        vecs = R.from_quat(Traj[i][13:17]).as_matrix().T*Srad
        
        x,y,z = [0,0,0]
        
        xhat2.set_data(np.array([x,x+vecs[0][0]]),np.array([y,y+vecs[0][1]]))
        xhat2.set_3d_properties(np.array([z,z+vecs[0][2]]),'z')
        
        yhat2.set_data(np.array([x,x+vecs[1][0]]),np.array([y,y+vecs[1][1]]))
        yhat2.set_3d_properties(np.array([z,z+vecs[1][2]]),'z')
        
        zhat2.set_data(np.array([x,x+vecs[2][0]]),np.array([y,y+vecs[2][1]]))
        zhat2.set_3d_properties(np.array([z,z+vecs[2][2]]),'z')
        
        
        return trace,xhat1,yhat1,zhat1,xhat2,yhat2,zhat2


    ani = animation.FuncAnimation(fig, animate, frames=len(TT[0]),
                                  interval=60, blit=True, init_func=init,repeat_delay=5000)


    ax3.set_xlabel(r'$X$')
    ax3.set_ylabel(r'$Y$')
    ax3.set_zlabel(r'$Z$')
    ax3.legend()

    fig.set_size_inches(7.5, 7.5, forward=True)
    plt.show()
    
    
    
    
def Form1():
    

    
    X0 = np.zeros((27))
    X0[1] = -10.0/Lstar
    X0[9]=1
    
    X0[13] = .05
    X0[16] = np.sqrt(1 - X0[13]**2)
    X0[18] = .0349*Tstar
    X0[19] = .017453*Tstar
    
    
    ## Rough initial guess for trajectory
    ## commment out optimize to see what it looks like

    X0[21]  = -MaxThrust
    X0[22]  =  MaxThrust
    X0[26]  = -MaxTorque/4
     
    ode = RelDynModel(Ivec,Ivec,n,m)
    integ = ode.integrator(.01)
    
    IG = integ.integrate_dense(X0,200/Tstar,1000)
    
    
    phase = ode.phase('LGL3',IG,384)
    phase.setControlMode('BlockConstant')
    
    phase.addBoundaryValue("Front",range(0,21),X0[0:21])
    phase.addLUVarBounds('Path',[21,22,23],-MaxThrust,MaxThrust,.1)
    phase.addLUVarBounds('Path',[24,25,26],-MaxTorque,MaxTorque,1)
    phase.addLowerNormBound('Path',[0,1,2],2*Srad,1.0)
    phase.addEqualCon('Back',RendCon(Udvec),range(0,20))
    phase.addDeltaTimeObjective(1.0)
    phase.optimizer.set_BoundFraction(.995)
    phase.optimizer.set_PrintLevel(1)
    
    phase.optimize()
   
    Traj = phase.returnTraj()
    

    tf = Traj[-1][20]*Tstar

    print("Final Time: ",tf," s")

    Animate(Traj)
    
def Form2():
    
    ode_torquefree = TorqueFree(Ivec)
    
    integ_torquefree = ode_torquefree.integrator(.01)
    integ_torquefree.Adaptive=True
    
    ## generate target trajectory for boundary condition
    SimTime = 600/Tstar
    
    TargetTraj_IS = np.zeros((8))
    
    TargetTraj_IS[0]=.05
    TargetTraj_IS[3] = np.sqrt(1 - TargetTraj_IS[0]**2)
    TargetTraj_IS[5] = .0349*Tstar
    TargetTraj_IS[6] = .017453*Tstar
    
    TargetTraj = integ_torquefree.integrate_dense(TargetTraj_IS,SimTime,2000)
    TargetTab  = oc.LGLInterpTable(ode_torquefree.vf(),7,0,TargetTraj)
    
    #########################################################################
    
    X0 = np.zeros((20))
    X0[1] = -10.0/Lstar
    X0[9]=1
    
   
    X0[14]  = -MaxThrust
    X0[15]  =  MaxThrust
    X0[19]  = -MaxTorque/4
     
    ode = RelDynModel2(Ivec,n,m)
    
    integ = ode.integrator(.01)
    integ.Adaptive = True
    
    Traj = integ.integrate_dense(X0,200/Tstar,1000)
    
    phase = ode.phase('LGL3',Traj,384)
    phase.setControlMode('BlockConstant')
    phase.addBoundaryValue("Front",range(0,14),X0[0:14])
    phase.addLUVarBounds('Path',[14,15,16],-MaxThrust,MaxThrust,.1)
    phase.addLUVarBounds('Path',[17,18,19],-MaxTorque,MaxTorque,1)
    phase.addLowerNormBound('Path',[0,1,2],2*Srad,1.0)
    phase.addEqualCon('Last',RendCon2(Udvec,TargetTab),range(0,14))
    phase.addUpperDeltaTimeBound(SimTime)
    phase.addDeltaTimeObjective(1.0)
    phase.optimizer.set_BoundFraction(.995)
    
    phase.optimizer.set_PrintLevel(1)
    phase.optimize()
    
    Traj = phase.returnTraj()
    
    tf = Traj[-1][13]*Tstar

    print("Final Time: ",tf," s")    

    
if __name__ == "__main__":

    Form2()
    Form1()




    







