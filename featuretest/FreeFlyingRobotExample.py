import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

'''
Example derived from 

https://arxiv.org/pdf/1905.11898.pdf

Control the motion of a free flying robot between two points

'''

class FreeFlyingRobotODE(oc.ODEBase):
    def __init__(self,alpha,beta):
        Xvars = 6
        Uvars = 4
        ############################################################
        args   = oc.ODEArguments(6,4)
        xy     = args.XVec().head(2)
        xydot  = args.XVec().segment2(2)
        theta = args.XVar(4)
        omega = args.XVar(5)
        
        u     =  args.UVec()
    
        vscale = u[0]-u[1]+u[2]-u[3]
        vxydot = vf.stack([vf.cos(theta),vf.sin(theta)])*vscale
        
        theta_dot=omega
        omega_dot = (u[0]-u[1])*alpha + (u[3]-u[2])*beta
        
        ode = vf.stack([xydot,vxydot,theta_dot,omega_dot])
        ##############################################################
        super().__init__(ode,Xvars,Uvars)

   


if __name__ == "__main__":


    ode = FreeFlyingRobotODE(.2,.2)
    
    t0 = 0
    tf = 12
    
    X0 =np.array([-10,-10,0,0,np.pi/2.0,0, 0])
    XF =np.array([0,0,0,0,0,0,tf])
    
    IG = []
    ts = np.linspace(0,tf,100)
    
    for t in ts:
        T = np.zeros((11))
        T[0:7] = X0[0:7] + ((t-t0)/(tf-t0))*( XF[0:7]- X0[0:7])
        T[7:11] = np.ones((4))*.50
        IG.append(T)
    
    
    phase = ode.phase("LGL3",IG,64)
    #phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("Front",range(0,7),X0)
    phase.addBoundaryValue("Back" ,range(0,7),XF)
    phase.addLUVarBounds("Path"   ,range(7,11),0.0,1.0,1)
    phase.addIntegralObjective(Args(4).sum(),range(7,11))
    phase.optimizer.set_PrintLevel(0)
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_MaxLSIters(2)
    phase.optimizer.set_tols(1.0e-9,1.0e-9,1.0e-9)
    phase.optimizer.MaxAccIters = 100
    phase.optimizer.deltaH=1.0e-6
    phase.AdaptiveMesh=True
    phase.MeshTol = 1.0e-6
    phase.MeshErrorEstimator='deboor'
    phase.optimize()
    
    plt.plot(phase.MeshTimes,phase.MeshDistInt)
    #plt.yscale("log")
    plt.show()
    TrajConv = phase.returnTraj()
    
    ##########################################################
    IGT = np.array(TrajConv).T
    
    plt.plot(IGT[0],IGT[1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid("True")
    plt.show()
    plt.plot(IGT[6],IGT[7]-IGT[8])
    plt.plot(IGT[6],IGT[9]-IGT[10])
    plt.show()
    ##########################################################




