import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from inspect import getmembers, isfunction, isclass


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags


F1 = Args(3).vf()/Args(3).norm()
F2 = Args(3).normalized()

#F1.rpt([1,1,1],1000000)
#F2.rpt([1,1,1],1000000)
#input("s")

#ast.PyMain()

def plot(T):
    IGT = np.array(T).T
    
    plt.plot(IGT[0],IGT[1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid("True")
    plt.show()
    '''
    plt.plot(IGT[6],IGT[0])
    plt.plot(IGT[6],IGT[1])
    plt.show()
    
    plt.plot(IGT[6],IGT[2])
    plt.plot(IGT[6],IGT[3])
    plt.show()
    
    plt.plot(IGT[6],IGT[4])
    plt.plot(IGT[6],IGT[5])
    plt.show()
    '''
    plt.plot(IGT[6],IGT[7]-IGT[8])
    plt.plot(IGT[6],IGT[9]-IGT[10])

    plt.show()



Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags





ode = ast.FreeFly.ode(.2,.2)


def FreeFly(alpha,beta):
    args  = Args(11)
    theta = args[4]
    omega = args[5]
    u = args.tail(4)
    xdot = args.segment_2(2)
    vscale = vf.SumElems([u[0],u[1],u[2],u[3]],
                         [1,     -1,   1 ,-1])
    
    vdot =vf.StackScalar([vf.cos(theta),vf.sin(theta)])*vscale
    theta_dot=omega
   
    omega_dot= vf.SumElems([u[0],u[1],u[2],u[3]],
                           [alpha, -alpha, -beta ,beta])
    
    ode = vf.Stack([xdot,vdot,vf.StackScalar([theta_dot,omega_dot])])
    return ode

IG = []
ts = np.linspace(0,12,100)

for t in ts:
    T = np.zeros((11))
    T[0] = -10 + t*10/12.0
    T[4] = np.pi/2.0 - t*(np.pi/2.0)/12.0
    T[1]=T[0]
    T[6]=t
    T[7] =.50
    T[8] =.50
    T[9] =.50
    T[10]=.51
    IG.append(T)

#FreeFly(.2,.2).rpt(IG[0],1000000)
#input("s")
phase=oc.ode_x_x.ode(FreeFly(.2,.2),6,4,0).phase(Tmodes.LGL7)
phase=ast.FreeFly.phase(ode,Tmodes.LGL7)
#phase=oc.ode_x_x.ode(ode,6,4,0).phase(Tmodes.LGL5)

phase.setControlMode(oc.HighestOrderSpline)
phase.setTraj(IG,96)
phase.addBoundaryValue(PhaseRegs.Front,range(0,7),[-10,-10,0,0,np.pi/2.0,0,0])
phase.addBoundaryValue(PhaseRegs.Back ,range(0,7),[0,0,0,0,0,0,12])
phase.addLUVarBounds(PhaseRegs.Path,[7,8,9,10],0.001,1.0,1)
#phase.EnableVectorization=True
obj = (Args(4)[0]+Args(4)[1]) + (Args(4)[2]+Args(4)[3])
phase.addIntegralObjective(obj,[7,8,9,10])

phase.test_threads(1,8,5000)
input("s")

phase.optimizer.OptLSMode = ast.L1
phase.Threads=8
phase.optimizer.QPThreads=6
phase.optimizer.BoundFraction=.997
phase.optimizer.MaxLSIters =1
phase.optimizer.deltaH = 1.0e-8
phase.optimizer.incrH = 8
phase.optimizer.decrH = .3
phase.optimizer.FastFactorAlg=True
phase.optimize()
plot(phase.returnTraj())


