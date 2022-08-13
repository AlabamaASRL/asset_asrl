import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes = oc.ControlModes

from DerivChecker import FDDerivChecker


'''
Vanderpol Osscilator Optimization Problem Taken From 

https://openmdao.github.io/dymos/examples/vanderpol/vanderpol.html

'''

def Plot(Traj):
    TT = np.array(Traj).T
    plt.plot(TT[0]*np.cos(TT[1]),TT[0]*np.sin(TT[1]))
    


class OrbitRaise(oc.ode_x_u.ode):
    def __init__(self,accval):
        ############################################################
        args  = oc.ODEArguments(5,2)
        X = args.XVec()
        U = args.UVec()
        
        r       = X[0]
        vr      = X[2]
        vtheta  = X[3]
        
        ur      = args[6]
        utheta  = args[7]
        
        drdt      = vr
        dthetadt  = vtheta/r
        dvrdt     = (vtheta**2)/r - 1.0/(r**2) + accval*ur
        dvthetadt = -(vr.dot(vtheta)/r) + accval*utheta
        ddvdt     = accval*U.norm()
        
        
        ode = vf.StackScalar([drdt,dthetadt,dvrdt,dvthetadt,ddvdt])
        ##############################################################
        super().__init__(ode,5,2)
        
        

rt0 = 1.0
vt0 = 1.0

rtf = 3.0
vtf = np.sqrt(1.0/3.0)

tf = 7.0
        

IState = np.zeros((8))
IState[0]=rt0
IState[2]=.00
IState[3] = vt0
IState[6] =.0001
IState[7] =.8



ode = OrbitRaise(.1)
integ = ode.integrator(.01)
TrajIG = integ.integrate_dense(IState,tf,1000)

IOrb = [[rt0,theta] for theta in np.linspace(0,2*np.pi,1000)]
TOrb = [[rtf,theta] for theta in np.linspace(0,2*np.pi,1000)]


Plot(TrajIG)
Plot(IOrb)
Plot(TOrb)
plt.show()

phase = ode.phase(Tmodes.LGL3,TrajIG,64)
phase.addBoundaryValue(PhaseRegs.Front,range(0,6),IState[0:6])
phase.addLUNormBound(PhaseRegs.Path,[6,7],.001,1.0,1.0)
phase.addValueObjective(PhaseRegs.Back,4,1.0)
phase.addBoundaryValue(PhaseRegs.Back,[0,2,3],[rtf,0.0,vtf])
phase.optimizer.OptLSMode = ast.Solvers.LineSearchModes.L1
phase.optimizer.MaxLSIters = 2
phase.optimizer.PrintLevel =1
phase.Threads=8
phase.optimizer.QPThreads =8

#phase.optimizer.KKTtol = 1.0e-9

phase.optimize()

TrajConv = phase.returnTraj()
Plot(TrajConv)
Plot(IOrb)
Plot(TOrb)
plt.show()

TT = np.array(TrajConv).T

plt.plot(TT[5],TT[4])
plt.show()














