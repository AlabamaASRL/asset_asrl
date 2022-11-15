import asset_asrl as ast
import numpy as np
import matplotlib.pyplot as plt


vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
ODEArgs   = oc.ODEArguments







############# Intro ########################
'''
In this section, we will explain the usage of ASSETs phase object. Blah Blah
'''


'''
As introduction, we will walk through using phase to optimize a low-trhrust
transfer between two-circular orbits using our TwoBodyLTODE model. In particular,
we will attempt to compute a tarsnfer from a fixed circular orbit at radius 1,
to a circular orbit at r = 2.
'''

class Brachistochrone(oc.ODEBase):
    
    def __init__(self,g):
        
        XVars = 3
        UVars = 1
        
        
        XtU = oc.ODEArguments(XVars,UVars)
        
        x,y,v = XtU.XVec().tolist()
        theta = XtU.UVar(0)
        
        xdot = vf.sin(theta)*v
        ydot = -1.0*vf.cos(theta)*v
        zdot = g*vf.cos(theta)     
        ode = vf.stack([xdot,ydot,zdot])
        
        super().__init__(ode,XVars,UVars)

g=9.81
ode = Brachistochrone(g)

x0=0
y0=10
v0=0
theta0 = 1.0

xf = 10
yf = 5

tf =1

ts = np.linspace(0,tf,100)

Xs =[]
for t in ts:
    X= np.zeros((5))
    X[0] = x0 + (xf-x0)*t/tf
    X[1] = y0 + (yf-y0)*t/tf
    X[2] = g*t*np.cos(theta0)
    X[3]=t
    X[4]=theta0
    Xs.append(X)

phase = ode.phase("LGL3",Xs,64)
phase.addBoundaryValue("Front",range(0,4),[x0,y0,v0,0])
phase.addLUVarBound("Path",4,-0.1,2.00)

phase.addBoundaryValue("Back",[0,1],[xf,yf])
phase.addDeltaTimeObjective(1.0)
#phase.optimizer.set_OptLSMode("L1")
phase.optimize()

Traj = phase.returnTraj()

TT = np.array(Traj).T

plt.plot(TT[0],TT[1])
plt.show()

plt.plot(TT[3],TT[4])
plt.show()
