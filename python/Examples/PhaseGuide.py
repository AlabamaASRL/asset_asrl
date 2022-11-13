import asset_asrl as ast
import numpy as np
import matplotlib.pyplot as plt


vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
ODEArgs   = oc.ODEArguments

class TwoBodyLTODE(oc.ODEBase):
    
    def __init__(self,mu,MaxLTAcc):
        
        XVars = 6
        UVars = 3
        
        
        XtU = oc.ODEArguments(XVars,UVars)
        
        R,V  = XtU.XVec().tolist([(0,3),(3,3)])
        U = XtU.UVec()
        
        
        G = -mu*R.normalized_power3()
        Acc = G + U*MaxLTAcc
        
        Rdot = V
        Vdot = Acc
        
        ode = vf.stack([Rdot,Vdot])
        
        super().__init__(ode,XVars,UVars)





############# Intro ########################
'''
In this section, we will explain the usage of the ODEPhase object
'''


def ULaw(throttle):
    V = Args(3)
    return V.normalized()*throttle
def RStop(rmax):
    X = Args(10)
    return X.head3().norm()-rmax

ode = TwoBodyLTODE(1,.02)

integULaw   = ode.integrator("DOPRI87",.1,ULaw(0.8),[3,4,5])


r  = 1.0
v  = 1.0
t0 = 0.0
tf = 100.0


X0t0U0 = np.zeros((10))
X0t0U0[0]=r
X0t0U0[4]=v
X0t0U0[6]=t0        



TrajULaw,Events   = integULaw.integrate_dense(X0t0U0,tf,[(RStop(2),0,1),])

print(len(TrajULaw))

def CircularOrbit(r):
    R,V = Args(6).tolist([(0,3),(3,3)])
    
    eq1 = R.norm()-r
    eq2 = V.norm() - np.sqrt(1/r)
    eq3 = V.dot(R)
    
    return vf.stack(eq1,eq2,eq3)




phase = ode.phase("LGL3",TrajULaw,128)

phase.addBoundaryValue("Front",range(0,7),X0t0U0[0:7])

phase.addLUNormBound("Path",range(7,10),.001,1.0)

phase.addEqualCon("Back",CircularOrbit(2.0),range(0,6),[],[])

#phase.addIntegralObjective(Args(3).norm(),[7,8,9])

phase.addDeltaTimeObjective(.1)
phase.setJetJobMode("")

phase.optimizer.set_OptLSMode("L1")
phase.optimizer.BoundFraction =.997







####################

TT = np.array(TrajULaw).T
plt.plot(TT[0],TT[1],label='80% Prograde Throttle')

TT = np.array(phase.returnTraj()).T
plt.plot(TT[0],TT[1],label='Solved')


plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.axis("Equal")
plt.grid(True)
        
plt.show()

############# Transcriptions ##############


#############  ##############




