import asset_asrl as ast
import matplotlib.pyplot as plt
import numpy as np


vf = ast.VectorFunctions
Args = vf.Arguments
oc = ast.OptimalControl

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


def ULaw(throttle):
    V = Args(3)
    return V.normalized()*throttle


ode = TwoBodyLTODE(1,.01)




r  = 1.0
v  = 1.1
t0 = 0.0
tf = 20.0


X0t0U0 = np.zeros((10))
X0t0U0[0]=r
X0t0U0[4]=v
X0t0U0[6]=t0        

integULaw   = ode.integrator(.1,ULaw(0.8),[3,4,5])
integULaw.setAbsTol(1.0e-14)

TrajULaw   = integULaw.integrate_dense(X0t0U0,tf,3000)
Uts = [ list(T[7:10]) +[T[6]] for T in TrajULaw  ]



Tab1 = oc.LGLInterpTable(ode.vf(),6,3,TrajULaw)

Tab2 = oc.LGLInterpTable(3,Uts,2000-1)

integTab1 = ode.integrator(.1,Tab1)
integTab2 = ode.integrator(.1,Tab2,range(0,3))


Traj1   = integTab1.integrate_dense(X0t0U0,tf)
Traj2   = integTab2.integrate_dense(X0t0U0,tf)




print(Traj1[-1]-TrajULaw[-1])
print(Traj2[-1]-TrajULaw[-1])




####################
#TT = np.array(TrajNoULaw).T
#plt.plot(TT[0],TT[1],label='TrajNoULaw',marker='o')

TT = np.array(TrajULaw).T
plt.plot(TT[0],TT[1],label='TrajULaw',marker='o')

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.axis("Equal")
plt.grid(True)
        
plt.show()
