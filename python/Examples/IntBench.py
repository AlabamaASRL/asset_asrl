import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import time


vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments

class ODE(oc.ODEBase):
    
    def __init__(self):
        
        Xt = oc.ODEArguments(4)
        
        x,y,vx,vy = Xt.XVec().tolist()
        
        vxdot = -x -2.0*x*y
        vydot = y**2 -y -x**2
        
        ode = vf.stack(Xt.segment2(2),vxdot,vydot)
        #ode = oc.HeyokaODE()
        super().__init__(ode,4)
        
        
 

 
    
 
    
 
 
ode = ODE()
ode =oc.Heyoka.ode(1)

integ = ode.integrator("DP87",1)
integ.setAbsTol(1.0e-14)
        
IState = np.zeros((5))
IState[0]=0
IState[1]= 0.2587703282931232
IState[2]=-0.2525875586263492
IState[3]=-0.2178423952983717

def Event():
    Xt = Args(5)
    return Xt[0]

for i in range(0,20):
    t00 = time.perf_counter()
    Traj = integ.integrate(IState,2000)
    tff = time.perf_counter()
    print((tff-t00)*1000)

#
#

t00 = time.perf_counter()
Trajt,Events = integ.integrate(IState,2000,[(Event(),1,0)])
tff = time.perf_counter()
print((tff-t00)*1000)


PT = np.array(Events[0]).T
TT = np.array(Traj).T

plt.plot(TT[0],TT[1])
plt.scatter(PT[0],PT[1],color='k')
plt.show()

