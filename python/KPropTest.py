import asset as ast
import spiceypy as spice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
################################################################################
# Setup
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
TModes = oc.TranscriptionModes



class Kepler(oc.ode_6.ode):
    def __init__(self, mu):
        Xvars = 6
       
        ############################################################
        args = oc.ODEArguments(Xvars)
        r = args.head3()
        v = args.segment3(3)
        g = r.normalized_power3() * (-mu)
        ode = vf.stack([v, g])
        #############################################################
        super().__init__(ode, Xvars)
        
        
        
       
        
        
X0 = np.zeros((6))
X0[0]=1.
X0[2]=.01
X0[4]=1.25

dt = 2.0


XK= np.zeros((7))
XK[0:6]=X0
XK[6]=dt


XI= np.zeros((8))
XI[0:6]=X0
XI[7]=dt


mu = 1.0
ode   = ast.Astro.Kepler.ode(mu)
kpint = ast.Astro.Kepler.KeplerPropagator(mu)




integ= ode.integrator(.01)
integ.Adaptive=True
integ.setAbsTol(1.0e-10)
integ.FastAdaptiveSTM=True

IG = integ.integrate_dense(XI[0:7],dt,1000)

L1 =[1,1,1,1,1,1,0]
L2 =L1[0:6]

'''
print(kpint.jacobian(XK)[0:6,0:6]-integ.jacobian(XI)[0:6,0:6])

print(kpint.jacobian(XK)[0:6,6]-integ.jacobian(XI)[0:6,7])

print(kpint.adjointhessian(XK,L2)[0:6,0:6]-integ.adjointhessian(XI,L1)[0:6,0:6])
'''

kpint.rpt(XK,20000)
integ.rpt(XI,20000)


XKK = [ 0.0475981   ,  2.64034 , 0.000475981  , -0.605902   ,  1.05489, -0.00605902 ,  -0.333333]

kpint.jacobian(XKK)[0:6,0:6]

input("s")
#print(integ.jacobian(XI)[0:6,0:6])

phase = ode.phase(TModes.CentralShooting)
#phase.UseKeplerPropagator = False
phase.setTraj(IG,5)
phase.addBoundaryValue(PhaseRegs.Front,range(0,7),[1.0,0,.01,0,1.65,0,0])

#phase.addBoundaryValue(PhaseRegs.Front,[4,6],[1.65,0.0])
phase.addBoundaryValue(PhaseRegs.Back,[6],[dt])

phase.Threads=1
phase.optimizer.QPThreads=1
phase.optimizer.PrintLevel=1

import time

t0 = time.perf_counter()

phase.solve()

tf = time.perf_counter()

print((tf-t0)*1000.0)



