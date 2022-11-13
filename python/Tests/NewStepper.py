import numpy as np
import asset as ast
import unittest

import matplotlib.pyplot as plt
import time
from DerivChecker import FDDerivChecker

vf = ast.VectorFunctions
oc = ast.OptimalControl
sol = ast.Solvers
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

np.set_printoptions(precision  = 3, linewidth = 200)

ode = ast.Astro.Kepler.ode(1.0).vf()
def Twobody():
    X = Args(7)
    R = X.head3()
    V = X.segment(3,3)
    return vf.stack([V,-R.normalized_power3()*(1+X[6])])


ode = oc.ode_x.ode(Twobody(),6)


atol = 1.0e-12
dstep = .01

X0 = np.zeros((7))
X0[0]=1
X0[1]=.03
X0[2]=.02
X0[4]=1.1
X0[5]=.02
tf =.1

XT = np.zeros((7))
XT[0:6]=X0[0:6]
XT[6]=tf

XIN = np.zeros((8))
XIN[0:7]=X0
XIN[7]=tf

integold = ode.integrator(dstep)
integold.Adaptive=True
integold.FastAdaptiveSTM=False
integold.setAbsTol(atol)
integnew = oc.TestIntegratorX("DOPRI87",ode,dstep)
integnew.setAbsTol(atol)
integnew.MaxStepChange=4
integnew.EventTol=1.0e-10
integnew.MaxEventIters=13


stepperold = integnew.getstepper()
steppernew = oc.TestStepperX(ode)

(stepperold.vf()*1).rpt(XIN,10000)
(steppernew.vf()*1).rpt(XIN,10000)


#stepperold.rpt(XIN,10000)
stepperold.SuperTest(XIN,10000)
steppernew.SuperTest(XIN,10000)

LIN = np.ones((7))

f1,Jold,g,Hold = stepperold.computeall(XIN,LIN)
f2,Jnew,g,Hnew = steppernew.computeall(XIN,LIN)

#Jold = steppernew.jacobian(XIN)



t0o = time.perf_counter()
Trajold = integold.integrate_dense(X0,tf,500)
tfo = time.perf_counter()

Fstate = ast.Astro.Kepler.KeplerPropagator(1.0).compute(XT)
Jstate = ast.Astro.Kepler.KeplerPropagator(1.0).jacobian(XT)[0:6,0:6]
Hstate = ast.Astro.kptest(1).adjointhessian(XT,np.ones(6))

FDDerivChecker(stepperold,XIN)

#Jnew = steppernew.jacobian(XIN)
print((Hstate))
print("")
print((Hold))
print("")
print((Hnew))
print("")
print((Hold-Hnew)/Hold)
