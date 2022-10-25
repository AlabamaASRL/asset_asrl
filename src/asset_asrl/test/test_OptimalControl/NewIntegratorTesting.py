import numpy as np
import asset as ast
import unittest

import matplotlib.pyplot as plt
import time

vf = ast.VectorFunctions
oc = ast.OptimalControl
sol = ast.Solvers
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

np.set_printoptions(precision  = 3, linewidth = 200)

ode = ast.Astro.Kepler.ode(1.0).vf()

ode = oc.ode_x.ode(ode,6)


atol = 1.0e-12
dstep = .01

X0 = np.zeros((7))
X0[0]=1
X0[2]=.01
X0[4]=1.335
tf = 60.0

XT = np.zeros((7))
XT[0:6]=X0[0:6]
XT[6]=tf

integold = ode.integrator(dstep)
integold.Adaptive=True
integold.FastAdaptiveSTM=False
integold.setAbsTol(atol)
integnew = oc.TestIntegratorX("DOPRI87",ode,dstep)
integnew.setAbsTol(atol)
integnew.MaxStepChange=4
t0o = time.perf_counter()
Trajold = integold.integrate_dense(X0,tf)
tfo = time.perf_counter()

t0n = time.perf_counter()
Trajnew = integnew.integrate_dense(X0,tf)
tfn = time.perf_counter()

Fstate = ast.Astro.Kepler.KeplerPropagator(1.0).compute(XT)
Jstate = ast.Astro.Kepler.KeplerPropagator(1.0).jacobian(XT)[0:6,0:6]

print(len(Trajold),(tfo-t0o)*1000,Trajold[-1][0:6]-Fstate[0:6])

print(len(Trajnew),(tfn-t0n)*1000,Trajnew[-1][0:6]-Fstate[0:6])


print(Trajnew[-1]-Trajold[-1])


t0o = time.perf_counter()
Jold = integold.integrate_stm(X0,tf)[1][0:6,0:6]
tfo = time.perf_counter()

t0n = time.perf_counter()
Jnew = integnew.integrate_stm(X0,tf)[1][0:6,0:6]
tfn = time.perf_counter()

print((tfo-t0o)*1000)

print((tfn-t0n)*1000)

print(Jold-Jstate)
print(Jnew-Jstate)

TT1 = np.array(Trajnew).T

plt.plot(TT1[0],TT1[1],marker='o')


plt.axis("Equal")

plt.grid()
plt.show()
plt.plot(TT1[6],TT1[0],marker='o')

plt.show()



