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

def Twobody():
    X = Args(7)
    return vf.stack([X.segment3(3),-X.head3().normalized_power3()-0*X.head3().normalized_power3()-0*X.head3().normalized_power3()])

ode = oc.ode_x.ode(ode,6)
ode = oc.ode_x.ode(Twobody(),6)


atol = 1.0e-13
dstep = .1

X0 = np.zeros((7))
X0[0]=1
X0[2]=.0
X0[4]=1.1
tf =200.0

XT = np.zeros((7))
XT[0:6]=X0[0:6]
XT[6]=tf

XIN = np.zeros((8))
XIN[0:7]=X0
XIN[7]=tf

LIN = np.zeros((7))
LIN[0:6]=np.ones((6))

integold = oc.ode_x.ode(Twobody(),6).integrator(dstep)
integold.Adaptive=True
integold.FastAdaptiveSTM=True
integold.setAbsTol(atol)
integnew = oc.TestIntegratorX(ode,"DOPRI87",dstep)
integnew.setAbsTol(atol)
integnew.MaxStepChange=4
integnew.EventTol=1.0e-10
integnew.MaxEventIters=13

t0o = time.perf_counter()
Trajold = integold.integrate_dense(X0,tf,500)
tfo = time.perf_counter()

Fstate = ast.Astro.Kepler.KeplerPropagator(1.0).compute(XT)
Jstate = ast.Astro.Kepler.KeplerPropagator(1.0).jacobian(XT)[0:6,0:6]
Hstate = ast.Astro.kptest(1).adjointhessian(XT,np.ones(6))[0:6,0:6]



def EventFunc():
    X = Args(7)
    R = X.head(3)
    V = X.segment(3,3)
    
    return R.dot(V)

'''
integnew.EnableVectorization=True
t0n = time.perf_counter()
for i in range(0,1000):
    integnew.jacobian(XIN)
tfn = time.perf_counter()
print((tfn-t0n))

t0n = time.perf_counter()
for i in range(0,1000):
    integnew.adjointhessian(XIN,LIN)
tfn = time.perf_counter()
print((tfn-t0n))
'''
input("S")


t0n2 = time.perf_counter()
Trajnew2 = integnew.integrate_dense(X0,tf)
tfn2 = time.perf_counter()


t0n = time.perf_counter()
events=[[]]
Trajnew,events = integnew.integrate_dense(X0,tf,[ (EventFunc(),0,False) ])
#Trajnew = integnew.integrate_dense(X0,tf)
tfn = time.perf_counter()



print(len(Trajold),(tfo-t0o)*1000,Trajold[-1][0:6]-Fstate[0:6])
print(len(Trajnew),(tfn-t0n)*1000,Trajnew[-1][0:6]-Fstate[0:6])
print(len(Trajnew2),(tfn2-t0n2)*1000,Trajnew2[-1][0:6]-Fstate[0:6])
print(Trajnew[-1][6])
#print(len(events[0]))





t0o = time.perf_counter()
Xf,Jold = integold.integrate_stm_parallel(X0,tf,8)
tfo = time.perf_counter()

t0n = time.perf_counter()
Jnew = integnew.jacobian(XIN)
tfn = time.perf_counter()

t0n2 = time.perf_counter()
Xf,Jnew2 = integnew.integrate_stm_parallel(X0,tf,8)
tfn2 = time.perf_counter()

t0n2 = time.perf_counter()
Xf,Jnew2 = integnew.integrate_stm_parallel(X0,tf,8)
tfn2 = time.perf_counter()


print("SSSS")
print((tfo-t0o)*1000,abs(Jold[0:6,0:6]-Jstate[0:6,0:6]).max())
print((tfn-t0n)*1000,abs(Jnew[0:6,0:6]-Jstate[0:6,0:6]).max())
print((tfn2-t0n2)*1000,abs(Jnew2[0:6,0:6]-Jstate[0:6,0:6]).max())

print((tfn-t0n)*1000,abs(Jnew-Jold).max())


t0o = time.perf_counter()
Hold = integold.adjointhessian(XIN,LIN)
tfo = time.perf_counter()

t0n = time.perf_counter()
Hnew = integnew.adjointhessian(XIN,LIN)
tfn = time.perf_counter()

print(Hnew[0:6,0:6]-Hstate[0:6,0:6])

print((tfo-t0o)*1000,abs(Hold[0:6,0:6]-Hstate[0:6,0:6]).max())
print((tfn-t0n)*1000,abs(Hnew[0:6,0:6]-Hstate[0:6,0:6]).max())


Errs = []
for T in Trajnew:
    XT[6]=T[6]
    Fstate = ast.Astro.Kepler.KeplerPropagator(1.0).compute(XT)
    Err= abs(abs(T[0:6]-Fstate[0:6]).max())
    Errs.append(Err)
    
    
        
TT= np.array(Trajnew).T
print(TT[6][0])

plt.plot(TT[6],Errs)

plt.yscale("log")

plt.show()



plt.plot(TT[0],TT[1])

for e in events[0]:
    print(EventFunc()(e))
    plt.scatter(e[0],e[1])
    
    
plt.show()



    




