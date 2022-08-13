import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from inspect import getmembers, isfunction, isclass


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags


ode = oc.ode_3.ode(ast.BrannamHW(.001))
integ = oc.ode_3.integrator(ode,.0001)

ode2 = oc.ode_4.ode(ast.BrannamHW2(.001))
integ2 = oc.ode_4.integrator(ode2,.0001)

X0=np.zeros((4))
X0[0]=1
X0[1]=0
X0[2]=1

T = integ.integrate_dense(X0,830,1000)

X1=np.zeros((5))
X1[0]=1
X1[1]=0
X1[2]=0
X1[3]=1

T1 = integ2.integrate_dense(X1,812.15,8000)
thetas = np.linspace(0,6.3,1000)

Xs =[x[0]*np.cos(x[1]) for x in T1]
Ys =[x[0]*np.sin(x[1]) for x in T1]

rm =384000/19171
Xm =[rm*np.cos(t) for t in thetas]
Ym =[rm*np.sin(t) for t in thetas]

Rs =[x[0] for x in T1]
Ts = [x[4] for x in T1]

fig,axs = plt.subplots(1,2)

axs[0].plot(Xs,Ys)
axs[0].plot(Xm,Ym,color='grey',label='Moon Orbit')

axs[0].set_xlabel("X(ND)")
axs[0].set_ylabel("Y(ND)")
circle1 = plt.Circle((0, 0), 0.333, color='g',label='Earth')
axs[0].add_patch(circle1)
axs[0].legend()
axs[0].grid(True)

axs[1].plot(Ts,Rs)
axs[1].plot([Ts[0],Ts[-1]],[rm,rm],color='black',label=r'$\rho_m$',linestyle='--')
axs[1].grid(True)
axs[1].set_xlabel(r'$\tau$')
axs[1].set_ylabel(r'$\rho$')
axs[1].legend()

plt.show()

