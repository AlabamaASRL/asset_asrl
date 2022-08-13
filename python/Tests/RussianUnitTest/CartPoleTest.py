import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import seaborn as sns


vf = ast.VectorFunctions
oc = ast.OptimalControl
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Args = vf.Arguments

class CartPole(oc.ode_x_u.ode):
    
    def __init__(self,l,m1,m2,g):
        
        args = oc.ODEArguments(4,1)
        
        q1 = args.XVar(0)
        q2 = args.XVar(1)
        q1d = args.XVar(2)
        q2d = args.XVar(3)
        
        u = args.UVar(0)
        
        q1dd = (l*m2*vf.sin(q2)*(q2d**2) + u + m2*g*vf.cos(q2)*vf.sin(q2))/( m1 + m2*((1-vf.cos(q2)**2)))
        
        q2dd = -1*(l*m2*vf.cos(q2)*vf.sin(q2)*(q2d**2) +u*vf.cos(q2) +(m1*g+m2*g)*vf.sin(q2))/( l*m1 + l*m2*((1-vf.cos(q2)**2)))
        
        ode = vf.stack([q1d,q2d,q1dd,q2dd])
        
        super().__init__(ode,4,1)
        
        
m1 = 1
m2 =.3
l=.5
g = 9.81

umax = 20
dmax = 2

tf = 2
d = 1

ts = np.linspace(0,tf,100)

IG = [[d*t/tf,np.pi*t/tf,0,0,t,.00] for t in ts]

ode = CartPole(l,m1,m2,g)
phase = ode.phase(Tmodes.LGL3,IG,64)

phase.addBoundaryValue(PhaseRegs.Front,range(0,5),[0,0,0,0,0])
phase.addBoundaryValue(PhaseRegs.Back,range(0,5),[d,np.pi,0,0,tf])
phase.addLUVarBound(PhaseRegs.Path,5,-umax,umax,1.0)
phase.addLUVarBound(PhaseRegs.Path,0,-dmax,dmax,1.0)
phase.addIntegralObjective(Args(1)[0]**2,[5])
phase.Threads=8
phase.optimizer.QPThreads = 8
phase.optimizer.PrintLevel= 1
phase.optimize()


print("Total Time (Sum of all below)             :",phase.optimizer.LastTotalTime," s")
print("Function/Derivative Eval Time             :",phase.optimizer.LastFuncTime," s")
print("KKT Matrix Factor/Solve Time              :",phase.optimizer.LastKKTTime," s")
print("KKT Matrix Pre-Analysis/Mem Alloc Time    :",phase.optimizer.LastPreTime," s")
print("Miscellaneous (Mostly Console Print) Time :",phase.optimizer.LastMiscTime," s")


T = phase.returnTraj()


###########################

T = np.array(T).T


P0X = T[0]
P0Y = np.zeros_like(T[0])

P1X  = T[0] + l*np.sin(T[1])
P1Y  = -l*np.cos(T[1])


n = len(P0X)
cols=sns.color_palette("viridis",n)


    

fig = plt.figure()
ax0 = plt.subplot(321)
ax1 = plt.subplot(323)
ax2 = plt.subplot(325)
ax3 = plt.subplot(122)





ax0.plot(T[4],T[0])
ax1.plot(T[4],T[1])
ax2.plot(T[4],T[5])

ax3.plot(P0X,P0Y,color='k')
ax3.plot(P1X,P1Y,color='k')

for i in range(0,n):
    xs = [P0X[i],P1X[i]]
    ys = [P0Y[i],P1Y[i]]
    ax3.plot(xs,ys,color=cols[i],marker='o')
  
ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

ax0.set_ylabel(r'$X$')
ax1.set_ylabel(r'$\theta$')
ax2.set_ylabel(r'$U$')
ax2.set_xlabel(r't')

ax3.set_xlabel(r'$X$')
ax3.set_ylabel(r'$Y$')


fig.set_size_inches(10.5, 5.5, forward=True)

fig.set_tight_layout(True)
plt.show()











