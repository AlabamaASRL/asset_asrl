import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from inspect import getmembers, isfunction, isclass
import time as time

def Plot(Traj,name,col,ax=plt):
    TT = np.array(Traj).T
    ax.plot(TT[0],TT[1],label=name,color=col)
    
def Scatter(State,name,col,ax=plt):
    ax.scatter(State[0],State[1],label=name,c=col)

#ast.CR3BP.ode(.012).rpt([.88,0,0,0,0,0,0],1000000)
#ast.PyMain()
#ast.CR3BP.ode(.012).rpt([.88,0,0,0,0,0,0],10000000)

vf = ast.VectorFunctions
oc = ast.OptimalControl

#F1 = ast.Lambda2(1)
#F2 = vf.Norm(5)
#F3 = ast.Lambda1(1)
#x=[1,1,1,1,1]

#F1.rpt(x,10000000)
#F2.rpt(x,10000000)
#F3.rpt(x,10000000)

#input("s")

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

odesize1 = oc.ode_6
odesize2 = oc.ode_6_3


mu = 0.0121505856


cr3bp         = ast.CR3BP.ode(mu)
cr3bp_integ   = ast.CR3BP.integrator(cr3bp, .001)
cr3bp_integ.Adaptive=True
cr3bp_integ.setAbsTol(1.0e-12)
cr3bp_integ.FastAdaptiveSTM=True


x0 = np.zeros((7))
x0[0]=0.7816
x0[4]=0.4435
T1 =1.975
T2 =3.95


TrajIG1 = cr3bp_integ.integrate_dense(x0,T1,200)
TrajIG2 = cr3bp_integ.integrate_dense(TrajIG1[-1],T2,200)

n=500
Tm = Tmodes.LGL7

phase1 = ast.CR3BP.phase(cr3bp,Tm)
phase1.setTraj(TrajIG1,n)
phase1.addBoundaryValue(PhaseRegs.Front,[1,2,3,5,6],[0,0,0,0,0])


Xs = np.linspace(x0[0],.83,15)

I = 0
def ContFunc(x): return x - np.array([Xs[I]])
phase1.addEqualCon(PhaseRegs.Front,vf.PyVectorFunction(1,1,ContFunc),[0])
phase1.setStaticParams([0])
#phase1.addIntegralParamFunction(vf.Norm(3),[3,4,5],0)

phase2 = ast.CR3BP.phase(cr3bp,Tm)
phase2.setTraj(TrajIG2,n)

phase2.addBoundaryValue(PhaseRegs.Front, [1],[0])
phase2.addBoundaryValue(PhaseRegs.Back, [1,3],[0,0])
phase2.setStaticParams([0])
#phase2.addIntegralParamFunction(vf.Norm(3),[3,4,5],0)

phase1.EnableVectorization=True
phase2.EnableVectorization=True

ocp = oc.OptimalControlProblem()
ocp.addPhase(phase1)
ocp.addPhase(phase2)


ocp.Threads = 5
ocp.optimizer.EContol = 1.0e-10
ocp.optimizer.PrintLevel = 0

ocp.addForwardLinkEqualCon(0,1,range(0,7))
ocp.setLinkParams([0])

fun =  Args(3)[0] + Args(3)[1] -Args(3)[2]
link = oc.LinkConstraint(fun,oc.LinkFlags.ParamsToParams,[[0,1]],[[],[]],[[],[]],[[0],[0]],[[0]])

ocp.addLinkEqualCon(link)
ocp.solve()

start = time.perf_counter()
for I in range(1,len(Xs)):
    ocp.solve()
stop = time.perf_counter()

print((stop-start)*1000.0)

print(phase1.returnStaticParams())
print(phase2.returnStaticParams())
#print(ocp..returnStaticParams())


Plot(phase1.returnTraj(),"P1",'r')
Plot(phase2.returnTraj(),"P2",'b')

plt.grid(True)
plt.show()











