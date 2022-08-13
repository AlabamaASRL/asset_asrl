import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags


intg = vf.Norm(2)
x = [.01,.01,0,1,2,1]
l=[1]
F1 =ast.LG2(intg,2,0)
F2 =ast.LGD(intg,2,0)
F3 =vf.PyScalarFunction(6,F1.compute,1.0e-7,1.0e-5)

F1.rpt(x,1000000)
F2.rpt(x,1000000)

F4 =ast.LGD(vf.Norm(4),4,0)
x = [.01,.01,0,0,0,1,2,0,0,1]
F4.rpt(x,1000000)


#input("s")

mu = 0.0121505856
ode   = ast.CR3BP.ode(mu)
integ = ast.CR3BP.integrator(ode, .001)
integ.Adaptive=True
integ.setAbsTol(1.0e-12)
integ.FastAdaptiveSTM=True
x0 = np.zeros((7))
x0[0]=0.7816
x0[4]=0.4435
TF =3.95

TrajIG = integ.integrate_dense(x0,TF,200)

def CRODE(mu):
    args = vf.Arguments(7)
    r = args.head_3()
    v = args.segment_3(3)
    
    x    = args[0]
    y    = args[1]
    xdot = args[3]
    ydot = args[4]
    
    t1 = vf.ScaledSum([ydot, 2.0], [x,1])
    t2 = vf.ScaledSum([xdot,-2.0], [y,1])
    
    rterms = vf.StackScalar([t1,t2]).padded_lower(1)
    
    p1loc = np.array([-mu,0,0])
    p2loc = np.array([1.0-mu,0,0])
    
    g1 = r.normalized_power3(-p1loc,(mu-1.0))
    g2 = r.normalized_power3(-p2loc,(-mu))
    acc = vf.Sum([rterms,g1,g2])
    return vf.Stack([v,acc])

l=[1,1,1,1,1,1]

print(CRODE(mu).adjointhessian(x0,l)-ode.adjointhessian(x0,l))

CRODE(mu).rpt(x0,1000000)
ode.rpt(x0,1000000)

odetemp = ode
phasetype = ast.CR3BP.phase

odetemp = oc.ode_6.ode_6(ode)
#odetemp = oc.ode_6.ode_6(CRODE(mu))
phasetype = oc.ode_6.phase

phase = phasetype(odetemp,Tmodes.LGL3)
phase.integrator.setAbsTol(1.0e-12)
phase.integrator.FastAdaptiveSTM=True

phase.optimizer.SoeBarMode = ast.LOQO
phase.optimizer.PDStepStrategy = 0
phase.optimizer.BoundFraction = 0.95
#phase.optimizer.QPPivotPerturb = 10
phase.optimizer.PrintLevel = 0
phase.optimizer.QPThreads = 1
phase.optimizer.QPOrd = 2
phase.optimizer.QPRefSteps = 0
phase.Threads = 1
phase.setTraj(TrajIG,320)
phase.addBoundaryValue2(PhaseRegs.Front,[1,2,3,5,6],[0,0,0,0,0])
phase.addBoundaryValue2(PhaseRegs.Front,[0],[x0[0]])
phase.addBoundaryValue2(PhaseRegs.Back, [1,3],[0,0])

#print(vf.Norm(2).ORows())
#phase.addIntegralParamFunction(vf.Norm(2),[3,4],0)
#phase.setStaticParams([0.0])

phase.Solve()

TP = np.array(TrajIG).T
plt.plot(TP[0],TP[1])

print(phase.returnStaticParams())
TrajConv = phase.returnTraj()
    
TP = np.array(TrajConv).T
plt.plot(TP[0],TP[1])



plt.grid(True)
plt.show()