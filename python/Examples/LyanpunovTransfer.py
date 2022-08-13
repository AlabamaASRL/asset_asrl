import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from DerivChecker import FDDerivChecker


vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

def CR3BP(mu,DoLT = False,ltacc=.05):
    irows = 7
    if(DoLT==True): irows+=3
    
    args = vf.Arguments(irows)
    r = args.head3()
    v = args.segment3(3)
    
    x    = args[0]
    y    = args[1]
    xdot = args[3]
    ydot = args[4]
    
    rterms = vf.stack([2*ydot + x,
                       -2.0*xdot +y]).padded_lower(1)
    
    p1loc = np.array([-mu,0,0])
    p2loc = np.array([1.0-mu,0,0])
    
    g1 = (r-p1loc).normalized_power3()*(mu-1.0)
    g2 = (r-p2loc).normalized_power3()*(-mu)
    
    if(DoLT==True):
        thrust = args.tail3()*ltacc
        acc = vf.sum([rterms,g1,g2,thrust])
    else:
        acc = vf.sum([g1,g2,rterms])
    return vf.stack([v,acc])

#######################################################
    
mu = 0.0121505856

cr3bp         = oc.ode_6.ode(CR3BP(mu),6)
cr3bp_integ   = cr3bp.integrator(.01)
cr3bp_integ.Adaptive=True
cr3bp_integ.setAbsTol(1.0e-12)


x0 = np.zeros((7))
x0[0]=0.7816
x0[4]=0.4435
TF =3.95

xin = np.zeros((8))
xin[0:7]=x0
xin[7]=.05

x2 = np.zeros((10))
x2[0:7]=x0
x2[8]=1


TrajIG = cr3bp_integ.integrate_dense(x0,TF,200)

X =np.zeros((8))
X[0:7]=x0
X[7]=TF/2


Xs = np.linspace(x0[0],.83,10)


cr3bp.vf().SpeedTest(TrajIG[0],100000)

phase = cr3bp.phase(Tmodes.LGL3)
phase.setTraj(TrajIG,128)
phase.addBoundaryValue(PhaseRegs.Front,[1,2,3,5,6],[0,0,0,0,0])
phase.addBoundaryValue(PhaseRegs.Back, [1,3],[0,0])

I = 0
def ContFunc(x): return x - np.array([Xs[I]])
phase.addEqualCon(PhaseRegs.Front,vf.PyVectorFunction(1,1,ContFunc),[0])

Trajs=[]

for I in range(0,len(Xs)):
    phase.solve()
    TrajConv = phase.returnTraj()
    Trajs.append(TrajConv)
    TP = np.array(TrajConv).T
    plt.plot(TP[0],TP[1])

#####################################################

plt.grid(True)
plt.axis('equal')
plt.show()


T1 = Trajs[2]
T2 = Trajs[6]

Tcopy = []
for T in T1:
    X = np.zeros((10))
    X[0:7] = T
    X[7]=.01
    X[8]=.01
    Tcopy.append(X)
    
cr3bp_lt   = oc.ode_6_3.ode(CR3BP(mu,True,.07),6,3)


ltphase    = cr3bp_lt.phase(Tmodes.LGL3,Tcopy,512)
ltphase.setControlMode(oc.ControlModes.BlockConstant)

Tab1 = oc.LGLInterpTable(cr3bp,6,0,Tmodes.LGL3,T1,200)
Tab1.makePeriodic()
Tab2 = oc.LGLInterpTable(CR3BP(mu),6,0,Tmodes.LGL3,T2,200)
Tab2 = oc.LGLInterpTable(6,T2,200)
Tab2.makePeriodic()

def RendFun(tab):
    args = Args(7)
    x = args.head(6)
    t = args[6]
    fun = oc.InterpFunction(tab,range(0,6)).vf()
    return fun.eval(t) - x


ltphase.setStaticParams([.01,.01])
ltphase.addBoundaryValue(PhaseRegs.Front,[6],[0])
ltphase.addEqualCon(PhaseRegs.Front,RendFun(Tab1),range(0,6),[],[0])
ltphase.addEqualCon(PhaseRegs.Back, RendFun(Tab2),range(0,6),[],[1])
ltphase.addLUNormBound(PhaseRegs.Path,[7,8,9],.01,1.0,1.0)
ltphase.addIntegralObjective(Args(3).norm(),[7,8,9])

#phase.optimizer.BoundFraction  = .99
ltphase.optimizer.OptLSMode = ast.Solvers.LineSearchModes.L1
ltphase.optimizer.MaxLSIters =1
#ltphase.optimizer.MaxAccIters =125
ltphase.optimizer.PrintLevel =1


ltphase.solve_optimize()
TConvM = ltphase.returnTraj()
ltphase.removeIntegralObjective(-1)

ltphase.addDeltaTimeObjective(1.0)
ltphase.optimize()
TConvT = ltphase.returnTraj()



########################################
TP = np.array(T1).T
plt.plot(TP[0],TP[1],label ='Orbit1')
TP = np.array(T2).T
plt.plot(TP[0],TP[1],label ='Orbit2')


TPT = np.array(TConvM).T
plt.plot(TPT[0],TPT[1],label ='Mass Optimal',color='red')
plt.scatter(TPT[0][0],TPT[1][0],c='red',marker='.')
plt.scatter(TPT[0][-1],TPT[1][-1],c='red',marker='*')

TPM = np.array(TConvT).T
plt.plot(TPM[0],TPM[1],label ='Time Optimal',color='blue')
plt.scatter(TPM[0][0],TPM[1][0],c='blue',marker='.')
plt.scatter(TPM[0][-1],TPM[1][-1],c='blue',marker='*')

plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
########################################










