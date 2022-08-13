import asset as ast
import numpy as np
import matplotlib.pyplot as plt

vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags

Tmax =8
V=4
def Cost(N,T,Tmax):
    N_t = N//T
    Nmt = N_t%V
    NN=N%T
    NN_t = NN//T
    NNmt = NN_t%V
    return ((T/min(T,Tmax))**1.01)*(N_t//V + Nmt + NN//V + NN%V ) 
    
Ts = range(1,19)
Ns = range(20,1000)

Bests=[]
Peaks=[]
Costs=[]
for N in Ns:
    costs = [Cost(N,T,Tmax) for T in Ts]
    Bests.append(Ts[costs.index(min(costs))])
    Costs.append((N/(4*Tmax))/min(costs))
    #Peaks.append(N/(4*Tmax))
plt.plot(Ns,Bests)
plt.show()

plt.plot(Ns,Costs)
#plt.plot(Ns,Peaks)

plt.show()

'''
Companion to "PhaseExample.py" but instead of transfering to another orbit, 
we maximize c3 for a solar system escape, intial state is in parabolic orbit
'''

def SolarSail_Acc(r, n, scale):
    ndr2 = vf.dot(r, n).squared()
    acc = scale * ndr2 * r.inverse_four_norm() * n.normalized_power3()
    return acc

class Full_TwoBody_SolarSail_Model(oc.ode_6_3.ode):
    def __init__(self,mu,beta):
        Xvars = 6
        Uvars = 3
        Ivars = Xvars + 1 + Uvars
        #############################
        args = Args(Ivars)
        r = args.head3()
        v = args.segment3(3)
        n = args.tail3()
        acc = -mu * r.normalized_power3() + SolarSail_Acc(r, n, beta * mu)
        odeeq =  vf.stack([v, acc])
        super().__init__(odeeq,Xvars,Uvars)


mu   = 1
beta = .19
ode = Full_TwoBody_SolarSail_Model(mu, beta)


R0=1
V0=np.sqrt(2)

IState = np.zeros((10))
IState[0]=R0
IState[4]=V0
IState[7]=1


def ProgradeFunc():
    args = Args(6)
    rhat = args.head3().normalized()
    vhat = args.tail3().normalized()
    return (rhat + .8*vhat).normalized()

integ = ode.integrator(0.01,ProgradeFunc(),range(0,6))

TOF = 12.0*np.pi
IG = integ.integrate_dense(IState,TOF,500)

phase = ode.phase(Tmodes.LGL3,IG,1024)
phase.setControlMode(oc.ControlModes.BlockConstant)
phase.addBoundaryValue(PhaseRegs.Front, range(0,7), IState[0:7])
phase.addLUNormBound(PhaseRegs.Path,[7,8,9],.7,1.3)

def CosAlpha():
    args = Args(6)
    rhat = args.head3().normalized()
    nhat = args.tail3().normalized()
    return vf.dot(rhat,nhat)

MaxAlpha = 80.0
CosMaxAlpha =np.cos(np.deg2rad(MaxAlpha))
phase.addLowerFuncBound(PhaseRegs.Path,CosAlpha(),[0,1,2,7,8,9],CosMaxAlpha,1.0)

def C3():
    args = Args(6)
    r = args.head3()
    v = args.tail3()
    return v.dot(v) - 2*mu/r.norm()

phase.addStateObjective(PhaseRegs.Back,-1.0*C3(),range(0,6))
phase.addBoundaryValue(PhaseRegs.Back, [6], [TOF])
phase.optimizer.OptLSMode = ast.Solvers.LineSearchModes.L1
phase.optimizer.MaxLSIters = 1
#phase.optimizer.QPOrderingMode = ast.Solvers.QPOrderingModes.MINDEG

ast.SoftwareInfo()
phase.optimize()


#############################################
TrajConv = phase.returnTraj()
TT = np.array(TrajConv).T
IGT = np.array(IG).T

plt.plot(TT[0],TT[1],label="Optimal Trajectory")
plt.plot(IGT[0],IGT[1],label="Initial Guess")

plt.scatter(TT[0][-1],TT[1][-1],marker="*",zorder=500)
plt.scatter(IGT[0][-1],IGT[1][-1],marker="*",zorder=500)
plt.scatter(IGT[0][0],IGT[1][0],color='k',marker="o",zorder=500)

plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")

angs = np.linspace(0,np.pi*2)

XX = np.cos(angs)
YY = np.sin(angs)

plt.plot(XX*R0,YY*R0,color='k',linestyle='dotted')
plt.legend()
plt.axis("Equal")
plt.show()
alphas =[]
for T in TrajConv:
    f = CosAlpha().compute
    X=np.zeros((6))
    X[0:3]=T[0:3]
    X[3:6]=T[7:10]
    alphas.append(np.rad2deg(np.arccos(f(X))))
plt.plot(TT[6],alphas)
plt.show()
################################################



    


    










