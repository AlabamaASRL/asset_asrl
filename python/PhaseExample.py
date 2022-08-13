import asset as ast
import numpy as np
import matplotlib.pyplot as plt
from DerivChecker import FDDerivChecker

vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags

def SolarSail_Acc(r, n, scale):
    ndr2 = vf.dot(r, n).squared()
    acc = scale * ndr2 * r.inverse_four_norm() * n.normalized_power3()
    return acc


def MccinnesSail(r,n,beta,mu,n1,n2,t1):
    ndr  = vf.dot(r, n)
    rn   = r.norm()*n.norm()
    ncr  = vf.cross(n,r)
    ncrn = vf.cross(ncr,n)
    N3DR4 = vf.dot(n.normalized_power3(),r.normalized_power4())
    sc= (beta*mu/2.0)
    acc = N3DR4*(((n1*sc)*ndr + (n2*sc)*rn)*n  + (t1*sc)*ncrn)
    return acc  #+ acc*0 + acc*0 #+ acc*0 + acc*0 #+ acc*0 + acc*0 + acc*0

def MccinnesSailC(r,n,beta,mu,rbar=.91,sbar=.89,Bf=.79,Bb=.67,ef=.025,eb=.27):
    n1 = 1 + rbar*sbar
    n2 = Bf*(1-sbar)*rbar + (1-rbar)*(ef*Bf - eb*Bb)/(ef+eb)
    t1 = 1 - sbar*rbar
    return MccinnesSail(r,n,beta,mu,n1,n2,t1)

class Full_TwoBody_SolarSail_Model(oc.ode_x_x.ode):
    def __init__(self,mu,beta):
        Xvars = 6
        Uvars = 3
        #############################
        args = oc.ODEArguments(Xvars,Uvars)
        r = args.XVec().head3()
        v = args.XVec().tail3()
        n = args.UVec().head3()
        acc = -mu * r.normalized_power3() + MccinnesSailC(r, n, beta,mu)
        odeeq =  vf.Stack([v, acc])
        super().__init__(odeeq,Xvars,Uvars)


mu   = 1
beta = .02

ode = Full_TwoBody_SolarSail_Model(mu, beta)


R0=1
V0=1
RF = 1.2

IState = np.zeros((10))
IState[0]=R0
IState[4]=V0
IState[7]=1

FDDerivChecker(ode.vf(),IState)


#ode.vf().SuperTest(IState,1000000)

def ProgradeFunc():
    args = Args(6)
    rhat = args.head3().normalized()
    vhat = args.tail3().normalized()
    return (rhat + .75*vhat).normalized()

integ = ode.integrator(0.01,ProgradeFunc(),range(0,6))

IG = integ.integrate_dense(IState,5.6*np.pi,500)
IGT = np.array(IG).T

#ode.vf().SuperTest(IG[0],1000000)

phase = ode.phase(Tmodes.LGL5,IG,900)
#phase.setControlMode(oc.ControlModes.BlockConstant)
phase.addBoundaryValue(PhaseRegs.Front, range(0,7), IState[0:7])
phase.addLUNormBound(PhaseRegs.Path,[7,8,9],.7,1.3)

def CosAlpha():
    args = Args(6)
    rhat = args.head3().normalized()
    nhat = args.tail3().normalized()
    return vf.dot(rhat,nhat)

MaxAlpha = 70.0
CosMaxAlpha =np.cos(np.deg2rad(MaxAlpha))
phase.addLowerFuncBound(PhaseRegs.Path,CosAlpha(),[0,1,2,7,8,9],CosMaxAlpha,1.0)


def VCirc(r):
    args = Args(6)
    rvec = args.head3()
    vvec = args.tail3()
    
    f1 = rvec.norm() + [-r]
    f2 = vvec.norm() + [-np.sqrt(mu/r)]
    f3 = vf.dot(rvec.normalized(),vvec.normalized())
    f4 = rvec[2]
    f5 = vvec[2]
    return vf.Stack([f1,f2,f3,f4,f5])


phase.addEqualCon(PhaseRegs.Back,VCirc(RF),range(0,6))
phase.addDeltaTimeObjective(1.0)
phase.enable_vectorization(True)

phase.optimizer.OptLSMode = ast.LineSearchModes.L1
phase.optimizer.MaxLSIters = 1
phase.Threads = 32
phase.optimizer.QPThreads=8
phase.optimizer.incrH = 8
phase.optimizer.decrH = .09

phase.optimize()

#############################################
TrajConv = phase.returnTraj()
Tab      = oc.LGLInterpTable(ode,6,3,Tmodes.LGL5,TrajConv,900)
integ    = ode.integrator(0.001,Tab,[7,8,9])
integ.Adaptive = True
TrajConv2=integ.integrate_dense(TrajConv[0],TrajConv[-1][6])


print(TrajConv[-1] - TrajConv2[-1])
TT = np.array(TrajConv).T
plt.plot(TT[0],TT[1],label="Optimal Trajectory")
plt.plot(IGT[0],IGT[1],label="Initial Guess")
TT2 = np.array(TrajConv2).T
plt.plot(TT2[0],TT2[1],label="Optimal Trajectory2")



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
plt.plot(XX*RF,YY*RF,color='k',linestyle='--')
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

#plt.plot(TT[6],TT[8])

plt.show()
################################################



    


    










