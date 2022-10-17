import asset as ast
import numpy as np
import matplotlib.pyplot as plt

vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags


'''
Companion to "PhaseExample.py" but instead of transfering to another orbit, 
we maximize c3 for a solar system escape, intial state is in a parabolic orbit
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

phase = ode.phase("LGL3",IG,1024)
phase.setControlMode("BlockConstant")
phase.addBoundaryValue("Front", range(0,7), IState[0:7])
phase.addLUNormBound("Path",[7,8,9],.7,1.3)

def CosAlpha():
    args = Args(6)
    rhat = args.head3().normalized()
    nhat = args.tail3().normalized()
    return vf.dot(rhat,nhat)

MaxAlpha = 80.0
CosMaxAlpha =np.cos(np.deg2rad(MaxAlpha))
phase.addLowerFuncBound("Path",CosAlpha(),[0,1,2,7,8,9],CosMaxAlpha,1.0)

def C3():
    args = Args(6)
    r = args.head3()
    v = args.tail3()
    return v.dot(v) - 2*mu/r.norm()

phase.addStateObjective("Back",-1.0*C3(),range(0,6))
phase.addBoundaryValue("Back", [6], [TOF])
phase.optimizer.set_OptLSMode("L1")

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
plt.xlabel("t")
plt.ylabel(r"$\alpha$")
plt.grid(True)

plt.show()
################################################



    


    










