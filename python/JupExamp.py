import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import MKgSecConstants as c
from SpiceRead import GetEphemTraj2

def Plot(Traj,name,col,ax=plt):
    TT = np.array(Traj).T
    ax.plot(TT[0],TT[1],label=name,color=col)
    
def Scatter(State,name,col,ax=plt):
    ax.scatter(State[0],State[1],label=name,c=col)

norm = np.linalg.norm
vf = ast.VectorFunctions
oc = ast.OptimalControl


Args = vf.Arguments

Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Normalized3 = vf.Normalized(3).vf()



JD0 = 2459599.0
JDF = JD0 + 20.0*365.0

LU = c.AU
TU = c.year/ (2.0*c.pi)

YEAR = 2.0*c.pi
KM_S = 1000.0*TU/LU


def ODE(mu, ltacc=False):
    irows = 7
    if(ltacc!=False): irows+=3
    
    args = vf.Arguments(irows)
    r = args.head_3()
    v = args.segment_3(3)
    g = r.normalized_power3()*(-mu)
    if(ltacc!=False):
        thrust = args.tail_3()*ltacc
        acc = g + thrust
    else:
        acc = g
    return vf.Stack([v,acc])





EDAT = GetEphemTraj2("EARTH",JD0,JDF,8000,LU,TU)
JDAT = GetEphemTraj2("JUPITER BARYCENTER",JD0,JDF,4000,LU,TU)

Etab = oc.LGLInterpTable(ODE(1.0),6,0,Tmodes.LGL3,EDAT,5000)
Jtab = oc.LGLInterpTable(ODE(1.0),6,0,Tmodes.LGL3,JDAT,5000)


ltacc = .01
VinfIG  = 2.99*KM_S
MaxVinf = 3.00*KM_S

ode    = oc.ode_6_3.ode(ODE(1.0,ltacc),6,3)
integ  = oc.ode_6_3.integrator(ode,YEAR/300.0,(Normalized3*.90),[3,4,5])

#ode    = oc.ode_x_x.ode(ODE(1.0,ltacc),6,3)
#integ  = oc.ode_x_x.integrator(ode,YEAR/300.0)


TLow    = 2*YEAR
THigh   = 4*YEAR
Nsamps  = 200
MAXTOF  = 13*YEAR



Ts = np.linspace(2*YEAR,4*YEAR,Nsamps)
Nvals = []
Trajs = []

for T in Ts:
    IG0 = np.zeros((10))
    IG0[0:7] = Etab.Interpolate(T)
    IG0[3:6] = IG0[3:6] + Normalized3.compute(IG0[3:6])*VinfIG
    IG0[7]=.5
    
    def BreakFunc(x):
        Jn = norm(Jtab.Interpolate(x[6])[0:3])
        Rn = norm(x[0:3])
        if(Rn>Jn):return True
        else:return False
        
    TrajIG = integ.integrate_dense(IG0,T + MAXTOF,1000,BreakFunc)
    
    Trajs.append(TrajIG)
    x = TrajIG[-1]
    Nvals.append(norm(x[0:3]-Jtab.Interpolate(x[6])[0:3]))
   
TrajIG = Trajs[Nvals.index(min(Nvals))]

################################################

Plot(EDAT,'Earth','g')
Plot(JDAT,'Jupiter','r')
Plot(TrajIG,'Transfer','b')
Scatter(Jtab.Interpolate(TrajIG[-1][6]),'Jupiter','r')
Scatter([0,0],'Earth','gold')
plt.xlabel("X (AU)")
plt.ylabel("Y (AU)")

plt.grid(True)
plt.axis("Equal")
plt.show()
#################################################

phase = oc.ode_6_3.phase(ode,Tmodes.LGL5)
phase = ast.KeplerLT.phase(ast.KeplerLT.ode(1.0,ltacc),Tmodes.LGL5)
nn=160
phase.setTraj(TrajIG,nn)
phase.setIntegralMode(oc.BaseIntegral)
phase.setControlMode(oc.BlockConstant)



def RendCon(tab,Vars):
    n = len(Vars)
    args = Args(n+1)
    x = args.head(n)
    t = args[n]
    fun = oc.InterpFunction(tab,Vars).vf()
    return (fun.eval(t) - x)

def VinfFunc(tab):
    args = Args(4)
    v = args.head(3)
    t = args[3]
    fun = oc.InterpFunction(tab,[3,4,5]).vf()
    dV = vf.Norm(3).sf().eval(fun.eval(t) - v)
    return (dV)
    
    

phase.addEqualCon(PhaseRegs.Front,RendCon(Etab,[0,1,2]),[0,1,2,6])

phase.addUpperFuncBound(PhaseRegs.Front,VinfFunc(Etab), [3,4,5,6],MaxVinf,1.0)


phase.addEqualCon(PhaseRegs.Back, RendCon(Jtab,range(0,6)),range(0,7))

idx = phase.addLUNormBound(PhaseRegs.Path,[7,8,9],.0001,1.0,1.0)

phase.addLUNormBound(PhaseRegs.Path,[0,1,2],.9,6.1 ,1.0)
#phase.addLUNormBound(PhaseRegs.Back, [0,1,2],1.0,6.0 ,1.0)


phase.addLUVarBound(PhaseRegs.Front,6,2*YEAR,5*YEAR,1.0)
phase.addUpperVarBound(PhaseRegs.Back,6,20*YEAR,1.0)

#phase.addIntegralObjective(vf.Norm(3).sf()*.01,[7,8,9])
phase.addDeltaTimeObjective(.02)

phase.optimizer.MaxIters = 1200
phase.optimizer.MaxAccIters = 500

phase.optimizer.BoundFraction  = .997
phase.optimizer.OptLSMode = ast.L1
phase.optimizer.MaxLSIters =2
phase.optimizer.QPPivotPerturb = 9
phase.optimizer.MaxSOC=1
phase.optimizer.PrintLevel=0
phase.optimizer.NegSlackReset = 1.0e-12
#phase.optimizer.SoeMode =1
phase.optimizer.incrH = 8
phase.optimizer.decrH = .33
#phase.optimizer.CNRMode = True
phase.EnableVectorization=True
phase.Threads=8
phase.optimizer.QPThreads=8


Flag = phase.solve()
Flag = phase.optimize()

TrajConv = phase.returnTraj()



#plt.yscale("log")


######################################
Plot(EDAT,'Earth','g')
Plot(JDAT,'Jupiter','r')
Plot(TrajConv,'Transfer','b')
Scatter(Jtab.Interpolate(TrajConv[-1][6]),'Jupiter','r')
Scatter(Etab.Interpolate(TrajConv[0][6]),'Earth','g')
Scatter([0,0],'Earth','gold')

plt.grid(True)
plt.axis("Equal")
plt.xlabel("X (AU)")
plt.ylabel("Y (AU)")

plt.show()

TU = np.array(TrajConv).T
n1 = 1#(TU[7]**2 + TU[8]**2 +TU[9]**2)**.5

plt.plot(TU[6],TU[7]/n1,label="Ux")
plt.plot(TU[6],TU[8]/n1,label="Uy")
plt.plot(TU[6],TU[9]/n1,label="Uz")
plt.plot(TU[6],(TU[7]**2 + TU[8]**2+TU[9]**2)**.5,label="|U|")

lm=phase.returnCostateTraj()
ts = np.linspace(TrajConv[0][6],TrajConv[-1][6],len(lm))
lmt = np.array(lm).T
n1 = (lmt[3]**2 + lmt[4]**2 +lmt[5]**2)**.5
n=1#n1



plt.plot(ts,-lmt[3]/n)
plt.plot(ts,-lmt[4]/n)
plt.plot(ts,-lmt[5]/n)
plt.plot(ts,n1)

lm=phase.returnInequalConLmults(idx)
ts = np.linspace(TrajConv[0][6],TrajConv[-1][6],len(lm))
lmt = np.array(lm).T
h= 2*(ts[1]-ts[0])/nn
#plt.plot(ts,(lmt[1]-lmt[0])/h+ 100)


plt.legend()
plt.grid(True)
plt.show()
############################################
