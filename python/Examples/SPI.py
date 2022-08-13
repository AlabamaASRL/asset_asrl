import numpy as np
import matplotlib.pyplot as plt
import asset as ast
from FramePlot import TBPlot,plt,colpal
from DerivChecker import FDDerivChecker

import time

norm = np.linalg.norm
def normalize(x): return np.copy(x)/norm(x)

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
LinkFlags = oc.LinkFlags

class SolarSail():
    def __init__(self,beta,Ideal=False,rbar=.91,sbar=.89,Bf=.79,Bb=.67,ef=.025,eb=.27):
        self.Ideal=Ideal
        self.beta = beta
        self.rbar = rbar
        self.sbar =sbar
        self.Bf =Bf
        self.Bb =Bb
        self.ef=ef
        self.eb=eb
       
        
        self.n1 = 1 + self.rbar*self.sbar
        self.n2 = self.Bf*(1-self.sbar)*self.rbar + (1-self.rbar)*(self.ef*self.Bf - self.eb*self.Bb)/(self.ef+self.eb)
        self.t1 = 1 - self.sbar*self.rbar
        
        if(Ideal==True):self.Normalbeta = self.beta
        else:self.Normalbeta = self.beta*(self.n1+self.n2)/2.0
        
    def ThrustExpr(self,r,n,mu):
        if(self.Ideal==True):return self.IdealSailExpr(r,n,mu)
        else :return self.MccinnesSailExpr(r,n,mu)
    def IdealSailExpr(self,r, n, mu):
        ndr2 = vf.dot(r, n)**2
        scale = self.beta*mu
        acc = scale * ndr2 * r.inverse_four_norm() * n.normalized_power3()
        return acc
    def MccinnesSailExpr(self,r,n,mu):
        
        ndr  = r.dot(n)
        rn   = r.norm()*n.norm()
        #ncr  = n.cross(r)
        ncrn = n.cross(r).cross(n)
        ncrn = vf.doublecross(n,r,n)
        
        N3DR4 = vf.dot(n.normalized_power3(),r.normalized_power4())
        sc= (self.beta*mu/2.0)
        acc1 = N3DR4*(((self.n1*sc)*ndr + (self.n2*sc)*rn)*n  + (self.t1*sc)*ncrn)
        
        return acc1


class SailModel(oc.ode_6_3.ode):
     def __init__(self, mu, beta):
        ## We will trackmass seperately
        Xvars = 6
        Uvars = 3
        Sail = SolarSail(beta)
        ############################################################
        args = oc.ODEArguments(Xvars, Uvars)
        p = args[0]
        f = args[1]
        g = args[2]
        h = args[3]
        k = args[4]
        L = args[5]
        
        sinL = vf.sin(L)
        cosL = vf.cos(L)
        n = args.tail3()#.normalized()
        
        w = 1.+f*cosL +g*sinL
        r = (p/w).padded_lower(2)
        
        ndr2 = vf.dot(r, n).squared()
        acc = (beta*mu) * ndr2 * r.inverse_four_norm() * n.normalized_power3()
        acc = Sail.ThrustExpr(r, n, mu)
        ode   = ast.Astro.ModifiedDynamics(mu).eval(vf.stack(args.head(6),acc))
        
        #############################################################
        super().__init__(ode, Xvars, Uvars)

def Beta(area,mass):
    sigma = 1000*mass/area
    ac = 9.08/sigma
    return ac/5.94
    
a=31
def Inc(fgL):
    f   =  fgL[0]
    g   =  fgL[1]
    L   =  fgL[2]
    
    st  = np.sin(L - np.arctan(g/(f+1.0e-10)))
    
    if(st>0):
        s = 1
    else:
        s=-1
    
    x = np.cos(np.deg2rad(a))
    y = 0
    z = np.sin(np.deg2rad(a))*s
    return [x,y,z]

def IncLaw(a):
    
    fgL = Args(3)
    
    ux = np.cos(np.deg2rad(a))
    uy = 0
    uz = np.sin(np.deg2rad(a))
    
    f   =  fgL[0]
    g   =  fgL[1]
    L   =  fgL[2]
    
    st  = np.sin(L - np.arctan(g/(f+1.0e-10)))
    
    u = vf.ifelse(st>0,
                  fgL.Constant([ux,uy,uz]),
                  fgL.Constant([ux,uy,-uz]))
    return u




    

m = 14.8 + 458.8 + 48.5 + 62.7 + 24.3 +50 + 9.5 + 293.3 + 60.3 

ac = (2*(1361)/(3e8))*(180**2)/m


beta   = .2/5.94
#beta = Beta(180*180,m)

acrank = .48
DT = 6.28*90/12

ode = SailModel(1.0,beta)



integA = ode.integrator(.1, vf.PyVectorFunction(3,3,Inc),[1,2,5])
integA.setAbsTol(1.0e-9)
integA.Adaptive=True

integI = ode.integrator(.1, IncLaw(a),[1,2,5])
integI.setAbsTol(1.0e-9)
integI.Adaptive=True

integ = ode.integrator(.1)
integ.setAbsTol(1.0e-10)
integ.Adaptive=True



#integ = ode.integrator(.1)

X = np.zeros((10))
X[0] =  acrank
X[5]=.0
X[7] =  1
X[9] =  1





t0 = time.perf_counter()
TrajIG = integI.integrate_dense(X,DT)
tf = time.perf_counter()

print(tf-t0)

#FDDerivChecker(ode.vf(),TrajIG[10])


phase = ode.phase(Tmodes.LGL3,TrajIG,300)
phase.addBoundaryValue(PhaseRegs.Front, range(1,7), X[1:7])
phase.addLUNormBound(PhaseRegs.Path,[7,8,9],.7,1.3)
phase.addLowerVarBound(PhaseRegs.Path,7, np.cos(np.deg2rad(80)),1.0)

phase.addLUVarBound(PhaseRegs.Path,0,.4795,.4805)


def VCirc():
    X = Args(3)
    p = X[0]
    f = X[1]
    g = X[2]
    
    a = p/(1-f**2 -g**2)
    e = f**2 + g**2
    return e
    #return vf.stack(p-acrank,e)
    #return p-acrank
    
def IncObj():
    X = Args(2)
    i = vf.arctan(X.norm())*2*180/np.pi
    return -i/1000

def Rfunc():
    X = Args(4)
    p = X[0]
    f = X[1]
    g = X[2]
    L = X[3]
    sinL = vf.sin(L)
    cosL = vf.cos(L)
        
    w = 1.+f*cosL +g*sinL
    return p/w
    
    

    
#phase.addLowerFuncBound(PhaseRegs.Path,Rfunc(),[0,1,2,5],.47,1.0)
#phase.addUpperFuncBound(PhaseRegs.Path,Rfunc(),[0,1,2,5],.49,1.0)

#phase.addBoundaryValue(PhaseRegs.Path, [0], [acrank])


#phase.addEqualCon(PhaseRegs.Back,VCirc(),range(0,3))
phase.addUpperDeltaTimeBound(DT*1.01)

phase.addStateObjective(PhaseRegs.Back,IncObj(),[3,4])

#phase.optimizer.OptLSMode = ast.Solvers.LineSearchModes.L1
phase.optimizer.MaxLSIters = 1

phase.optimizer.MaxAccIters = 450
phase.optimizer.PDStepStrategy = ast.Solvers.PDStepStrategies.MaxEq
phase.optimizer.QPPivotPerturb =8
phase.optimizer.BoundFraction = .999
#phase.optimizer.OptBarMode = ast.Solvers.BarrierModes.PROBE

phase.optimizer.deltaH = 1.0e-8
phase.optimizer.incrH = 8
phase.optimizer.decrH = .33
#phase.solve_optimize()

'''
phase.refineTrajManual(1500)

phase.solve_optimize()
'''

#TrajIG = phase.returnTraj()






print((TrajIG[0][7:10]))

TrajXY = [ast.Astro.modified_to_cartesian(T[0:6],1) for T in TrajIG]

TT = np.array(TrajIG).T

fig,axs = plt.subplots(7,1)
for i in range(0,6):
    axs[i].plot(TT[6],TT[i])
    
axs[6].plot(TT[6],2*np.arctan((TT[3]**2 + TT[4]**2)**.5)*180/np.pi)
axs[6].plot(TT[6],TT[6]*88.2*(beta)/(2*np.pi*np.sqrt((acrank**3)/(1-beta))))

plt.show()



plot = TBPlot(ode)

plot.addTraj(TrajXY, "IG",'k')

plot.Plot3d()









