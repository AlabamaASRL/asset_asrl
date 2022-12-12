import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import random as rand
import time

from DerivChecker import FDDerivChecker

vf    = ast.VectorFunctions
Args = vf.Arguments



def C2(tol = 1.0e-10):
    psi = Args(1)[0]
    sqsi = vf.sqrt(psi)
    
    return vf.ifelse(psi>tol, 
              (1.0 - vf.cos(sqsi)) / psi,
              (1.0 - vf.cosh(vf.sqrt(-psi))) / psi)
def C3(tol = 1.0e-10):
    psi = Args(1)[0]
    sqsi = vf.sqrt(psi)
    
    return vf.ifelse(psi>tol, 
              (sqsi - vf.sin(sqsi)) / (sqsi * psi),
              (vf.sinh(vf.sqrt(-psi)) - vf.sqrt(-psi)) / vf.sqrt(-psi * psi * psi))
        
    
    

def Funiv(mu):
    
    X0,dt,r,drv,alpha = Args(5).tolist()
    
    X02 = X0**2
    
    X03 = X0**3
    
    psi = X02*alpha
    
    c2 = C2().eval(psi)
    c3 = C3().eval(psi)
    
    Nargs = vf.stack(Args(5).head(4),psi,c2,c3)
    
    #############################################
    
    X0,dt,r,drv,psi,c2,c3 = Args(7).tolist()

    X02 = X0**2
    X03 = X0**3
    
    
    X0tOmPsiC3 = X0 * (1.0 - psi * c3)
    X02tC2 = X02 * c2
    
    FU  = np.sqrt(mu)*dt - X03 * c3 - drv * X02tC2 - r * X0tOmPsiC3
    dFU = -(X02tC2 + drv * X0tOmPsiC3 + r * (1.0 - psi * c2))
    
    FU  = FU(Nargs)
    dFU = dFU(Nargs)
    
    
    X0 = vf.ScalarRootFinder(FU,dFU,10,1.0e-12)
    
    return vf.stack([X0,Args(5).tail(4)])



def FGs(mu):

    SQM = np.sqrt(mu)
    
    X0,dt,r,drv,alpha = Args(5).tolist()
    
    X02 = X0**2
    
    X03 = X0**3
    
    psi = X02*alpha
    
    c2 = C2().eval(psi)
    c3 = C3().eval(psi)
    
    Nargs = vf.stack(Args(5).head(4),psi,c2,c3)
    
    #############################################
    
    X0,dt,r,drv,psi,c2,c3 = Args(7).tolist()
    X02 = X0**2
    
    X0tOmPsiC3 = X0 * (1.0 - psi * c3)
    X02tC2 = X02 * c2
    rs = (X02tC2 + drv * X0tOmPsiC3 + r * (1.0 - psi * c2))
    
    f = 1.0 - X02 * c2 / r;
    g = dt - (X02 * X0) * c3 / SQM;
    fdot = X0 * (psi * c3 - 1.0) * SQM / (rs * r);
    gdot = 1.0 - c2 * (X02) / rs;
    
    return vf.stack([f,g,fdot,gdot]).eval(Nargs)

def ApplyRVFG():
    RVFG = Args(10)
    
    R,V = RVFG.head(6).tolist([(0,3),(3,3)])
    
    f,g,fdot,gdot = RVFG.tail(4).tolist()
    
    Rf = f*R + g*V
    Vf = fdot*R + gdot*V
    
    return vf.stack(Rf,Vf)
    

def X0IG(mu):
    
    SQM = np.sqrt(mu)
    dt,r,drv,alpha = Args(4).tolist()
    
    X0e = SQM*dt*alpha
    
    X0 = X0e
    
    return vf.stack([X0,dt,r,drv,alpha])


def KeplerPropagator(mu):
    SQM = np.sqrt(mu)

    R,V,dt = Args(7).tolist([(0,3),(3,3),(6,1)])
    
    RV = Args(7).head(6)
    r = R.norm()
    v = V.norm()
    drv = R.dot(V)/SQM
    alpha = -(v * v) / (mu)+(2.0 / r)
    
    X0IG = SQM*dt*alpha
    
    X0IGhyp = vf.sign(dt) * vf.sqrt(-1/alpha) * vf.log(vf.abs((-2.0 * mu * alpha * dt) /
                    (R.dot(V) +vf.sign(dt)) *vf.sqrt(-mu /alpha) * (1.0 - r * alpha)))
    
    X0IG = vf.ifelse(alpha>0, X0IG,X0IGhyp)
    
    #return Funiv(mu)([X0IG,dt,r,drv,alpha] )
    
    FG = FGs(mu)( Funiv(mu)([X0IG,dt,r,drv,alpha] ) )
    
    #return Funiv(mu)([X0IG,dt,r,drv,alpha] )
   
    return ApplyRVFG()([RV,FG])
   

    
ast.SoftwareInfo()

np.set_printoptions(precision  = 3, linewidth = 200)

mu = 1.00
F1 = KeplerPropagator(mu)

ode = ast.Astro.Kepler.ode(mu)
integ = ode.integrator(.001)
integ.setAbsTol(1.0e-14)



F2 = ast.Astro.Kepler.KeplerPropagator(mu).vf()
F3 = ast.Astro.kptest(mu)

v0 = 1.4
tf = 4*np.pi

X =  [1,0,.1,0,v0,.0, tf]
X2 = [1,0,.1,0,v0,.0,0,tf]

J1 = F3.jacobian(X)
J2 = integ.jacobian(X2)



print(F2.adjointhessian(X,range(1,7))-F3.adjointhessian(X,range(1,7)))

k =100000
#F1.SpeedTest(X,k)
F2.SpeedTest(X,k)
F3.SpeedTest(X,k)



#F2.rpt(X,1000)
#F3.rpt(X,1000)


#F1.rpt(X,100000)

F2.vf().rpt(X,100000)

F3.rpt(X,100000)


    
    
    
    
    
    
    
    
    



    
    
    
    

    
    
    
    
    

    


Funiv(1)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    