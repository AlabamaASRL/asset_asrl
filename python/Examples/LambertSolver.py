import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import random as rand
import time

vf    = ast.VectorFunctions
Args = vf.Arguments



def StumpfCS():
    psi = Args(1)[0]
    sqsi = vf.sqrt(psi)
    
    Cell = (1.0 - vf.cos(sqsi)) / psi
    Sell = (sqsi - vf.sin(sqsi)) / (sqsi * psi)
    
    Chyp = (1.0 - vf.cosh(vf.sqrt(-psi))) / psi
    Shyp = (vf.sinh(vf.sqrt(-psi)) - vf.sqrt(-psi)) / vf.sqrt(-psi * psi * psi)
    
    return vf.ifelse(psi>0.0, 
              vf.stack(Cell,Sell),
              vf.stack(Chyp,Shyp))


def LamUniv(mu):
    
    sqm = np.sqrt(mu)
    
    psi,r1,r2,A,dt = Args(5).tolist()
    
    CS = StumpfCS()(psi)
    
    NewArgs = vf.stack(Args(5),CS)
    
    #######################################
    
    psi,r1,r2,A,dt,C,S = Args(7).tolist()
    
    Cp = (1-psi*S-2*C)/(2*psi)
    Sp = (C-3*S)/(2*psi)
    
    y = r1+r2 - A*(1.0 - psi*S)/vf.sqrt(C)
    x = vf.sqrt(y/C)
    dtsqm = S*(x**3) + A*vf.sqrt(y)
    
    F = dtsqm - dt*sqm
    
    dFdpsi= x^3*(Sp-1.5*S*Cp/C)+0.125*A*(3*S*vf.sqrt(y)/C+A/x)
    
    F  = F(NewArgs)
    dFdpsi = dFdpsi(NewArgs)
    
    return  vf.ScalarRootFinder(F,dFdpsi,10,1.0e-12)

def LamUnivF(mu):
    
    sqm = np.sqrt(mu)
    
    psi,r1,r2,A,dt = Args(5).tolist()
    
    CS = StumpfCS()(psi)
    
    NewArgs = vf.stack(Args(5),CS)
    
    #######################################
    
    psi,r1,r2,A,dt,C,S = Args(7).tolist()
    
    Cp = (1-psi*S-2*C)/(2*psi)
    Sp = (C-3*S)/(2*psi)
    
    y = r1+r2 - A*(1.0 - psi*S)/vf.sqrt(C)
    x = vf.sqrt(y/C)
    dtsqm = S*(x**3) + A*vf.sqrt(y)
    
    F = dtsqm #- dt*sqm
    

    return  F.eval(NewArgs)



def CalFGGD(mu):
    sqm = np.sqrt(mu)
    
    psi,r1,r2,A,dt = Args(5).tolist()
    
    CS = StumpfCS()(psi)
    
    NewArgs = vf.stack(Args(5),CS)
    
    #######################################
    
    psi,r1,r2,A,dt,C,S = Args(7).tolist()
    
    y = r1+r2 - A*(1.0 - psi*S)/vf.sqrt(C)
    
    f = 1-y/r1
    g = A*vf.sqrt(y/mu)
    gd = 1-y/r2
    
    return vf.stack(f,g,gd)(NewArgs)

def ApplyFGGD():
    
    R1,R2,f,g,gd = Args(9).tolist([(0,3),(3,3),(6,1),(7,1),(8,1)])
    
    V1 = (R2 - f*R1)
    V2 = (R2*gd -R1)
    
    return vf.stack(V1,V2)/g
    

def LambertUniversalT(mu, nrevs = 0, longway=False):
    
    R1,R2,dt,psi = Args(8).tolist([(0,3),(3,3),(6,1),(7,1)])
    
    r1 = R1.norm()
    r2 = R2.norm()
    
    dnu = vf.arccos(R1.normalized().dot(R2.normalized()))
    A = vf.sqrt(r1*r2*(1+vf.cos(dnu)) )
    
    if(longway):A = -1*A
    
    return LamUnivF(mu)(psi,r1,r2,A,dt)
    
    

def LambertUniversal(mu, nrevs = 0, longway=False):
    
    R1,R2,dt = Args(7).tolist([(0,3),(3,3),(6,1)])
    
    r1 = R1.norm()
    r2 = R2.norm()
    
    dnu = vf.arccos(R1.normalized().dot(R2.normalized()))
    A = vf.sqrt(r1*r2*(1+vf.cos(dnu)) )
    
    if(longway):
        A = -1*A
        
        
    
    #msens = R1[0]*R2[1] - R1[1]*R2[0]
    
    
    
Test =  LambertUniversalT(1)

X1  = np.array([1,0,0])
X2  = np.array([-1.2,.1,0])
dt  = 3.5

zs = np.linspace(-3*np.pi,36*np.pi**2,1000)
fs = []
for z in zs:
    
    XX = np.zeros((8))
    XX[0:3]=X1
    XX[3:6]=X2
    XX[6]=dt
    XX[7]=z
    
    fs.append(Test(XX)[0])
    
plt.ylim(0, 1000)
plt.plot(zs,fs)

plt.show()
    
    





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




