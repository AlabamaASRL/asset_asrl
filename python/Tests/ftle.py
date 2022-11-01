import numpy as np
import matplotlib.pyplot as plt
import asset as ast
from mpl_toolkits.mplot3d import Axes3D
#import time
import math

from asset_asrl.Astro.AstroModels import CR3BP

import asset_asrl.Astro.Constants as c
 
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags


def computexdot(x, mu, jv):
    Omu = 1.0 - mu;
    dx = (x[0] + mu);
    rx = (x[0] + mu - 1.0);
    d = np.sqrt(dx*dx + x[1] * x[1] + x[2] * x[2]);
    r = np.sqrt(rx*rx + x[1] * x[1] + x[2] * x[2]);
    yd2 = (x[4]*x[4])
    dnorm = (x[0]*x[0]) + (x[1]*x[1])
    drterm = 2.0*(((1.0-mu)/d) + (mu/r))
    #print(dnorm, drterm,jv, yd2)
    xdot = np.sqrt(dnorm + drterm - jv - yd2)
    return xdot

def computejv(x, mu):
    Omu = 1.0 - mu;
    dx = (x[0] + mu);
    rx = (x[0] + mu - 1.0);
    d = np.sqrt(dx*dx + x[1] * x[1] + x[2] * x[2]);
    r = np.sqrt(rx*rx + x[1] * x[1] + x[2] * x[2]);
    yd2 = (x[4]*x[4])
    xd2 = (x[3]*x[3])
    x2 = x[0]*x[0]
    y2 = x[1]*x[1]
    drterm = 2.0*(((1.0-mu)/d) + (mu/r))
    jv = x2 + y2 + drterm - (xd2 + yd2)
    return jv
def CR3BP2(mu,DoLT = False,ltacc=.05):
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


mu = 0.0121505856
cr3bp = CR3BP(c.MuEarth,c.MuMoon,c.LD)

ode=CR3BP2(mu)
ode = oc.ode_x.ode(ode,6)

print("S")
integ = oc.TestIntegratorX(ode,"DOPRI87",np.pi/1000)
integ.EnableVectorization=True

print(integ.IRows())
#integ = cr3bp.integrator(np.pi/10000)

integ.Adaptive=True
integ.setAbsTol(1.0e-12)
integ.MaxStepChange=4
#integ.FastAdaptiveSTM=False

GMearth = 3.9860044189e14/1e9
LD = 384000.0
ts = np.sqrt((LD*LD*LD)/GMearth)
ry1 = -192394.0/(LD)
ry2 = -76957.0/(LD)
rydot1 = -.4591/(LD)*ts
rydot2 = .1793/(LD)*ts


disc = 500
x = np.linspace(ry1, ry2, disc)
y = np.linspace(rydot1, rydot2 , disc)
initvecs = []
endvecs = []
ydotval = []
yval = []
T = 3.5
jv = computejv([1.155682, 0, 0, 0, 0 ,0], mu)

FullInitStates = []
FullInitTimes = []

for i in range(0, disc):
    for j in range(0, disc):
        init =np.array([0.0,x[i],	0.0,	0.0, y[j], 0.0,	0.0])
        init[3] = computexdot(init, mu, jv)
        FullInitStates.append(init)
        FullInitTimes.append(T)
        
#STMS = integ.integrate(FullInitStates[0], FullInitTimes[0])

print(len(FullInitStates),len(FullInitTimes))

import time
t00 = time.perf_counter()
STMS = integ.integrate_stm_parallel(FullInitStates, FullInitTimes, 20)
tff = time.perf_counter()
print(tff-t00)
FTLEvec = []
for entry in STMS:
    stm = entry[1][0:6][:,0:6]
    C = np.dot(stm.T, stm)
    w, v = np.linalg.eig(C)
    maxeig = np.sqrt(max(w))
    FTLE = (1/T)*np.log(maxeig)
    FTLEvec.append(FTLE)
#np.resize(FTLEvec, (disc, disc))

fig = plt.figure()
#ax = fig.gca(projection = '3d')
FTLEmat = np.array(FTLEvec).reshape(disc,disc)

im = plt.imshow((FTLEmat.T)[::-1], cmap=plt.cm.Greys, interpolation='none', extent = [ry1*LD, ry2*LD, rydot1*LD/ts, rydot2*LD/ts], aspect = 'auto')
 
plt.colorbar(im);

#plt.scatter(yval, ydotval, c=FTLEval, cmap='Greys')
plt.title("FTLE Values in Earth-Moon CR3BP")
plt.ylabel("ydot (km/s)")
plt.xticks([-192394, -163535, -134676, -105816, -76957.2])
plt.yticks([-.5891, -.397, -.2049, -.0128, .1793])
plt.xlabel("y (km)")
plt.show()
plt.show()