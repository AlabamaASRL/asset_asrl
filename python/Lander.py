import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from numpy import cos,sin,tan,arccos,arcsin,arctan2
from scipy.spatial.transform import Rotation as R
import MKgSecConstants as c

x = ((1.0 +5.68e-7)**2 + (4.05e-6)**2)**.5
print(1/x**.5)
input("S")

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes = oc.ControlModes
Imodes = oc.IntegralModes


class PlanarLander(oc.ode_x_x.ode):
    def __init__(self,mu,Rbod,Ascale,Mscale):
        args = oc.ODEArguments(5,2)
        alt = args[0]
        drange = args[1]
        vr = args[2]
        vt = args[3]
        m  = args[4]
        
        Tr = Ascale*(args.UVec()[0]/m)
        Tt = Ascale*(args.UVec()[1]/m)
        
        r = alt + Rbod
        
        alt_dt    = vr
        drange_dt = vt
        vr_dt     = (vt**2)/r - mu/(r**2) + Tr
        vt_dt     =  - vr*vt/r +Tt
        mdot      =  - args.UVec().norm()*Mscale
        
        xdot = vf.Stack([alt_dt,drange_dt,vr_dt,vt_dt,mdot])
        super().__init__(xdot,5,2)




# using appollo specs
MTotal   = 16000 #Kg
MDesFuel = 8200  #Kg
Tmax     = 45000 #N 
ISP      = 311   #
mdot     = Tmax/(9.81*ISP)
print(mdot)
h0       = 50000   #m
Rmoon    = 1736000 #m
Vt0      = np.sqrt(c.MuMoon/(Rmoon + h0))


lstar     = Rmoon
vstar     = np.sqrt(c.MuMoon/lstar)
tstar     = np.sqrt(lstar**3/c.MuMoon)
astar     = c.MuMoon/lstar**2
mstar     = MTotal
mdstar    = mstar/tstar

    


print("lstar: ", lstar," m")
print("tstar: ", tstar," s")

print("vstar: ", vstar," m/s")
print("astar: ", astar," m/s^2")
print("mdstar: ", mdstar," Kg/s")

mustar = (lstar**3)/(tstar**2)

mu_nd     = 1
Rbod_nd   = Rmoon/lstar
Ascale_nd = (Tmax/MTotal)/astar
Mscale_nd = mdot/mdstar
h0_nd     = h0/lstar
v0_nd     = Vt0/vstar  

print("mu_nd: ", mu_nd)
print("g_nd: ", (c.MuMoon/Rmoon**2)/astar)
print("a_nd: ", Ascale_nd)
print("a_nd: ", h0_nd)

print("Mscale_nd: ", Mscale_nd)


ode = PlanarLander(mu_nd,Rbod_nd,Ascale_nd,Mscale_nd)

IG =np.array([h0_nd,0,0,v0_nd,1,0,0,-.9])

integ = ode.integrator(.01,-Args(2).normalized()*.8,[2,3])

Traj = integ.integrate_dense(IG,.45,1000)


phase = ode.phase(Tmodes.LGL3,Traj,900)
phase.addBoundaryValue(PhaseRegs.Front,range(0,6),IG[0:6])
phase.addBoundaryValue(PhaseRegs.Back,[0,2,3],[0,0,0])
phase.addLUNormBound(PhaseRegs.Path,[6,7],.001,1.0,1.0)
#phase.addLowerVarBound(PhaseRegs.Back,4,.3)
phase.addDeltaTimeObjective(1)
phase.optimizer.MaxAccIters = 100
phase.solve_optimize()

Traj = phase.returnTraj()

TT = np.array(Traj).T



plt.plot(TT[1]*lstar/1000,TT[0]*lstar/1000)
plt.ylabel("Altitude (Km)")
plt.xlabel("Down Range Distance (Km)")
plt.grid(True)
plt.show()




