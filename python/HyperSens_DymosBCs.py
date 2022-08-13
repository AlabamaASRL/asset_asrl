import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes    = oc.ControlModes

'''
Space Shuttle Reentry
https://openmdao.github.io/dymos/examples/reentry/reentry.html
'''

##############################################################################
g0 = 32.2 
W  = 203000

Lstar = 80000.0     ## feet
Tstar = 60.0       ## sec
Mstar = W/g0         ## slugs

Vstar   = Lstar/Tstar
Fstar   = Mstar*Lstar/(Tstar**2)
Astar   = Lstar/(Tstar**2)
Rhostar = Mstar/(Lstar**3)
BTUstar = 778.0*Lstar*Fstar
Mustar  = (Lstar**3)/(Tstar**2)



Re = 2090200          /Lstar
S  = 2690.0           /(Lstar**2)
m  = (W/g0)           /Mstar
mu = (0.140765e17)    /Mustar
rho0 =.002378        /Rhostar
h_ref = 23800         /Lstar

a0 = -.20704
a1 = .029244

b0 = .07854
b1 = -.61592e-2
b2 = .621408e-3

c0 =  1.0672181
c1 = -.19213774e-1
c2 = .21286289e-3
c3 = -.10117e-5

##############################################################################



class ShuttleReentry(oc.ode_x_u.ode):
    def __init__(self):
        ############################################################
        args  = oc.ODEArguments(5,2)
        
        h       = args.XVar(0)
        theta   = args.XVar(1)
        v       = args.XVar(2)
        gamma   = args.XVar(3)
        psi     = args.XVar(4)
        
        alpha   = args.UVar(0)
        beta    = args.UVar(1)
        
        alphadeg = (180.0/np.pi)*alpha
        
        CL  = a0 + a1*alphadeg
        CD  = b0 + b1*alphadeg + b2*(alphadeg**2)
        rho = rho0*vf.exp(-h/h_ref)
        r   = h + Re
        
        
        L   = 0.5*CL*S*rho*(v**2)
        D   = 0.5*CD*S*rho*(v**2)
        g   = mu/(r**2)
        
        sgam = vf.sin(gamma)
        cgam = vf.cos(gamma)
        
        sbet = vf.sin(beta)
        cbet = vf.cos(beta)
        
        spsi = vf.sin(psi)
        cpsi = vf.cos(psi)
        
        hdot     = v*sgam
        thetadot = (v/r)*cgam*cpsi
        vdot     = -D/m - g*sgam
        gammadot = (L/(m*v))*cbet +cgam*(v/r - g/v)
        psidot   = L*sbet/(m*v*cgam) + (v/(r*vf.cos(theta)))*cgam*spsi*vf.sin(theta)
        
    
        ode = vf.stack(hdot,thetadot,vdot,gammadot,psidot)
        ##############################################################
        super().__init__(ode,5,2)

##############################################################################
tf  = 2500/Tstar

ht0 = 260000/Lstar
htf = 80000/Lstar
vt0  = 25600/Vstar
vtf  = 2500 /Vstar



gammat0 = np.deg2rad(-1.0)
gammatf = np.deg2rad(-5.0)
psit0   = np.deg2rad(90.0)

thetatf = .5

ts = np.linspace(0,tf,100)

for t in ts:
    X = np.zeros((8))
    X[0] = ht0*(1-t/tf) + htf*t/tf
    X[1] = thetaf*t/tf
    X[2] = vt0*(1-t/tf) + vtf*t/tf
    X[3]
    


 




