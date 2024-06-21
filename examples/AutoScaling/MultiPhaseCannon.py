import numpy as np
import asset_asrl as ast
import asset_asrl.VectorFunctions as vf
import asset_asrl.OptimalControl as oc
from asset_asrl.VectorFunctions import Arguments as Args

import matplotlib.pyplot as plt



'''
This example was taken from the Dymos Optimal control library. It is an excellent
teaching example of what ODE parameters are used for.
https://openmdao.github.io/dymos/examples/multi_phase_cannonball/multi_phase_cannonball.html
Only difference is that we use a simple exponenital atnospheric density function

Assuming constant materiel density, and launch energy we need to find
the radius of the cannon ball that maximizes range.
'''

##########################################

g0      =  9.81 
Lstar   =  1000           ## m
Tstar   =  60.0           ## sec
Mstar   =  10             ## kgs
Vstar   =  Lstar/Tstar

CD      = .5
RhoAir  = 1.225     
RhoIron = 7870      
h_scale = 8.44e3    
E0      = 400000    
g       = g0

###########################################


def MFunc(rad,RhoIron):return (4/3)*(np.pi*RhoIron)*(rad**3)
def SFunc(rad):  return np.pi*(rad**2)
    
    
##############################################################################

class Cannon(oc.ODEBase):
    def __init__(self, CD,RhoAir,RhoIron,h_scale,g):
        ############################################################
        args  = oc.ODEArguments(4,0,1)
        
        v     = args.XVar(0)
        gamma = args.XVar(1)
        h     = args.XVar(2)
        r     = args.XVar(3)
        
        rad = args.PVar(0)
        
        S    = SFunc(rad)
        M    = MFunc(rad,RhoIron)
        
        rho     = RhoAir * vf.exp(-h / h_scale)
        
        D       = (0.5*CD)*rho*(v**2)*S
        
        vdot     = -D/M - g*vf.sin(gamma)
        gammadot = -g*vf.cos(gamma)/v
        hdot     = v*vf.sin(gamma)
        rdot     = v*vf.cos(gamma)
        
        ode = vf.stack([vdot,gammadot,hdot,rdot])
        
        ##############################################################
        Vgroups = {}
        Vgroups["v"]=v
        Vgroups["gamma"]=gamma
        Vgroups["h"]=h
        Vgroups[("r","range")]=r
        Vgroups["rad"]=rad
        Vgroups["t"]=args.TVar()

        super().__init__(ode,4,0,1,Vgroups=Vgroups)
        
        self.apogee_event = v*vf.sin(gamma)
        self.ground_contact_event = h
        
        
def EFunc():
    v,rad =  Args(2).tolist()
    M = MFunc(rad,RhoIron)
    E = 0.5*M*(v**2)
    return E - E0

##############################################################################        
def Plot(Ascent,Descent):
    AT = np.array(Ascent).T
    DT = np.array(Descent).T

    
    AT[1]*=180/np.pi
    DT[1]*=180/np.pi

    

    fig = plt.figure()
    ax0 = plt.subplot(421)
    ax1 = plt.subplot(423)
    ax2 = plt.subplot(425)
    ax3 = plt.subplot(427)
    axs =[ax0,ax1,ax2,ax3]

    labs = [r'$v\; \frac{m}{s}$',r'$\gamma\;^\circ$', r'Altitude $m$', r' Range $m$']

    for i in range(0,4):
        
        axs[i].plot(AT[4],AT[i],color='r')
        axs[i].plot(DT[4],DT[i],color='b')
        axs[i].set_ylabel(labs[i])
        axs[i].set_xlabel(r"$t$")
        
        axs[i].grid(True)


    ax4 = plt.subplot(122)
    ax4.plot(AT[3],AT[2],color='r',label='Ascent')
    ax4.plot(DT[3],DT[2],color='b',label='Descent')
    ax4.grid(True)


    ax4.set_ylabel(" Altitude (m) ")
    ax4.set_xlabel(" Down Range (m) ")
    ax4.legend()
    
    fig.set_size_inches(15.0, 7.5, forward=True)
    fig.tight_layout()

    plt.show()
        

##############################################################################        
if __name__ == "__main__":

    rad0   = .1 
    h0     = 100 
    r0     = 0
    m0     = MFunc(rad0,RhoIron)
    gamma0 = np.deg2rad(45)
    v0     = np.sqrt(2*E0/m0)*.99
    
    
    
    ode = Cannon(CD,RhoAir,RhoIron,h_scale,g)
    integ = ode.integrator(.01)
    integ.setAbsTol(1.0e-12)    
    
    IG = np.zeros((6))
    IG[0] = v0
    IG[1] =gamma0
    IG[2] = h0
    IG[3] = r0
    IG[5] = rad0
    
    
    XtP0 = ode.make_input(v=v0,gamma=gamma0,h=h0,r=r0,rad=rad0)
    
    AscentIG = integ.integrate_dense(XtP0,60,[(ode.apogee_event,0,1)])[0]

    DescentIG = integ.integrate_dense(AscentIG[-1],
                                      AscentIG[-1][4]+ 1000,
                                      [(ode.ground_contact_event,0,1)])[0]
    

    ##########################################################################
    
   
    
    tmode = "LGL5"
    nsegs = 128
    
    units = ode.make_units(v=Vstar,h=Lstar,r=Lstar,rad=Lstar)
    
    aphase = ode.phase(tmode,AscentIG,nsegs)
    aphase.setAutoScaling(True)
    aphase.setUnits(units)
    aphase.addLowerVarBound("ODEParams","rad",0.0)
    aphase.addLowerVarBound("Front","gamma",0.0)
    aphase.addBoundaryValue("Front",["h","r","t"],[h0,r0,0])
    
    aphase.addInequalCon("Front",EFunc(),["v"],["rad"],[])
    aphase.addBoundaryValue("Back","gamma",0.0)
        
    dphase = ode.phase(tmode,DescentIG,nsegs)
    dphase.setAutoScaling(True)
    dphase.setUnits(units)
    
    dphase.addBoundaryValue("Back","h",0.0)
    dphase.addValueObjective("Back","r",-1.0)
    
    ocp = oc.OptimalControlProblem()
    ocp.setAutoScaling(True)
    ocp.addPhase(aphase)
    ocp.addPhase(dphase)
    
    # Enforce continuity in time dependent vars
    ocp.addForwardLinkEqualCon(aphase,dphase,range(0,5))
    # Enforce continuity in ODEParams
    ocp.addParamLinkEqualCon(aphase,dphase,"ODEParams","rad")
    
    ocp.optimize()
    
    
    Ascent  = aphase.returnTraj()
    Descent = dphase.returnTraj()
    
    gammaopt = ode.get_vars("gamma",Ascent[0])[0]
    ropt,radopt = ode.get_vars(["r","rad"],Descent[-1])
    
    
    print("Launch Angle:",gammaopt*180/np.pi," deg")
    print("Optimized Range:",ropt," m")
    print("Optimized Radius:"  ,radopt," cm")

    Plot(Ascent,Descent)

    ##########################################################################


