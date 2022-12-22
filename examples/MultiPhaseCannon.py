import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments


##########################################

g0      =  9.81 
Lstar   =  1000           ## m
Tstar   =  60.0           ## sec
Mstar   =  10             ## kgs
Astar   =  Lstar/Tstar**2
Vstar   =  Lstar/Tstar
Rhostar =  Mstar/Lstar**3
Estar   =  Mstar*(Vstar**2)


CD      = .5
RhoAir  = 1.225     /Rhostar
RhoIron = 7870      /Rhostar
h_scale = 8.44e3    /Lstar
E0      = 400000    /Estar
g       = g0/Astar

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
        super().__init__(ode,4,0,1)
        
        
def EFunc():
    v,rad =  Args(2).tolist()
    M = MFunc(rad,RhoIron)
    E = 0.5*M*(v**2)
    return E - E0
##############################################################################        
def Plot(Ascent,Descent):
    AT = np.array(Ascent).T
    DT = np.array(Descent).T

    AT[0]*=Vstar
    DT[0]*=Vstar

    AT[1]*=180/np.pi
    DT[1]*=180/np.pi

    AT[2]*=Lstar
    DT[2]*=Lstar

    AT[3]*=Lstar
    DT[3]*=Lstar

    AT[4]*=Tstar
    DT[4]*=Tstar


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

    rad0   = .1 /Lstar
    h0     = 100 /Lstar
    r0     = 0
    m0     = MFunc(rad0,RhoIron)
    gamma0 = np.deg2rad(45)
    v0     = np.sqrt(2*E0/m0)*.99
    
    
    
    ode = Cannon(CD,RhoAir,RhoIron,h_scale,g)
    integ = ode.integrator(.01)
    integ.Adaptive = True
    
    
    IG = np.zeros((6))
    IG[0] = v0
    IG[1] =gamma0
    IG[2] = h0
    IG[3] = r0
    IG[5] = rad0
    
    
    
    
    AscentIG = integ.integrate_dense(IG,
                                     60/Tstar,
                                     1000,
                                     lambda x:x[0]*np.sin(x[1])<0)
    DescentIG = integ.integrate_dense(AscentIG[-1],
                                      AscentIG[-1][4]+ 30/Tstar,
                                      1000,
                                      lambda x:x[2]<0)
    
    ##########################################################################
    tmode = "LGL5"
    nsegs = 128
    
    aphase = ode.phase(tmode,AscentIG,nsegs)
    aphase.addLowerVarBound("ODEParams",0,0.0,1)
    aphase.addLowerVarBound("Front",1,0.0,1.0)
    aphase.addBoundaryValue("Front",[2,3,4],[h0,r0,0])
    
    aphase.addInequalCon("Front",EFunc()*.01,[0],[0],[])
    aphase.addBoundaryValue("Back",[1],[0.0])
        
    dphase = ode.phase(tmode,DescentIG,nsegs)
    dphase.addBoundaryValue("Back",[2],[0.0])
    dphase.addValueObjective("Back",3,-1.0)
    
    ocp = oc.OptimalControlProblem()
    ocp.addPhase(aphase)
    ocp.addPhase(dphase)
    
    
    # Enforce continuatiy in time dependent vars
    ocp.addForwardLinkEqualCon(aphase,dphase,[0,1,2,3,4])
    ocp.addDirectLinkEqualCon(0,"ODEParams",[0],
                              1,"ODEParams",[0])
    
    
    ocp.optimizer.set_OptLSMode("L1")
    ocp.optimize()
    
    Ascent  = aphase.returnTraj()
    Descent = dphase.returnTraj()
    Plot(Ascent,Descent)

    ##########################################################################






