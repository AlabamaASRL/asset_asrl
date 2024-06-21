import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments

'''
Space Shuttle Reentry
Betts, J.T. Practical methods for Optimal Control and Estimation Using Nonlinear Programming, Cambridge University Press, 2009
'''

################### Non Dimensionalize ##################################
g0 = 32.2 
W  = 203000


Lstar = 1.0     ## feet
Tstar = 1.0         ## sec
Mstar = 1         ## slugs

Vstar   = Lstar/Tstar
Fstar   = Mstar*Lstar/(Tstar**2)
Astar   = Lstar/(Tstar**2)
Rhostar = Mstar/(Lstar**3)
BTUstar = 778.0*Lstar*Fstar
Mustar  = (Lstar**3)/(Tstar**2)

print(BTUstar)

tmax = 2500            /Tstar
Re = 20902900          /Lstar
S  = 2690.0            /(Lstar**2)
m  = (W/g0)            /Mstar
mu = (0.140765e17)     /Mustar
rho0 =.002378          /Rhostar
h_ref = 23800          /Lstar

a0 = -.20704
a1 = .029244

b0 = .07854
b1 = -.61592e-2
b2 = .621408e-3

c0 =  1.0672181
c1 = -.19213774e-1
c2 = .21286289e-3
c3 = -.10117e-5

Qlimit = 70.0

##############################################################################
class ShuttleReentry(oc.ODEBase):
    def __init__(self):
        
        Xvars = 5
        Uvars = 2
        
        ############################################################
        XtU  = oc.ODEArguments(Xvars,Uvars)
        
        
        h,theta,v,gamma,psi = XtU.XVec().tolist()
        
        alpha,beta = XtU.UVec().tolist()
        
        
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
        tantheta = vf.tan(theta)
        
        hdot     = v*sgam
        thetadot = (v/r)*cgam*cpsi
        vdot     = -D/m - g*sgam
        gammadot = (L/(m*v))*cbet +cgam*(v/r - g/v)
        psidot   = L*sbet/(m*v*cgam) + (v/(r))*cgam*spsi*tantheta
        
    
        ode = vf.stack([hdot,thetadot,vdot,gammadot,psidot])
        
        Vgroups = {}
        Vgroups[('h','altitude')] =  h
        Vgroups[("v","velocity")] =  v
        Vgroups["theta"] =  theta
        Vgroups["gamma"] =  gamma
        Vgroups["psi"]   =  psi
        Vgroups[("alpha","AoA")] =  alpha
        Vgroups["beta"]  =  beta
        Vgroups[("t","time")] =  XtU.TVar()

        
        
        ##############################################################
        super().__init__(ode,Xvars,Uvars,Vgroups = Vgroups)

def QFunc():
    
    h,v,alpha = Args(3).tolist()
    alphadeg = (180.0/np.pi)*alpha
    rhodim = rho0*vf.exp(-h/h_ref)*Rhostar
    vdim = v*Vstar
    qr = 17700*vf.sqrt(rhodim)*((.0001*vdim)**3.07)
    qa = c0 + c1*alphadeg + c2*(alphadeg**2)+ c3*(alphadeg**3)
    return qa*qr
 
#############################################################################

def Plot(Traj1):
    TT1 = np.array(Traj1).T

    fig, axs = plt.subplots(4,1)

    axs[0].plot(TT1[5]*Tstar/60.0,TT1[0]*Lstar/5280,color='k')
    axs[0].set_ylabel("Altitude (Miles)")


    axs[1].plot(TT1[5]*Tstar/60.0,TT1[2]*Vstar,color='k')

    axs[1].set_ylabel(r"Velocity $\frac{ft}{s}$")


    axs[2].plot(TT1[5]*Tstar/60.0,np.rad2deg(TT1[6]),color='k')
    axs[2].plot(TT1[5]*Tstar/60.0,np.rad2deg(TT1[7]),color='k',linestyle='dotted')
    
    axs[2].set_ylabel("Angle(deg)")

    qfunc = QFunc().eval(8,[0,2,6])

    qs1 = [qfunc(T)[0] for T in Traj1]

    axs[3].plot(TT1[5]*Tstar/60.0,qs1,color='k')
    axs[3].set_ylabel(r"Qdot $(\frac{BTU}{ft^2 *s})$")


    for i in range(0,4):
        axs[i].grid(True)
        axs[i].set_xlabel("Time (min)")
        axs[i].legend()

    fig.set_size_inches(8.0, 11.0, forward=True)
    fig.tight_layout()

    plt.show()
    



if __name__ == "__main__":
    ##########################################################################
    tf  = 2000/Tstar

    ht0  = 260000/Lstar
    htf  = 80000 /Lstar
    vt0  = 25600/Vstar
    vtf  = 2500 /Vstar

    
    gammat0 = np.deg2rad(-1.0)
    gammatf = np.deg2rad(-5.0)
    psit0   = np.deg2rad(90.0)


    ts = np.linspace(0,tf,200)

    TrajIG = []
    for t in ts:
        X = np.zeros((8))
        X[0] = ht0*(1-t/tf) + htf*t/tf
        X[1] = np.deg2rad(10.0)*(1-t/tf)
        X[2] = vt0*(1-t/tf) + vtf*t/tf
        X[3] = gammat0*(1-t/tf) + gammatf*t/tf
        X[4] = psit0
        X[5] = t
        X[6] =.00
        X[7] =.00
        TrajIG.append(np.copy(X))
        
        
    ################################################################

    ode = ShuttleReentry()
    tstar = 60
    lstar = 100000
    
    phase = ode.phase("LGL3",TrajIG,50)
    
    phase.addStaticParamVgroup(0,"Qupper")
    phase.addStaticParamVgroups({"Qupper":[0]})

    phase.setAdaptiveMesh(True)

    phase.setAutoScaling(True)    
    phase.setUnits(h = lstar,
                   v = lstar/tstar,
                   t = tstar)



    
    phase.addBoundaryValue("Front",range(0,6),TrajIG[0][0:6])
    
    phase.addLUVarBound("Path","theta",np.deg2rad(-89.0),np.deg2rad(89.0))
    phase.addLUVarBound("Path","gamma",np.deg2rad(-89.0),np.deg2rad(89.0))

    phase.addLUVarBound("Path","AoA",np.deg2rad(-70.0),np.deg2rad(70.0))
    phase.addLUVarBound("Path","beta" ,np.deg2rad(-60.0),np.deg2rad(60.0))
    
    
    phase.addUpperDeltaTimeBound(tmax,1.0)
    
    phase.addBoundaryValue("Back" 
                              ,["h","v","gamma"]
                              ,[htf,vtf,gammatf])
    
    phase.addBoundaryValue("Back","theta",np.deg2rad(15.0))

    
    
    ## Bounding the heat rate with static param
    phase.addInequalCon("Path",QFunc()(Args(4).head(3))-Args(4)[3],
                               ["h","v","alpha"],[],
                               "Qupper",
                               AutoScale="auto")
    
    
    ## Minimize it
    
    Qscale = 1
    Iscale = 1/tstar
    
    phase.addStaticParams([100,50*tf],[1,(tstar*50)])  

    phase.addIntegralParamFunction(QFunc(),["h","v","alpha"],1,AutoScale="auto")

    '''
    s1=phase.addValueObjective("StaticParams","Qupper",Qscale,AutoScale="auto")
    s2=phase.addValueObjective("StaticParams",1,Iscale,AutoScale="auto")
    '''
    def ObjT():
        s1,s2 = Args(2).tolist()
        return s1*Qscale + s2*Iscale
    
    s1=phase.addStateObjective("StaticParams",ObjT(),[0,1])
    s2=s1
    
    #phase.SyncObjectiveScales=False
    phase.addLowerDeltaTimeBound(1900,1.0)
    

    

    phase.setThreads(8,8)
    
    phase.optimizer.set_SoeLSMode("L1")
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_PrintLevel(0)
    phase.optimizer.MaxLSIters =1
    phase.optimize()
    
    print(phase.returnStaticParams())
    print(phase.returnStateObjectiveScales(s1))
    print(phase.returnStateObjectiveScales(s2))
    
    
    
    Traj = phase.returnTraj()
    
   
    Plot(Traj)



