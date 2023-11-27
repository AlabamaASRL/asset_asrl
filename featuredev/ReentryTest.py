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

Lstar = 100000.0     ## feet
Tstar = 60.0         ## sec
Mstar = W/g0         ## slugs

Lstar = 1.0     ## feet
Tstar = 1.0         ## sec
Mstar = 1         ## slugs




Vstar   = Lstar/Tstar
Fstar   = Mstar*Lstar/(Tstar**2)
Astar   = Lstar/(Tstar**2)
Rhostar = Mstar/(Lstar**3)
BTUstar = 778.0*Lstar*Fstar
Mustar  = (Lstar**3)/(Tstar**2)

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

def Plot(Traj1,Traj2):
    TT1 = np.array(Traj1).T
    TT2 = np.array(Traj2).T

    fig, axs = plt.subplots(4,1)

    axs[0].plot(TT1[5]*Tstar/60.0,TT1[0]*Lstar/5280,label='No Q limit',color='r')
    axs[0].plot(TT2[5]*Tstar/60.0,TT2[0]*Lstar/5280,label='Q limited',color='b')
    axs[0].set_ylabel("Altitude (Miles)")


    axs[1].plot(TT1[5]*Tstar/60.0,TT1[2]*Vstar,label='No Q limit',color='r')
    axs[1].plot(TT2[5]*Tstar/60.0,TT2[2]*Vstar,label='Q limited',color='b')

    axs[1].set_ylabel(r"Velocity $\frac{ft}{s}$")


    axs[2].plot(TT1[5]*Tstar/60.0,np.rad2deg(TT1[6]),label='Angle of Attack No Q limit',color='r')
    axs[2].plot(TT1[5]*Tstar/60.0,np.rad2deg(TT1[7]),label='Bank Angle No Q limit',color='r',linestyle='dotted')
    axs[2].plot(TT2[5]*Tstar/60.0,np.rad2deg(TT2[6]),label='Angle of Attack  Q limited',color='b')
    axs[2].plot(TT2[5]*Tstar/60.0,np.rad2deg(TT2[7]),label='Bank Angle  Q limited',color='b',linestyle='dotted')

    axs[2].set_ylabel("Angle(deg)")

    qfunc = QFunc().eval(8,[0,2,6])

    qs1 = [qfunc(T)[0] for T in Traj1]
    qs2 = [qfunc(T)[0] for T in Traj2]

    axs[3].plot(TT1[5]*Tstar/60.0,qs1,label='No Q limit',color='r')
    axs[3].plot(TT2[5]*Tstar/60.0,qs2,label='Q limited',color='b')
    axs[3].set_ylabel(r"Q $(\frac{BTU}{ft^2 *s})$")

    axs[3].plot(TT2[5]*Tstar/60.0,np.ones_like(TT2[5])*70,label=r'Q = 70 $\frac{BTU}{ft^2 *s}$',color='k',linestyle='dashed')


    for i in range(0,4):
        axs[i].grid(True)
        axs[i].set_xlabel("Time (min)")
        axs[i].legend()

    fig.set_size_inches(8.0, 11.0, forward=True)
    fig.tight_layout()

    plt.show()
    



if __name__ == "__main__":
    ##########################################################################
    tf  = 1000/Tstar

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
        X[1] = 0
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
    
    phase = ode.phase("LGL3",TrajIG,40)
    scales = np.ones((8))
    scales[0]=lstar
    scales[2]=lstar/tstar
    scales[5] =tstar
        
    phase.setUnits(scales,[])
    phase.AutoScaling = True
    
    phase.addBoundaryValue("Front",range(0,6),TrajIG[0][0:6])
    
    phase.addLUVarBound("Path","theta",np.deg2rad(-89.0),np.deg2rad(89.0))
    phase.addLUVarBound("Path","gamma",np.deg2rad(-89.0),np.deg2rad(89.0))

    phase.addLUVarBound("Path","AoA",np.deg2rad(-90.0),np.deg2rad(90.0))
    phase.addLUVarBound("Path","beta" ,np.deg2rad(-90.0),np.deg2rad(1.0),AutoScale = "auto")
    
    
    phase.addUpperDeltaTimeBound(tmax,1.0)
    
    
    phase.addBoundaryValue("Back" 
                              ,["h","v","gamma"]
                              ,[htf,vtf,gammatf])
    
    phase.addDeltaVarObjective(1,-1.0)
    phase.setThreads(8,8)
    
    ## Our IG is bad, so i turn on line search
    phase.optimizer.set_SoeLSMode("L1")
    phase.optimizer.set_OptLSMode("L1")
    phase.optimizer.set_PrintLevel(1)
    phase.setAdaptiveMesh(True)
    phase.MeshErrorEstimator='integrator'
    #phase.MeshErrorCriteria = 'endtoend'
    phase.MeshErrFactor = 10
    #phase.MeshRedFactor = 1.1

    ## IG is bad, solve first before optimize
    phase.solve_optimize()

    input("S")

    #Refine to more segments and Reoptimize
    #phase.refineTrajManual(300)
    phase.optimize()

    Traj1 = phase.returnTraj()
    
    ## Add in Heating Rate Constraint, scale so rhs is order 1
    phase.addUpperFuncBound("Path",QFunc(),
                               ["h","v","alpha"],
                               Qlimit,AutoScale='auto')
    
    
    #phase.addLUFuncBoundNEW("Path",QFunc(),["h","v","alpha"],[],[],
    #                        -5,Qlimit,1.0,AutoScale='auto')

    
    phase.optimize()
    
    Traj2 = phase.returnTraj()
    
    print("Final Time:",Traj1[-1][5]*Tstar,"(s) , Final Cross Range:",Traj1[-1][1]*180/np.pi, " deg")
    print("Final Time:",Traj2[-1][5]*Tstar,"(s) , Final Cross Range:",Traj2[-1][1]*180/np.pi, " deg")
   

    Plot(Traj1,Traj2)


    
    


 




