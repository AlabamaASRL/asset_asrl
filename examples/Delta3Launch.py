import numpy as np
import asset as ast
import matplotlib.pyplot as plt

#### PIP INSTALL Basemap if you dont have it
from mpl_toolkits.basemap import Basemap

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes    = oc.ControlModes


############################################################################

g0      =  9.80665 
Lstar   =  6378145           ## m   Radius of Earth
Tstar   =  961.0             ## sec Engine Burn Time
Mstar   =  301454.0          ## kgs Inital Mass of Rocket


Astar   =  Lstar/Tstar**2
Vstar   =  Lstar/Tstar
Rhostar =  Mstar/Lstar**3
Estar   =  Mstar*(Vstar**2)
Mustar  =  (Lstar**3)/(Tstar**2)
Fstar   =  Astar*Mstar
#############################################################################

mu      = 3.986012e14     /Mustar
Re      = 6378145      /Lstar
We      = 7.29211585e-5          *Tstar

RhoAir  = 1.225        /Rhostar
h_scale = 7200         /Lstar
g       = g0           /Astar


CD = .5
S  = 4*np.pi   /Lstar**2



TS = 628500      /Fstar
T1 = 1083100     /Fstar
T2 = 110094      /Fstar

IS = 283.33364   /Tstar
I1 = 301.68      /Tstar
I2 = 467.21       /Tstar

tS = 75.2        /Tstar
t1 = 261         /Tstar
t2 = 700         /Tstar


TMS = 19290      /Mstar
TM1 = 104380     /Mstar
TM2 = 19300      /Mstar
TMPay = 4164     /Mstar


PMS = 17010     /Mstar
PM1 = 95550     /Mstar
PM2 = 16820     /Mstar

SMS = TMS - PMS
SM1 = TM1 - PM1
SM2 = TM2 - PM2

T_phase1 = 6*TS + T1
T_phase2 = 3*TS + T1
T_phase3 = T1
T_phase4 = T2

mdot_phase1 = (6*TS/IS + T1/I1)/g
mdot_phase2 = (3*TS/IS + T1/I1)/g
mdot_phase3 = T1/(g*I1)
mdot_phase4 = T2/(g*I2)


tf_phase1 = tS
tf_phase2 = 2*tS
tf_phase3 = t1
tf_phase4 = t1+t2

m0_phase1 = 9*TMS + TM1 + TM2 + TMPay
mf_phase1 = m0_phase1 - 6*PMS - (tS/t1)*PM1

m0_phase2 = mf_phase1 - 6*SMS
mf_phase2 = m0_phase2 - 3*PMS - (tS/t1)*PM1

m0_phase3 = mf_phase2 - 3*SMS
mf_phase3 = m0_phase3 - (1 - 2*tS/t1)*PM1

m0_phase4 = mf_phase3 - SM1
mf_phase4 = m0_phase4 - PM2


#############################################################################
class Delta3(oc.ode_x_u.ode):
    def __init__(self,T,mdot):
        ####################################################
        args  = oc.ODEArguments(7,3)
        
        r = args.XVec().head3()
        v = args.XVec().segment3(3)
        m = args.XVar(6)
        u = args.tail3().normalized()
        
        h       = r.norm() - Re
        rho     = RhoAir * vf.exp(-h / h_scale)
        vr      = v + r.cross(np.array([0,0,We]))
        D       = (-0.5*CD*S)*rho*(vr*vr.norm())
        
        rdot    =  v
        vdot    =  (-mu)*r.normalized_power3() + (T*u + D)/m
        
        ode = vf.stack(rdot,vdot,-mdot)
        ####################################################
        super().__init__(ode,7,3)

def TargetOrbit(at,et,it, Ot,Wt):
    rvec,vvec = Args(6).tolist([(0,3),(3,3)])
    
    hvec = rvec.cross(vvec)
    nvec = vf.cross([0,0,1],hvec)
    
    r    = rvec.norm()
    v    = vvec.norm()
    
    eps = 0.5*(v**2) - mu/r
    
    a =  -0.5*mu/eps
    
    evec = vvec.cross(hvec)/mu - rvec.normalized()
    
    i = vf.arccos(hvec.normalized()[2]) 
    
    Omega = vf.arccos(nvec.normalized()[0])
    Omega = vf.ifelse(nvec[1]>0,Omega,2*np.pi -Omega)
    W = vf.arccos(nvec.normalized().dot(evec.normalized()))
    W = vf.ifelse(evec[2]>0,W,2*np.pi-W)
    return vf.stack([a-at,evec.norm()-et,i-it,Omega-Ot,W-Wt])
    
###############################################################################

def Plot(Phase1,Phase2,Phase3,Phase4):
    ############################################

    def LatLongAlt(Traj):
        LLs = []
        for T in Traj:
            x = T[0]
            y = T[1]
            z = T[2]
            r   = np.linalg.norm(T[0:3])
            
            lat = np.rad2deg(np.arcsin(z/r))
            
            long = np.rad2deg(np.arctan2(y,x))
            if(x>0):long+=0
            elif(y>0):long+=180
            else:long-=180
            LLs.append([lat,long-80.649])
        return LLs


    def AltVelMass(Traj):
        Xs =[]
        for T in Traj:
            r   = np.linalg.norm(T[0:3])
            alt = (r-Re)*Lstar/1000
            v = np.linalg.norm(T[3:6])*Vstar/1000
            m = T[6]*Mstar
            t = T[7]*Tstar
            Xs.append([alt,v,m,t])
            
        return np.array(Xs).T

    LLs = LatLongAlt([Phase1[0],Phase2[0],Phase3[0],Phase4[0],Phase4[-1]])
            
    P1 = AltVelMass(Phase1)
    P2 = AltVelMass(Phase2)
    P3 = AltVelMass(Phase3)
    P4 = AltVelMass(Phase4)

    fig = plt.figure()
    ax0 = plt.subplot(321)
    ax1 = plt.subplot(323)
    ax2 = plt.subplot(325)
    ax3 = plt.subplot(122)    

    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)

    ax0.set_ylabel("h (km)")
    ax1.set_ylabel("v (km)/s")
    ax2.set_ylabel("M (kg)")
    ax2.set_xlabel("t (s)")

    PS = [P1,P2,P3,P4]

    for i,P in enumerate(PS):
        s = i+1
        ax0.plot(P[3],P[0])
        ax1.plot(P[3],P[1])
        ax2.plot(P[3],P[2],label=f'Stage {s}')


    
    m = Basemap(projection='lcc',
                lat_1=45.,lat_2=55,lat_0=50,lon_0=-65.,
                resolution=None,width=9000000,height=9000000,ax=ax3)

    for i in range(0,len(LLs)-1):
        lon1 = float(LLs[i][1])
        lat1 = float(LLs[i][0])
        lon2 = float(LLs[i+1][1])
        lat2 = float(LLs[i+1][0])
        m.drawgreatcircle(lon1=lon1,lat1=lat1,lon2=lon2,lat2=lat2)

    ax2.legend()
    #m.bluemarble()
    m.shadedrelief()

    m.drawparallels(np.arange(-90.,91.,30.))
    m.drawmeridians(np.arange(-180.,181.,60.))
    #m.drawmapboundary(fill_color='aqua')
    plt.title("Ground Track")
    
    fig.set_size_inches(15.0, 7.5, forward=True)
    fig.tight_layout()

    plt.show()

###############################################################################

if __name__ == "__main__":

    ast.SoftwareInfo()
    
    at = 24361140 /Lstar
    et = .7308
    Ot = np.deg2rad(269.8)
    Wt = np.deg2rad(130.5)
    istart = np.deg2rad(28.5)
    
    
    y0      = np.zeros((6))
    y0[0:3] = np.array([np.cos(istart),0,np.sin(istart)])*Re
    y0[3:6] =-np.cross(y0[0:3],np.array([0,0,We]))
    y0[3]  += 0.0001/Vstar
    print(y0[3:6]*Vstar)
    
    
    ## M0 is the only magic number in the script, just trying to find
    ## an intital terminal state that is along the orbit, downrange from KSC in
    ## the correct direction and doesnt pass through earth when LERPed from KSC
    M0   =-.05
    #M0   =-.00
    OEF  = [at,et,istart,Ot,Wt,M0]
    yf   = ast.Astro.classic_to_cartesian(OEF,mu)
    
    ts   = np.linspace(0,tf_phase4,200)
    
    IG1 =[]
    IG2 =[]
    IG3 =[]
    IG4 =[] 
    
    
    for t in ts:
        X = np.zeros((11))
        X[0:6]= y0 + (yf-y0)*(t/ts[-1])
        X[7]  = t
        
        if(t<tf_phase1):
            m= m0_phase1 + (mf_phase1-m0_phase1)*(t/tf_phase1)
            #X[0:6]=y0
            X[6]=m
            X[8:11]= vf.normalize([0,1,0])
            IG1.append(X)
        elif(t<tf_phase2):
            m= m0_phase2 + (mf_phase2-m0_phase2)*(( t-tf_phase1) / (tf_phase2 - tf_phase1))
            #X[0:6]=y0
            X[6]=m
            X[8:11]= vf.normalize([0,1,0])
            IG2.append(X)
        elif(t<tf_phase3):
            m= m0_phase3 + (mf_phase3-m0_phase3)*(( t-tf_phase2) / (tf_phase3 - tf_phase2))
            #X[0:6]=yf
            X[6]=m
            X[8:11]= vf.normalize([0,1,0])
            IG3.append(X)
        elif(t<tf_phase4):
            #X[0:6]=yf
            m= m0_phase4 + (mf_phase4-m0_phase4)*(( t-tf_phase3) / (tf_phase4 - tf_phase3))
            X[6]=m
            X[8:11]= vf.normalize([0,1,0])
            IG4.append(X)
        
    
    
    ode1 = Delta3(T_phase1,mdot_phase1)
    ode2 = Delta3(T_phase2,mdot_phase2)
    ode3 = Delta3(T_phase3,mdot_phase3)
    ode4 = Delta3(T_phase4,mdot_phase4)
    
    tmode = "LGL3"
    
    phase1 = ode1.phase(tmode,IG1,len(IG1)-1)
    phase1.addLUNormBound("Path",[8,9,10],.5,1.5)
    
    phase1.addBoundaryValue("Front",range(0,8),IG1[0][0:8])
    phase1.addBoundaryValue("Back",[7],[tf_phase1])
    
    phase2 = ode2.phase(tmode,IG2,len(IG2)-1)
    phase2.addLUNormBound("Path",[8,9,10],.5,1.5)
    phase2.addBoundaryValue("Front",[6], [m0_phase2])
    phase2.addBoundaryValue("Back", [7] ,[tf_phase2])
    
    phase3 = ode3.phase(tmode,IG3,len(IG3)-1)
    phase3.addLUNormBound("Path",[8,9,10],.5,1.5)
    phase3.addBoundaryValue("Front",[6], [m0_phase3])
    phase3.addBoundaryValue("Back", [7] ,[tf_phase3])
    
    phase4 = ode4.phase(tmode,IG4,len(IG4)-1)
    phase4.addLUNormBound("Path",[8,9,10],.5,1.5)
    phase4.addBoundaryValue("Front",[6], [m0_phase4])
    phase4.addValueObjective("Back",6,-1.0)
    phase4.addUpperVarBound("Back",7,tf_phase4,1.0)
    phase4.addEqualCon("Back",TargetOrbit(at,et,istart,Ot,Wt),range(0,6))
    
    phase1.addLowerNormBound("Path",[0,1,2],Re*.999999)
    phase2.addLowerNormBound("Path",[0,1,2],Re*.999999)
    phase3.addLowerNormBound("Path",[0,1,2],Re*.999999)
    phase4.addLowerNormBound("Path",[0,1,2],Re*.999999)
    
    
    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase1)
    ocp.addPhase(phase2)
    ocp.addPhase(phase3)
    ocp.addPhase(phase4)
    
    ocp.addForwardLinkEqualCon(phase1,phase4,[0,1,2,3,4,5,7,8,9,10])
    ocp.optimizer.set_OptLSMode("L1")
    ocp.optimizer.set_alphaRed(2.0)
    
    ocp.optimizer.MaxLSIters = 2
    
    phase1.setControlMode("BlockConstant")
    phase2.setControlMode("BlockConstant")
    phase3.setControlMode("BlockConstant")
    phase4.setControlMode("BlockConstant")

    #ocp.optimizer.KKTtol = 1.0e-9
    ocp.optimizer.PrintLevel=1
    #ocp.optimizer.CNRMode=True


    ocp.Threads=8
    ocp.optimizer.QPThreads=8

    #ocp.optimizer.SoeMode = ast.Solvers.AlgorithmModes.OPTNO
    ocp.optimize_solve()
    
    
    
    Phase1Traj = phase1.returnTraj()  # or ocp.Phase(i).returnTraj()
    Phase2Traj = phase2.returnTraj()
    Phase3Traj = phase3.returnTraj()
    Phase4Traj = phase4.returnTraj()
    
    Traj = Phase1Traj + Phase2Traj +Phase3Traj +Phase4Traj
    MEEs =[list(ast.Astro.cartesian_to_modified(T[0:6], mu))+[T[7]] for T in Traj]
    print(mu)
    
    
    TT = np.array(MEEs).T
    plt.plot(TT[6],TT[0])
    
    plt.plot(TT[6],TT[1])
    plt.plot(TT[6],TT[2])
    plt.plot(TT[6],TT[3])
    plt.plot(TT[6],TT[4])
    
    plt.plot(TT[6],TT[5])
    plt.show()
    
    print("Final Mass = ",Phase4Traj[-1][6]*Mstar,' kg')

    Plot(Phase1Traj,Phase2Traj,Phase3Traj,Phase4Traj)




