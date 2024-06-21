import numpy as np
import asset_asrl as ast
import asset_asrl.VectorFunctions as vf
import asset_asrl.OptimalControl as oc
from asset_asrl.VectorFunctions import Arguments as Args

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap ## PIP INSTALL Basemap if you dont have it



############################################################################
Lstar   =  6378145           ## m   Radius of Earth
Tstar   =  961.0             ## sec Engine Burn Time
Mstar   =  301454.0          ## kgs Inital Mass of Rocket
Vstar   = Lstar/Tstar
#############################################################################

g0      =  9.80665 

mu      = 3.986012e14      
Re      = 6378145
We      = 7.29211585e-5    

RhoAir  = 1.225        
h_scale = 7200         
g       = g0           


CD = .5
S  = 4*np.pi   



TS = 628500      
T1 = 1083100     
T2 = 110094      

IS = 283.33364   
I1 = 301.68      
I2 = 467.21       

tS = 75.2        
t1 = 261         
t2 = 700         


TMS = 19290      
TM1 = 104380     
TM2 = 19300      
TMPay = 4164     

PMS = 17010     
PM1 = 95550     
PM2 = 16820     

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
class RocketODE(oc.ODEBase):
    def __init__(self,T,mdot):
        ####################################################
        XtU  = oc.ODEArguments(7,3)
        
        R = XtU.XVec().head(3)
        V = XtU.XVec().segment(3,3)
        m = XtU.XVar(6)
       
        U = XtU.UVec()
        
        h       = R.norm() - Re
        rho     = RhoAir * vf.exp(-h / h_scale)
        Vr      = V + R.cross(np.array([0,0,We]))
        
        D       = (-0.5*CD*S)*rho*(Vr*Vr.norm())
        
        Rdot    =  V
        Vdot    =  (-mu)*R.normalized_power3() + (T*U.normalized() + D)/m
        
        ode = vf.stack(Rdot,Vdot,-mdot)
        
        Vgroups = {}
        ## Multiple names allowed, just index with tuple of names
        Vgroups[("R","Position")]=R
        Vgroups[("V","Velocity")]=V
        Vgroups["U"]=U
        Vgroups[("t","time")]=XtU.TVar()
        Vgroups[("m","mass")]=m
        
        Vgroups["RV"] =[R,V]


        ####################################################
        super().__init__(ode,7,3,Vgroups = Vgroups)

def TargetOrbit(at,et,it, Ot,Wt):
    R,V = Args(6).tolist([(0,3),(3,3)])
    
    r    = R.norm()
    v    = V.norm()
    
    #Angular momentum vector
    hvec = R.cross(V)
    
    #Node vector
    nvec = vf.cross([0,0,1],hvec)
    
    # Energy
    eps = 0.5*(v**2) - mu/r
    
    # Semi-major axis
    a =  -0.5*mu/eps
    
    evec = V.cross(hvec)/mu - R.normalized()
    #Eccentrcity
    e = evec.norm()
    
    #inclination
    i = vf.arccos(hvec.normalized()[2]) 
    
    #RAAN
    O = vf.arccos(nvec.normalized()[0])
    # Quadrant Check
    O = vf.ifelse(nvec[1]>0,O,2*np.pi -O)
    
    # Argument of periapse
    W = vf.arccos(nvec.normalized().dot(evec.normalized()))
    #QuadrantCheck
    W = vf.ifelse(evec[2]>0,W,2*np.pi-W)
    
    return vf.stack([a,e,i,O,W]) - np.array([at,et,it,Ot,Wt])
    
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
            alt = (r-Re)/1000
            v = np.linalg.norm(T[3:6])/1000
            m = T[6]
            t = T[7]
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
    
    
    # Target orbital elements
    at     = 24361140 
    et     = .7308
    Ot     = np.deg2rad(269.8)
    Wt     = np.deg2rad(130.5)
    istart = np.deg2rad(28.5)
    
    
    y0      = np.zeros((6))
    y0[0:3] = np.array([np.cos(istart),0,np.sin(istart)])*Re
    y0[3:6] =-np.cross(y0[0:3],np.array([0,0,We]))
    y0[3]  += 0.00001  # cant be exactly zero,our drag equation's derivative would NAN !!!
    
    
    ## MF is the only magic number in the script, just trying to find
    ## a mean anomaly such that the terminal state on the orbit is downrange
    ## eastward from KSC in and doesnt pass through earth when LERPed from KSC
    MF   =-.05
    OEF  = [at,et,istart,Ot,Wt,MF]
    yf   = ast.Astro.classic_to_cartesian(OEF,mu)
    
    ts   = np.linspace(0,tf_phase4,1000)
    
    IG1 =[]
    IG2 =[]
    IG3 =[]
    IG4 =[] 
    
    ode1 = RocketODE(T_phase1,mdot_phase1)
    ode2 = RocketODE(T_phase2,mdot_phase2)
    ode3 = RocketODE(T_phase3,mdot_phase3)
    ode4 = RocketODE(T_phase4,mdot_phase4)
    
    for t in ts:
        RVi= y0 + (yf-y0)*(t/ts[-1])
        Ui = np.array([0,1,0])
        
        if(t<tf_phase1):
            m= m0_phase1 + (mf_phase1-m0_phase1)*(t/tf_phase1)
            ## Create ODEinputs w/ named variable groups, 0 if unspecified
            X = ode1.make_input(RV=RVi,t=t,m=m,U=Ui)
            IG1.append(X)
        elif(t<tf_phase2):
            m= m0_phase2 + (mf_phase2-m0_phase2)*(( t-tf_phase1) / (tf_phase2 - tf_phase1))
            X = ode2.make_input(RV=RVi,t=t,m=m,U=Ui)
            IG2.append(X)
        elif(t<tf_phase3):
            m= m0_phase3 + (mf_phase3-m0_phase3)*(( t-tf_phase2) / (tf_phase3 - tf_phase2))
            X = ode3.make_input(RV=RVi,t=t,m=m,U=Ui)
            IG3.append(X)
        elif(t<tf_phase4):
            m= m0_phase4 + (mf_phase4-m0_phase4)*(( t-tf_phase3) / (tf_phase4 - tf_phase3))
            X = ode4.make_input(RV=RVi,t=t,m=m,U=Ui)
            IG4.append(X)
        
    
    
    
    
    tmode = "LGL3"
    cmode = "HighestOrderSpline"
    
    nsegs1 = 16
    nsegs2 = 16
    nsegs3 = 16
    nsegs4 = 16
    
    #########################################
    phase1 = ode1.phase(tmode,IG1,nsegs1)
    phase1.setControlMode(cmode)
    
    phase1.addLUNormBound("Path","U",.5,1.5)
    phase1.addBoundaryValue("Front",["R","V","m","t"],IG1[0][0:8])
    
    #Dont want our bound to interfere with initial condition which starts at Re
    #so i relax the Earth radius constraint slightly here
    phase1.addLowerNormBound("Path","R",Re*.999999)
    
    phase1.addBoundaryValue("Back","time",tf_phase1)
    
    #########################################
    phase2 = ode2.phase(tmode,IG2,nsegs2)
    phase2.setControlMode(cmode)
    
    phase2.addLowerNormBound("Path","R",Re)
    phase2.addLUNormBound("Path","U",.5,1.5)
    
    ## Fixing initial mass and final time on first 3 phases.
    ## Since the engine cant be throttled, constraining final mass
    ## as well would be redundant and overconstrained
    phase2.addBoundaryValue("Front","mass", m0_phase2)
    phase2.addBoundaryValue("Back", "time" ,tf_phase2)
    
    #########################################
    phase3 = ode3.phase(tmode,IG3,nsegs3)
    phase3.setControlMode(cmode)
    
    phase3.addLowerNormBound("Path","R",Re)
    phase3.addLUNormBound("Path","U",.5,1.5)
    phase3.addBoundaryValue("Front","mass", m0_phase3)
    phase3.addBoundaryValue("Back", "time" ,tf_phase3)
    
    #########################################
    phase4 = ode4.phase(tmode,IG4,nsegs4)
    phase4.setControlMode(cmode)

    phase4.addLowerNormBound("Path","R",Re)
    phase4.addLUNormBound("Path","U",.5,1.5)
    phase4.addBoundaryValue("Front","mass", m0_phase4)
    phase4.addUpperVarBound("Back","time",tf_phase4)
    
    orbitidx = phase4.addEqualCon("Back",TargetOrbit(at,et,istart,Ot,Wt),["R","V"],AutoScale = "auto")
    # Maximize final mass
    phase4.addValueObjective("Back","mass",-1.0)
    

    #########################################
    
    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase1)
    ocp.addPhase(phase2)
    ocp.addPhase(phase3)
    ocp.addPhase(phase4)
    
    ocp.setAutoScaling(True,True)
    ocp.setAdaptiveMesh(True)  
    
    
    for phase in ocp.Phases:
        phase.AutoScaling=True
        phase.setUnits(R=Lstar,V=Vstar,t=Tstar,m=Mstar)
        phase.setMeshTol(1.0e-6)
        phase.setMeshErrorCriteria('max')
        phase.setMeshErrorEstimator('integrator')  


    ## Each Phase does not have to have the same AutoScale units even if its the same ODE
    phase4.setUnits(R=2*Lstar,V=Vstar,t=.8*Tstar,m=Mstar)

    ## Everything but mass
    ocp.addForwardLinkEqualCon(phase1,phase4,["R","V","t","U"])


    ocp.optimizer.set_OptLSMode("L1")
    ocp.optimizer.set_SoeLSMode("L1")
    ocp.optimizer.set_MaxLSIters(2)
    ocp.optimizer.set_PrintLevel(0)
    ocp.solve_optimize()

    
    Phase1Traj = phase1.returnTraj()  # or ocp.Phase(i).returnTraj()
    Phase2Traj = phase2.returnTraj()
    Phase3Traj = phase3.returnTraj()
    Phase4Traj = phase4.returnTraj()
    
    
    ## retrieve vars from vectors or trajs by name using ode.get_vars
    mf = ode4.get_vars("mass", Phase4Traj[-1],retscalar=True)
    print("Final Mass = ",mf,' kg')

    

    Plot(Phase1Traj,Phase2Traj,Phase3Traj,Phase4Traj)
    
    
    
    TotalTraj = Phase1Traj + Phase2Traj + Phase3Traj + Phase4Traj
    
    ## ode.get_vars alse works on full trajectories
    ts,rx,ry,rz,vx,vy,vz = ode4.get_vars(["t","R","V"],TotalTraj).T
    
    plt.plot(ts,rx,label="rx")
    plt.plot(ts,ry,label="ry")
    plt.plot(ts,rz,label="rz")
    plt.xlabel("t")
    plt.ylabel("R (m)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.plot(ts,vx,label="vx")
    plt.plot(ts,vy,label="vy")
    plt.plot(ts,vz,label="vz")
    plt.xlabel("t")
    plt.ylabel("V (m/s)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    #########################################



