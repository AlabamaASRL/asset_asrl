import numpy as np
import asset_asrl as ast
import asset_asrl.VectorFunctions as vf
import asset_asrl.OptimalControl as oc
from asset_asrl.VectorFunctions import Arguments as Args

import matplotlib.pyplot as plt
from asset_asrl.OptimalControl.MeshErrorPlots import PhaseMeshErrorPlot
import time
from mpl_toolkits.basemap import Basemap
import seaborn as sns    # pip install seaborn if you dont have it
from matplotlib import ticker




'''
Low-Thrust Orbit Transfer taken from example 6 on page 265 of
Betts, J.T. Practical methods for Optimal Control and Estimation Using Nonlinear Programming, Cambridge University Press, 2009

Optimizes a low thrust tranfer from LEO to MEO with MEE dynamics and Zonal Harmonics
'''

g0 = 32.174 
W  = 1  #lb
mu_e = 1.407645794e16


Re     = 20925662.73             
mdot   = (4.446618e-3/(450*g0))  
mu     = 1.407645794e16                 
Thrust = 4.446618e-3             
Isp = 450                        
gs = g0
J2 = 1082.639e-6
J3 = -2.565e-6
J4 = -1.608e-6

pt0 = 21837080.052835
ptf = 40007346.015232
ht0 = -0.25396764647494

Lstar = 20925662.73     ## feet
Tstar = Lstar/np.sqrt(mu/Lstar)
Mstar = W/g0         ## slugs
Fstar   = Mstar*Lstar/(Tstar**2)




#######################Helper functions for plotting ,ignore #################
def normalize(x): return np.copy(x)/np.linalg.norm(x)


def MEEToCart(Traj):
    Itraj=[]
    f=MEECartFunc(mu)
    for T in Traj:
        Tmp = np.copy(T)
        Xcart = f.compute(np.copy(T[0:6]))
        Ucart = RTNtoCart(np.copy(Xcart[0:3]),np.copy(Xcart[3:6]), np.copy(T[7:10]))
        Tmp[0:6]=Xcart
        Itraj.append(Tmp)
    return Itraj

def RTNtoCart(r, v, u):
    r = r/np.linalg.norm(r)
    n = np.cross(r, v) #/ np.linalg.norm(np.cross(r, v))
    n = normalize(n)
    t = np.cross(n, r)

    Rot = np.zeros((3, 3))
    Rot[:, 0] = r
    Rot[:, 1] = normalize(t)
    Rot[:, 2] =n
    
    return np.dot(Rot, u)

def Zlevels(Traj,zaxis=2):
    
    GP = []
    GN = []
    ns = [0]
    for i in range(2,len(Traj)):
        Tl = Traj[i-1]
        T  = Traj[i]
        switch = T[zaxis]/Tl[zaxis]
        
        if(switch<0):
            ns.append(i)
    ns.append(len(Traj)+1)
    for i in range(0,len(ns)-1):
        Tmp = Traj[ns[i]:ns[i+1]+1]
        
        if(Traj[ns[i]+1][zaxis]>0):
            GP.append(Tmp)
        else:
            GN.append(Tmp)
    return GP,GN

def Plot(Traj):
    
    color = 'k'
    
    rs = 6370997.0
    fig = plt.figure()
    
    axu = plt.subplot(211)
    ax0 = plt.subplot(234)
    ax1 = plt.subplot(235)
    ax2 = plt.subplot(236)

    axs = [ax0,ax1,ax2]
    
    ms = []
    m1 = Basemap(projection='ortho',lon_0=0,lat_0=90,resolution='l',anchor='SW',ax=axs[0],suppress_ticks=False)
    m2 = Basemap(projection='ortho',lon_0=0,lat_0=0,resolution='l',anchor='SW',ax=axs[0],suppress_ticks=False)
    m3 = Basemap(projection='ortho',lon_0=90,lat_0=0,resolution='l',anchor='SW',ax=axs[0],suppress_ticks=False)

    ms =[m1,m2,m3]

    @ticker.FuncFormatter
    def major_formatter(x, pos):
         t = (x - rs)/rs
         return f'{t:.1f}'
     
    views1 = [2,1,0]
    views2 = [[0,1],[0,2],[1,2]]
    
    for i in range(0,3):
        
        ZPs,ZNs = Zlevels(np.copy(Traj),zaxis=views1[i])
        
        if(views1[i]==1):
            tmp = ZPs
            ZPs = ZNs
            ZNs = tmp
        
        for ZP in ZPs:
            T = np.array(ZP).T*.3048
            axs[i].plot(T[views2[i][0]]+rs,T[views2[i][1]]+rs,zorder=10,color=color)
            
        for ZP in ZNs:
            T = np.array(ZP).T*.3048
            axs[i].plot(T[views2[i][0]]+rs,T[views2[i][1]]+rs,zorder=2,color=color)
            
        axs[i].xaxis.set_major_formatter(major_formatter)
        axs[i].yaxis.set_major_formatter(major_formatter)

        axs[i].xaxis.set_major_locator(ticker.MultipleLocator(rs))
        axs[i].yaxis.set_major_locator(ticker.MultipleLocator(rs))

        axs[i].grid(True)
        
        m = ms[i]
        m.drawcoastlines(ax=axs[i],linewidth=.25,zorder=5)
        m.fillcontinents(color='coral',lake_color='aqua',ax=axs[i],zorder=5)
        m.drawparallels(np.arange(-90.,120.,30.),ax=axs[i],zorder=5)
        m.drawmeridians(np.arange(0.,420.,60.),ax=axs[i],zorder=5)
        m.drawmapboundary(fill_color='aqua',ax=axs[i],zorder=2)


    axs[0].set_xlabel(r"$X (R_e)$")
    axs[0].set_ylabel(r"$Y (R_e)$")

    axs[1].set_xlabel(r"$X (R_e)$")
    axs[1].set_ylabel(r"$Z (R_e)$")
    
    axs[2].set_xlabel(r"$Y (R_e)$")
    axs[2].set_ylabel(r"$Z (R_e)$")
    
    
    axs[1].set_ylim([-2*rs,5*rs])
    axs[2].set_ylim([-2*rs,5*rs])
    
    
    T = np.array(Traj).T
    
    axu.plot(T[7]/3600,T[8],label=r'$u_r$')
    axu.plot(T[7]/3600,T[9],label=r'$u_t$')
    axu.plot(T[7]/3600,T[10],label=r'$u_n$')
    axu.grid(True)
    
    axu.set_ylabel(r"$\mathbf{u}$")
    
    axu.set_xlabel(r"$t (hrs)$")
    
    axu.legend()

    for i in range(0,3):
        axs[i].axis("Equal")
        axs[i].spines[['left', 'right', 'top']].set_visible(True)
        axs[i].set_frame_on(True)
        
        
    fig.set_size_inches(12.0, 9, forward=True)
    fig.tight_layout()

    plt.show()
    

##################### Vector Functions to define problem #####################

def RTNBasisFunc():
    # Computes RTN Basis vectors from cartesian position and velocity
    R,V = Args(6).tolist([(0,3),(3,3)])

    Rhat = R.normalized()
    Nhat = R.cross(V).normalized()
    That = Nhat.cross(R).normalized()

    return vf.stack(Rhat,That,Nhat)



def MEECartFunc(mu):
    # Function for computing cartesian position and velocity from MEEs
    X = Args(6)
    p,f,g,h,k,L = X.tolist()
    
    
    sinL = vf.sin(L)
    cosL = vf.cos(L)
    sqp  = vf.sqrt(mu/p)
    
    w = 1+f*cosL +g*sinL
    s2 = 1+h**2 +k**2
    a2 = h**2 - k**2
    r = p/w
    r_s2 = r/s2
    subs2 = 1.0/s2
    
    R = r_s2*vf.stack([cosL + a2*cosL + 2.*h*k*sinL, sinL - a2*sinL + 2.*h*k*cosL,
                     2.0*(h*sinL - k*cosL)])
    
    V = -subs2*sqp*vf.stack([sinL + a2*sinL - 2.*h*k*cosL + g - 2.*f*h*k + a2*g,
                     -cosL + a2*cosL + 2.*h*k*sinL - f + 2.*g*h*k + a2*f,
                     -2.0*(h*cosL + k*sinL + f*h + g*k)])
    
    return vf.stack([R, V])

def RadFunc(mu):
    # Function for computing radius from central body from MEEs
    X = Args(6)
    p,f,g,h,k,L = X.tolist()

    sinL = vf.sin(L)
    cosL = vf.cos(L)
    w  = 1.+f*cosL +g*sinL
    r = p/w
    return r

def ZonalGrav(mu,Re,J2,J3,J4):
    ## Function for computing Zonal Gravity in the RTN frame given the Cartesian
    ## position and velocity
    X = Args(6)
    
    R,V = X.tolist([(0,3),(3,3)])
    
    r = R.norm()
    
    
    #### Eq. 6.46-6.49 in reference ###############
    Ir = R.normalized()
    North = np.array([0,0,1.0])
    In = (North - Ir*(Ir.dot(North))).normalized()
    
    
    sphi = R[2]/r
    sphi = Ir[2]
    cphi = vf.sqrt(1 - sphi**2)
    
    
    P2 = 0.5*(3.0*(sphi**2)-1.0)
    P3 = 0.5*(5.0*(sphi**3)- 3*sphi)
    P4 = (35/8)*(sphi**4) - (30/8)*(sphi**2) + 3/8
    
    D2 = 3*sphi
    D3 = 0.5*(15.0*(sphi**2)-3.0)
    D4 = (35/2)*(sphi**3) - (30/4)*(sphi) 
    
    Js = [J2,J3,J4]
    Ps = [P2,P3,P4]
    Ds = [D2,D3,D4]
    
    grs = []
    gns = []
    
    for k in range(2,5):
        gns.append( Ds[k-2]*Js[k-2]*((Re/r)**k) )
        grs.append( ((k+1)*Ps[k-2]*Js[k-2])*((Re/r)**k)   )
        
        
    gn = vf.sum(gns)*cphi
    gr = vf.sum(grs)
    
    Gcart = (gn*In - gr*Ir)*(-mu/R.squared_norm())
    #########################
    
    ## Transform to RTN frame
    RTNBasis = RTNBasisFunc()
    M = vf.RowMatrix(RTNBasis,3,3)
    return M*Gcart
    
    
   
def MEEDynamics(mu):
    # MEE equations of motion with perturbing accellerations in RTN frame
    X = Args(9)
    
    p,f,g,h,k,L,ur,ut,un = X.tolist()
    
    sinL = vf.sin(L)
    cosL = vf.cos(L)
    sqp  = vf.sqrt(p)/np.sqrt(mu)
    
    hk = X.segment2(3)
    
    w  = 1.+f*cosL +g*sinL
    s2 = 1.+hk.squared_norm()
    
    pdot  = 2.*(p/w)*ut
    fdot  = vf.sum([ur*sinL ,((w+1)*cosL +f)*(ut/w), -(h*sinL-k*cosL)*(g*un/w)])
    gdot  = vf.sum([-ur*cosL ,((w+1)*sinL +g)*(ut/w) ,(h*sinL-k*cosL)*(f*un/w)])
    hkdot = vf.stack([cosL,sinL])*((s2*un/w)/2.0)
    Ldot  = mu*(w/p)*(w/p) + (1.0/w)*(h*sinL -k*cosL)*un
    
    return vf.stack([pdot,fdot,gdot,hkdot,Ldot])*sqp

def MEEProgradeUlaw():
    MEEs = Args(6).vf()
    RV = MEECartFunc(mu)(MEEs)
    RTNBasis = RTNBasisFunc()(RV)
    U = vf.RowMatrix(RTNBasis,3,3)*RV.tail(3).normalized() 
    return U

def MEEDynamics2(mu):
    
    ## More efficient way to write the same thing that eliminates expensive
    ## sub-expressions
    X = Args(9)
    
    p,f,g,h,k,L,ur,ut,un = X.tolist()
    
    sinL = vf.sin(L)
    cosL = vf.cos(L)
    sqp  = vf.sqrt(p)/np.sqrt(mu)
    hk = X.segment2(3)
    w  = 1.+f*cosL +g*sinL
    
    Xtmp = vf.stack(X,sinL,cosL,w)
    
    ###################################################
    ## Sin cos and w are now addditonal args
    X2 = Args(12)
    p,f,g,h,k,L,ur,ut,un,sinL,cosL,w = X2.tolist()
    hk = X2.segment2(3)

    sqp  = vf.sqrt(p)/np.sqrt(mu)
    s2 = 1.+hk.squared_norm()

    pdot  = 2.*(p/w)*ut
    fdot  = vf.sum([ur*sinL ,((w+1)*cosL +f)*(ut/w), -(h*sinL-k*cosL)*(g*un/w)])
    gdot  = vf.sum([-ur*cosL ,((w+1)*sinL +g)*(ut/w) ,(h*sinL-k*cosL)*(f*un/w)])
    hkdot = vf.stack([cosL,sinL])*((s2*un/w)/2.0)
    Ldot  = mu*(w/p)*(w/p) + (1.0/w)*(h*sinL -k*cosL)*un
    
    ## Compose back with Xtmp
    return (vf.stack([pdot,fdot,gdot,hkdot,Ldot])*sqp)(Xtmp)
     
    

class LTModel(oc.ODEBase):
     def __init__(self, mu,T,gs,Isp,Re,J2= False):
        
        ############################################################
        XtUP = oc.ODEArguments(7,3,1)
        
        ## MEEs and vehicle weight
        MEEs =  XtUP.XVec().head(6)
        ww = XtUP.XVar(6)
        
        p,f,g,h,k,L = MEEs.tolist()  # Assumed order
        
        ## Control Direction in RTN
        U = XtUP.UVec().head3().normalized()
        ## Static throttle parameter
        tau = XtUP.PVar(0)
        wwdot = -T*(1+.01*tau)/(Isp)
        acc_T =  gs*T*(1+.01*tau)*U/ww
        
        ## Convert MEEs to Cartesian then forward the ZonalGrav
        acc_J2 =  ZonalGrav(mu, Re, J2, J3, J4)( MEECartFunc(mu) )(MEEs)
        
        # Combine perturbing accellerations and call MEE dynamics
        acc = acc_T + acc_J2
        Xdot   = MEEDynamics2(mu).eval(vf.stack(MEEs,acc))
        
        ode = vf.stack([Xdot,wwdot])
        
        Vgroups = {}
        Vgroups["p"]=p
        Vgroups["f"]=f
        Vgroups["g"]=g
        Vgroups["h"]=h
        Vgroups["k"]=k
        Vgroups["L"]=L
        Vgroups["MEEs"]=MEEs
        Vgroups["U"]=XtUP.UVec()
        Vgroups["tau"]=tau
        Vgroups["t"]=XtUP.TVar()
        Vgroups["w"]=ww


        
        #############################################################
        super().__init__(ode, 7,3,1,Vgroups=Vgroups)



def EqBCon():
    ## Boundary constraint eq 6.52-6.55 from reference
    X = Args(6)
    p,f,g,h,k,L = X.tolist()
    
    eq1 = p - ptf
    eq2 = X.segment2(1).squared_norm()-.73550320568829**2
    eq3 = X.segment2(3).squared_norm()-.61761258786099**2
    eq4 = f*h + g*k
    return vf.stack([eq1,eq2,eq3,eq4])

def IqBCon():
    ## Boundary constraint eq 6.56 from reference
    X = Args(6)
    p,f,g,h,k,L = X.tolist()
    return g*h - k*f


#############################################################################

if __name__ == "__main__":
    
    
    
    
    ode = LTModel(mu,Thrust,gs,Isp,Re,J2)
    
    
    X0 = ode.make_input(p = pt0,
                        h=ht0,
                        L=np.pi,
                        w=1.0,
                        U=[0.0,1.0,0.0],
                        tau = -25.0)
    
    tfig = 90000
    integ = ode.integrator(1,MEEProgradeUlaw(),"MEEs")
    ## Same as above
    #integ = ode.integrator(1,MEEProgradeUlaw(),["p","f","g","h","k","L"])
    #integ = ode.integrator(1,MEEProgradeUlaw(),range(0,6))


    integ.setAbsTol(1.0e-12)
    integ.setRelTol(1.0e-5)
    
    IG = integ.integrate_dense(X0,tfig)
    
    #############################################
    
    
    phase = ode.phase("LGL5",IG,16)
    phase.setAutoScaling(True)
    
    ## If you need to make units vector manually
    Units = np.ones((12))  # Size of ODE Input
    Units[0]=Lstar  
    Units[6]=Fstar
    Units[7]=Tstar
    phase.setUnits(Units)    
    ## Same as above
    # Units = ode.make_units(p=Lstar,w=Fstar,t=Tstar)
    # phase.setUnits(Units)    
    ## Or if you have named variables in your ODE
    # phase.setUnits(p=Lstar,w=Fstar,t=Tstar)
    
    phase.addBoundaryValue("Front",["MEEs","w","t"],X0[0:8])
    phase.addEqualCon("Path",Args(3).norm()-1,"U")
    # Dont use control splines when placing equality path constraints on controls
    phase.setControlMode("NoSpline")
    phase.addLUFuncBound("Path",RadFunc(mu),"MEEs",Re,10*Re)

    phase.addEqualCon("Back",EqBCon(),"MEEs")
    phase.addInequalCon("Back",IqBCon(),"MEEs")
    
    phase.addLUVarBound("ODEParams","tau", -50,0)
    phase.addLowerVarBound("Back","w",.05)
    phase.addValueObjective("Back",6,-1.0)
    phase.setThreads(8,8)
    phase.optimizer.PrintLevel = 2
    phase.optimizer.set_EContol(1.0e-9)
    
    
    ## All error estimates and tolerances are in reference to the scaled ODE system
    phase.setAdaptiveMesh(True)
    phase.setMeshErrorEstimator("integrator")
    phase.setMeshTol(1.0e-7)
    
    phase.optimize_solve()

    Traj = phase.returnTraj()
    
    FinalWeight = Traj[-1][6]
    FinalTime   = Traj[-1][7]
    ThrottleParam = Traj[-1][-1]
    
    print(f"Final Weight:{FinalWeight} lb")
    print(f"Final Time:{FinalTime} s")
    print(f"Throttle Parameter:{ThrottleParam} ")

    PhaseMeshErrorPlot(phase,show=False)
    Plot(MEEToCart(Traj))
    
    
    

