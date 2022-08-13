from __future__ import print_function

import numpy as np
import asset as ast
from QuatPlot import AnimSlew,PlotSlew,CompSlew,CompSlew2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from DerivChecker import FDDerivChecker
import time


import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from itertools import product, combinations

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from decimal import Decimal
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.tri as tri




norm      = np.linalg.norm
vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
solvs     = ast.Solvers
JetJobModes=solvs.JetJobModes
Jet = solvs.Jet
#ast.PyMain()

def normalize(x):return np.array(x)/np.linalg.norm(x)
def octant_points(samples=1):

    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    Nsamps = samples*8
    for i in range(Nsamps):
        y = 1 - (i / float(Nsamps - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        p=np.array([x, y, z])
        if(x>=0 and y>=0 and z>=0):points.append(p)
        
        
        
    ps = points[1::]
    eps = 1.0e-2
    ps.append(normalize(np.array([1,eps,eps])))
    ps.append(normalize(np.array([eps,1,eps])))
    ps.append(normalize(np.array([eps,eps,1])))
    
    return ps

def octant_points2(samples=1,edges = 10):
    ps = octant_points(samples)
    eps = 1.0e-3

    kk = np.linspace(eps,1-eps,edges)
    
    for k in kk:
        YZ = normalize([eps,1-k,k])
        XY = normalize([1-k,k,eps/1.1])
        XZ = normalize([1-k,eps/1.2,k])
        ps.append(YZ)
        ps.append(XY)
        ps.append(XZ)
    return ps

def OctSubPLot(fig1, ax1,points,zs,Ivec,EqualColor=True,cmap='PuOr',Legend=False,badpoints=[]):

    
    R1 = R.from_euler('Z',45,degrees=True)
    k = np.cross([1,1,1],[0,0,1])
    khat = k/np.linalg.norm(k)
    theta = np.deg2rad(44)
    R2 = R.from_rotvec(khat*theta)
    
    Rf = R2*R1
    spoints =[]
    for p in points:
        spoints.append(Rf.inv().apply(p))
        
    cpoints=[]
    for p in badpoints:
        cpoints.append(Rf.inv().apply(p))
        
    xyz1 = [ [1,0,0], [0,1,0] , [0,0,1]]    
    xyz=[]
    for p in xyz1:xyz.append(Rf.inv().apply(p))
        
        
    pp = np.array(spoints).T
    x = pp[1]
    y=  pp[2]
    
    triang = tri.Triangulation(x, y)
    #mask = tri.TriAnalyzer(triang).get_flat_tri_mask(.01)
    #triang.set_mask(mask)
    
    vmax = max(zs)
    vmin = min(zs)
    
    if(EqualColor==True):
        if(vmax>abs(vmin)):
            vmin=-vmax
        else:
            vmax=-vmin
     
    ax1.set_aspect('equal')
    tpc = ax1.tripcolor(triang, zs, shading='gouraud',cmap=cmap,vmin=vmin,vmax=vmax,alpha=.9)
    #tpc.set_clim(min(zs), max(zs))
    

    cbar=fig1.colorbar(tpc,ax=ax1,pad = 0.007,boundaries=np.linspace(min(zs),max(zs),60))
    
    '''
    if(max(zs)>0 and False):
        cbar.set_ticks([.95*min(zs),0,.95*max(zs)])
        cbar.ax.set_yticklabels(['{0:.1f}'.format(.95*min(zs))+"%", '0.0%', '{0:.1f}'.format(.95*max(zs))+"%"])
    else:
        cbar.set_ticks([.9*min(zs),np.mean(zs),.1*min(zs)])
        cbar.ax.set_yticklabels(['{0:.1f}'.format(.9*min(zs))+"%", '{0:.1f}'.format(np.median(zs))+"%", '{0:.1f}'.format(.1*min(zs))+"%"])
    '''
    #cbar.ax.locator_params(nbins=3)
    #cbar.set_label(r'$\eta$', labelpad=-3)


    zz = np.linspace(0,1,len(zs))
    #ax1.scatter(pp[1],pp[2],c=zz,cmap='viridis',zorder=10,s=.5)


    ax1.scatter(xyz[0][1],xyz[0][2],color='red',label=r'$\hat{X}$',zorder=10,edgecolor='k')
    ax1.scatter(xyz[1][1],xyz[1][2],color='green',label=r'$\hat{Y}$',zorder=10,edgecolor='k')
    ax1.scatter(xyz[2][1],xyz[2][2],color='blue',label=r'$\hat{Z}$',zorder=10,edgecolor='k')
    
    
    
    
    
    
    Xl = [[0,0],xyz[0][1:3]]
    Xl = np.array(Xl).T
    ax1.plot(Xl[0],Xl[1],color='r',alpha=.2)
    
    Xl = [[0,0],xyz[1][1:3]]
    Xl = np.array(Xl).T
    ax1.plot(Xl[0],Xl[1],color='g',alpha=.2)
    
    Xl = [[0,0],xyz[2][1:3]]
    Xl = np.array(Xl).T
    ax1.plot(Xl[0],Xl[1],color='b',alpha=.2)
    
    ax1.scatter(0,0,color='k',zorder=2,alpha=.6,s=3)

    
    limit = np.sqrt(2)/2.0
    xi = np.linspace(-limit, limit, 300)
    yi = np.linspace(-limit, limit, 300)
    interpolator = tri.LinearTriInterpolator(triang, zs)
    #interpolator = tri.CubicTriInterpolator(triang, zs,kind='min_E')

    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    
    
    sp = np.array(spoints).T
    ax1.scatter(sp[1],sp[2],color='k',marker = '.',s=.5,label=r'$\hat{n}_i$')
    if(Legend==True): ax1.legend()
    
    
    #######################
    angs=np.linspace(0.0,90,30)
    XLat = normalize(np.array([1,0,0]))
    XLats=[]
    
    YLat = normalize(np.array([0,1,0]))
    YLats=[]
    k1= normalize(np.array([1,0,0]))

    nn=1.5
    ZLon = normalize(np.array([1,0,0]))
    ZLons=[]
    k2= normalize(np.array([0,-1,0]))


    for a in angs:
        Rz = R.from_euler('Z',a,degrees=True)
        Rk = R.from_rotvec(k2*np.deg2rad(a))
        Rx = R.from_rotvec(k1*np.deg2rad(a))
        

        tmp1=Rz.apply(XLat)
        tmp2=Rx.apply(YLat)
        tmp3=Rk.apply(ZLon)
        XLats.append(Rf.inv().apply(tmp1))
        YLats.append(Rf.inv().apply(tmp2))
        ZLons.append(Rf.inv().apply(tmp3))
    
    XLats = np.array(XLats).T
    YLats = np.array(YLats).T
    ZLons = np.array(ZLons).T
    ax1.plot(XLats[1],XLats[2],color='k',linestyle='--',alpha=.7, linewidth=0.6)
    ax1.plot(YLats[1],YLats[2],color='k',linestyle='--',alpha=.7, linewidth=0.6)
    ax1.plot(ZLons[1],ZLons[2],color='k',linestyle='--',alpha=.7, linewidth=0.6)

    #########################
    if(len(cpoints)>0):
        cp = np.array(cpoints).T
        #ax1.scatter(cp[1],cp[2],color='r',marker = '.',s=4)
    
    
    ###
    #np.ma.masked_array(zi,mask=)
    if(max(zs)>0):
        ax1.contour(xi, yi, zi, levels=[0], linewidths=0.5, colors='k')
    else:
        f=2
        ax1.contour(xi, yi, zi, levels=[np.median(zs)], linewidths=0.5, colors='white',linestyles='solid')
        
    ax1.set_xlim([-.765,.765])
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])



def GetEigTraj(TrajF,Ivec,nvec):
    C1 = Ivec*nvec
    C2 = np.cross(nvec,C1)
    
    TrajR =[]
    for T in TrajF:
        X = np.zeros((11))
        X[0:3] = nvec*np.sin(T[0]/2)
        X[3] = np.cos(T[0]/2)
        X[4:7] = nvec*T[1]
        X[7]=T[2]
        X[8:11] = T[3]*C1 + T[1]*T[1]*C2
        TrajR.append(X)
        
    return TrajR  


class QuatModel(oc.ode_x_u.ode):
    def __init__(self,Ivec):
        Xvars = 7
        Uvars = 3
        ##################################################
        args = Args(Xvars + Uvars + 1)
        
        q    = args.head(4).normalized()
        w    = args.segment3(4)
        T    = args.tail3()
        
        qdot  = vf.quatProduct(q,w.padded_lower(1))/2.0
        L     = w.cwiseProduct(Ivec)
        wdot  = (L.cross(w) + T).cwiseQuotient(Ivec)
        ode   = vf.stack(qdot,wdot)
        
        ###################################################
        super().__init__(ode,Xvars,Uvars)
        

class EigModel(oc.ode_x_u.ode):
    def __init__(self):
        args = oc.ODEArguments(2,1)
        ode = vf.stack([args[1],args[3]])
        super().__init__(ode,2,1)



def EigAxisPhase(Ivec,Tmax,nvec,theta,Nsegs):
    
    alpha=(abs(theta)/theta)*Tmax/norm(nvec*Ivec)
    h = np.sqrt(theta/alpha)
    ts =  np.linspace(0,2*h,50)
    IG = []
    for t in ts:
        u =alpha
        tdot = alpha*t
        thet = alpha*t*t/2
        if(t>h):
            tt=t-h
            u =-alpha
            tdot = alpha*(h-tt)
            thet = alpha*h*h/2 - alpha*tt*tt/2 + alpha*h*tt
        IG.append([thet,tdot,t,u*.7])
        
    ode = EigModel()
    phase = ode.phase(Tmodes.LGL3,IG,Nsegs)
    phase.setControlMode(oc.ControlModes.BlockConstant)
    phase.addBoundaryValue(PhaseRegs.Front,[0,1,2],[0,0,0])
    
    C1 = Ivec*nvec
    C2 = np.cross(nvec,C1)
    
    def SBoundFunc(c1,c2):
        args = Args(2)
        tdot = args[0]
        u = args[1]
        return u*c1 + (tdot**2)*c2
    
    for i in range(0,3):
        F = SBoundFunc(C1[i],C2[i])
        phase.addUpperFuncBound(PhaseRegs.Path,F,[1,3], Tmax,1.0)
        phase.addLowerFuncBound(PhaseRegs.Path,F,[1,3], -Tmax,1.0)
        
       
    phase.addBoundaryValue(PhaseRegs.Back ,[0,1],[theta,0])

    phase.addDeltaTimeObjective(1.0)
    phase.optimizer.OptLSMode = solvs.LineSearchModes.L1
    phase.optimizer.MaxLSIters = 1
    phase.optimizer.QPOrderingMode = solvs.QPOrderingModes.MINDEG
    
    phase.JetJobMode = JetJobModes.Optimize
    
    return phase

def ExtractIG(EigAxPhase,Ivec,nvec):
    IG = GetEigTraj(EigAxPhase.returnTraj(),Ivec,nvec)
    for I in IG:
        I[7]*=.8
        I[8:11]*=.6
    return IG

   
    
def TrueOptPhase(Ivec,Tmax,nvec,theta, EigAxPhase,Nsegs):
    
    IG = ExtractIG(EigAxPhase,Ivec,nvec)
    ode = QuatModel(Ivec)
    phase= ode.phase(Tmodes.LGL3,IG,Nsegs)
    
    phase.setControlMode(oc.ControlModes.BlockConstant)
    phase.addBoundaryValue(PhaseRegs.Front,range(0,8),[0,0,0,1,0,0,0,0])
    phase.addBoundaryValue(PhaseRegs.Back,range(4,7),[0,0,0])
    phase.addLUVarBounds(PhaseRegs.Path,[8,9,10],-Tmax,Tmax,0.01)
    
    def QuatToAxAngCon(q = Args(4)):
        n = q.head3().normalized()
        thetan = 2.0*vf.arctan(q.head3().norm()/q[3])
        return thetan*n - nvec*theta
    
    phase.addEqualCon(PhaseRegs.Back,QuatToAxAngCon(),range(0,4))
    phase.addDeltaTimeObjective(1.0)
    
    phase.JetJobMode = JetJobModes.Optimize
    return phase
    
    
    phase.optimizer.OptLSMode = solvs.LineSearchModes.L1
    phase.optimizer.MaxLSIters =1
    phase.optimizer.MaxAccIters =100
    phase.optimizer.QPOrderingMode = solvs.QPOrderingModes.MINDEG
    phase.optimizer.BoundFraction=.997
    phase.optimizer.deltaH=1.0e-6
    phase.optimizer.KKTtol=1.0e-6
    phase.JetJobMode = JetJobModes.Optimize
    return phase



def CalcManeuvers(Ivec,thetadeg=120,Tmax=1,n=3000,nsegs=250,ccount=8):
    nvecs = octant_points(n)
    nvecs.append(normalize([1,1,1]))
    theta = np.deg2rad(thetadeg)
    
    EAArgs=[(Ivec,Tmax,nvec,theta,nsegs) for nvec in nvecs]
    
    EigAxisRes = Jet.map(EigAxisPhase,EAArgs,ccount)
    
    TOPTArgs=[(Ivec,Tmax,nvec,theta,EigAx,nsegs) 
                    for nvec,EigAx in zip(nvecs,EigAxisRes)]
    
    TrueOptRes = Jet.map(TrueOptPhase,TOPTArgs,ccount)
    
    
        
    t0 = time.perf_counter()
    
    #ccount =  ast.Utils.get_core_count()+2

    
    
    
    tf = time.perf_counter()
    print(tf-t0)
    
    zs = []
    
    for i in range(0,len(TrueOptRes)):
        
        TrajEig   = EigAxisRes[i].returnTraj()
        tfeig     = TrajEig[-1][2]
        
        TrajFull  = TrueOptRes[i].returnTraj()
        tffull    = TrajFull[-1][7]
        
        zs.append(100*(tffull-tfeig)/tfeig)
        #zs.append(tffull)
        
    ni = nvecs[-1]
    TrajFull  = TrueOptRes[-1].returnTraj()[1:-1]
    TrajEig  = GetEigTraj(EigAxisRes[-1].returnTraj(),Ivec,ni)[1:-1]
    
    AnimSlew(TrajFull)
    AnimSlew(TrajEig)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    OctSubPLot(fig, ax1,nvecs,zs,Ivec,cmap='plasma',EqualColor=False,Legend = True)
        
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    
    T = [TrajFull,TrajEig,TrajFull]
    CompSlew2(ax2,T,nvec=ni)
    
    plt.show()

    
        
    
    
    
    #for Res in FullAxisRes:
    #    AnimSlew(Res.returnTraj(),Anim=False,Ivec=Ivec)
    
if __name__ == "__main__":
    #
    Ivec = np.array([1,2.0,2.6])    ## Inertia of a 6U cubesat
    Ivec = np.array([1,3.13,3.92])    ## Inertia of a 6U cubesat

    input("")
    #CalcManeuvers(Ivec,45.0,n=500,nsegs=128,ccount=1)
    #CalcManeuvers(Ivec,45.0,n=500,nsegs=128,ccount=8)
    CalcManeuvers(Ivec,120.0,n=30,nsegs=250,ccount=8)



    
