from __future__ import print_function

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
        
        if(x>=0 and y>=0 and z>=0):points.append(np.array([x, y, z]))
        
        
        
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



def OctPlot1(points,zs,Ivec,EqualColor=True,cmap='PuOr'):

    
    
    R1 = R.from_euler('Z',45,degrees=True)
    k = np.cross([1,1,1],[0,0,1])
    khat = k/np.linalg.norm(k)
    theta = np.deg2rad(45)
    R2 = R.from_rotvec(khat*theta)
    
    Rf = R2*R1
    spoints =[]
    for p in points:
        spoints.append(Rf.inv().apply(p))
        
    
    xyz1 = [ [1,0,0], [0,1,0] , [0,0,1]]    
    xyz=[]
    
    for p in xyz1:
        xyz.append(Rf.inv().apply(p))
        
        
    pp = np.array(spoints).T
    x  = pp[1]
    y  = pp[2]
    
    triang = tri.Triangulation(x, y)
    
    vmin = max(zs)
    vmax = min(zs)
    
    if(EqualColor==True ):
        if(vmax>abs(vmin)):
            vmin=-vmax
        else:
            vmax=-vmin
        
    
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    tpc = ax1.tripcolor(triang, zs, shading='gouraud',cmap=cmap,vmin=vmin,vmax=vmax)
    #tpc.set_clim(min(zs), max(zs))
    

    fig1.colorbar(tpc,)
    
    
    Ihat = normalize((Ivec))
    Ihat = Rf.inv().apply(Ihat)
    
    plt.scatter(Ihat[1],Ihat[2],color='gold',label='X',marker='*',zorder=10)

    plt.scatter(xyz[0][1],xyz[0][2],color='red',label='X')
    plt.scatter(xyz[1][1],xyz[1][2],color='green',label='Y')
    plt.scatter(xyz[2][1],xyz[2][2],color='blue',label='Z')
    
    Xl = [[0,0],xyz[0][1:3]]
    Xl = np.array(Xl).T
    plt.plot(Xl[0],Xl[1],color='r',alpha=.1)
    
    Xl = [[0,0],xyz[1][1:3]]
    Xl = np.array(Xl).T
    plt.plot(Xl[0],Xl[1],color='g',alpha=.1)
    
    Xl = [[0,0],xyz[2][1:3]]
    Xl = np.array(Xl).T
    plt.plot(Xl[0],Xl[1],color='b',alpha=.1)
    
    
    
    limit = np.sqrt(2)/2
    xi = np.linspace(-limit, limit, 700)
    yi = np.linspace(-limit, limit, 700)
    interpolator = tri.LinearTriInterpolator(triang, zs)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    
    
    #######################
    angs=np.linspace(0.0,90,30)
    XLat = normalize(np.array([1,0,Ivec[0]/Ivec[2]]))
    XLats=[]
    
    YLat = normalize(np.array([0,1,Ivec[1]/Ivec[2]]))
    YLats=[]
    
    nn=1.5
    ZLon = normalize(np.array([Ivec[1]**nn,Ivec[0]**nn,0]))
    ZLons=[]
    k2= normalize(np.array([Ivec[0]**nn,-Ivec[1]**nn,0]))


    for a in angs:
        Rz = R.from_euler('Z',a,degrees=True)
        Rk = R.from_rotvec(k2*np.deg2rad(a))

        tmp1=Rz.apply(XLat)
        tmp2=Rz.inv().apply(YLat)
        tmp3=Rk.apply(ZLon)
        XLats.append(Rf.inv().apply(tmp1))
        YLats.append(Rf.inv().apply(tmp2))
        ZLons.append(Rf.inv().apply(tmp3))
    
    XLats = np.array(XLats).T
    YLats = np.array(YLats).T
    ZLons = np.array(ZLons).T
    plt.plot(XLats[1],XLats[2],color='r',linestyle='--',alpha=.5, linewidth=0.6)
    plt.plot(YLats[1],YLats[2],color='k',linestyle='--',alpha=.6, linewidth=0.6)
    plt.plot(ZLons[1],ZLons[2],color='k',linestyle='--',alpha=.6, linewidth=0.6)

    #########################
        
    
    
    
    #np.ma.masked_array(zi,mask=)
    ax1.contour(xi, yi, zi, levels=[0], linewidths=0.5, colors='k')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    plt.show()
    
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
    
    if(max(zs)>0):
        cbar.set_ticks([.95*min(zs),0,.95*max(zs)])
        cbar.ax.set_yticklabels(['{0:.1f}'.format(.95*min(zs))+"%", '0.0%', '{0:.1f}'.format(.95*max(zs))+"%"])
    else:
        cbar.set_ticks([.9*min(zs),np.mean(zs),.1*min(zs)])
        cbar.ax.set_yticklabels(['{0:.1f}'.format(.9*min(zs))+"%", '{0:.1f}'.format(np.median(zs))+"%", '{0:.1f}'.format(.1*min(zs))+"%"])
        
    #cbar.ax.locator_params(nbins=3)
    #cbar.set_label(r'$\eta$', labelpad=-3)


    zz = np.linspace(0,1,len(zs))
    #ax1.scatter(pp[1],pp[2],c=zz,cmap='viridis',zorder=10,s=.5)


    ax1.scatter(xyz[0][1],xyz[0][2],color='red',label=r'$\hat{X}$',zorder=10,edgecolor='k')
    ax1.scatter(xyz[1][1],xyz[1][2],color='green',label=r'$\hat{Y}$',zorder=10,edgecolor='k')
    ax1.scatter(xyz[2][1],xyz[2][2],color='blue',label=r'$\hat{Z}$',zorder=10,edgecolor='k')
    if(Legend==True): ax1.legend(loc='center left')
    
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


