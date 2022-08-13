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


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
def get_cube():   
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta)
    y = np.sin(Phi)*np.sin(Theta)
    z = np.cos(Theta)/np.sqrt(2)
    return x,y,z

        
#sns.set()
from math import pi


    
def AnimSlew(traj,sp=6,Qsig=[],fname = 'Video',Anim=True,Elev=30,Azim =0,Ivec=[1,1,1], scaleH=True):
    BT =[]
    Ivec = np.array(Ivec)
    TT = np.array(traj).T
    xbasis = []
    ybasis = []
    zbasis = []
    hbasis = []
    for T in traj:
        Q = T[0:4]
        qv =T[0:3]/np.linalg.norm(T[0:3])
        H = (Ivec*T[4:7])/np.linalg.norm((Ivec*T[4:7]))
        #H = (T[4:7])/np.linalg.norm((T[4:7]))

        RR = R.from_quat(Q).as_matrix()
        BT.append(T[8:11])
        xbasis.append(RR.T[0])
        #xbasis.append(qv)
        ybasis.append(RR.T[1])
        #ybasis.append(np.dot(RR,H1))

        zbasis.append(RR.T[2])
        hbasis.append(np.dot(RR,H))
        #hbasis.append(H)
        
    
    xbasis = np.array(xbasis).T
    ybasis = np.array(ybasis).T
    zbasis = np.array(zbasis).T
    hbasis = np.array(hbasis).T
    BT=np.array(BT).T

    Axis = 7
    fig = plt.figure()
    
    ax = fig.add_subplot(121, projection='3d')
    ax.view_init(elev=Elev, azim=Azim)

    XT,=ax.plot(xbasis[0],xbasis[1],xbasis[2], color="r")
    YT,=ax.plot(ybasis[0],ybasis[1],ybasis[2], color="g")
    ZT,=ax.plot(zbasis[0],zbasis[1],zbasis[2], color="b")
    HT,=ax.plot(hbasis[0],hbasis[1],hbasis[2], color="k")

    XTT,=ax.plot([0,xbasis[0][-1]],[0,xbasis[1][-1]],[0,xbasis[2][-1]], color="r",marker='o',markeredgecolor='black',markerfacecolor='black')
    YTT,=ax.plot([0,ybasis[0][-1]],[0,ybasis[1][-1]],[0,ybasis[2][-1]], color="g",marker='o',markeredgecolor='black',markerfacecolor='black')
    ZTT,=ax.plot([0,zbasis[0][-1]],[0,zbasis[1][-1]],[0,zbasis[2][-1]], color="b",marker='o',markeredgecolor='black',markerfacecolor='black')
    HTT,=ax.plot([0,hbasis[0][-1]],[0,hbasis[1][-1]],[0,hbasis[2][-1]], color="k",marker='o',markeredgecolor='black',markerfacecolor='black')

    #ax.set_aspect("equal")
    ax.set_title('Horizon Track and Final Orientation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color="cyan",alpha=0.2)
    
    #x,y,z = get_cube()
    
    #SS = .2*Ivec/Ivec[2]

    #ax.plot_surface(SS[0]*x, SS[1]*y, SS[2]*z, color="black")
    
    ax1 = fig.add_subplot(322)
    ax2 = fig.add_subplot(324)
    ax3 = fig.add_subplot(326)
    
    axs=[ax1,ax2,ax3]
    fig.set_size_inches(15, 8)
    
    Qi, =axs[0].plot(TT[Axis],TT[0],label=r'$q_1$')
    Qj, =axs[0].plot(TT[Axis],TT[1],label=r'$q_2$')
    Qk, =axs[0].plot(TT[Axis],TT[2],label=r'$q_3$')
    Qw, =axs[0].plot(TT[Axis],TT[3],label=r'$q_4$')
    #Qm =axs[0].plot(TT[11],TT[0]**2 + TT[1]**2 +TT[2]**2 +TT[3]**2,label='|Q|')
    axs[0].grid(True)
    axs[0].set_ylabel(r'$\mathbf{q}$')
    axs[0].legend(loc=1)
    c = sns.color_palette()
    if(len(Qsig)>0):
        Q = np.array(Qsig).T
        axs[0].plot(Q[4],Q[0],label='RQi', linestyle='dashed',color=c[0])
        axs[0].plot(Q[4],Q[1],label='RQj', linestyle='dashed',color=c[1])
        axs[0].plot(Q[4],Q[2],label='RQk', linestyle='dashed',color=c[2])
        axs[0].plot(Q[4],Q[3],label='RQw', linestyle='dashed',color=c[3])
        
    WX,=axs[1].plot(TT[Axis],(TT[4]),label=r'$\omega_1$', color="r")
    WY,=axs[1].plot(TT[Axis],(TT[5]),label=r'$\omega_2$', color="g")
    WZ,=axs[1].plot(TT[Axis],(TT[6]),label=r'$\omega_3$', color="b")
    axs[1].grid(True)
    axs[1].set_ylabel(r'$\vec{\omega}$')
    axs[1].legend(loc=1)
    
    TX, = axs[2].plot(TT[Axis],BT[0],label=r'$T_1$', color="r")
    TY, =axs[2].plot(TT[Axis],BT[1],label=r'$T_2$', color="g")
    TZ, =axs[2].plot(TT[Axis],BT[2],label=r'$T_3$', color="b")
    axs[2].grid(True)
    axs[2].set_ylabel(r'$\vec{T}$')
    axs[2].legend(loc=1)
    
    
    
    def init():
        Qi.set_data([], [])
        Qj.set_data([], [])
        Qk.set_data([], [])
        Qw.set_data([], [])
        
        WX.set_data([], [])
        WY.set_data([], [])
        WZ.set_data([], [])
        
        XT.set_data([], [])
        YT.set_data([], [])
        ZT.set_data([], [])
        HT.set_data([], [])

        XT.set_3d_properties([], 'z')
        YT.set_3d_properties([], 'z')
        ZT.set_3d_properties([], 'z')
        HT.set_3d_properties([], 'z')
        
        XTT.set_data([], [])
        YTT.set_data([], [])
        ZTT.set_data([], [])
        HTT.set_data([], [])
        
        XTT.set_3d_properties([], 'z')
        YTT.set_3d_properties([], 'z')
        ZTT.set_3d_properties([], 'z')
        HTT.set_3d_properties([], 'z')
        
        TX.set_data([], [])
        TY.set_data([], [])
        TZ.set_data([], [])
        
        
        return Qi,Qj,Qk,Qw,WX,WY,WZ,TX,TY,TZ,XT,YT,ZT,XTT,YTT,ZTT,HT,HTT
    def animate(i):
        Qi.set_data(TT[Axis][0:i],TT[0][0:i])
        Qj.set_data(TT[Axis][0:i],TT[1][0:i])
        Qk.set_data(TT[Axis][0:i],TT[2][0:i])
        Qw.set_data(TT[Axis][0:i],TT[3][0:i])
        
        WX.set_data(TT[Axis][0:i],(TT[4][0:i]))
        WY.set_data(TT[Axis][0:i],(TT[5][0:i]))
        WZ.set_data(TT[Axis][0:i],(TT[6][0:i]))
        
        TX.set_data(TT[Axis][0:i],BT[0][0:i])
        TY.set_data(TT[Axis][0:i],BT[1][0:i])
        TZ.set_data(TT[Axis][0:i],BT[2][0:i])
        
       
        
        XT.set_data(xbasis[0][0:i],xbasis[1][0:i])
        YT.set_data(ybasis[0][0:i],ybasis[1][0:i])
        ZT.set_data(zbasis[0][0:i],zbasis[1][0:i])
        HT.set_data(hbasis[0][0:i],hbasis[1][0:i])

        XT.set_3d_properties(xbasis[2][0:i], 'z')
        YT.set_3d_properties(ybasis[2][0:i], 'z')
        ZT.set_3d_properties(zbasis[2][0:i], 'z')
        HT.set_3d_properties(hbasis[2][0:i], 'z')
        
        XTT.set_data([0,xbasis[0][i]],[0,xbasis[1][i]])
        YTT.set_data([0,ybasis[0][i]],[0,ybasis[1][i]])
        ZTT.set_data([0,zbasis[0][i]],[0,zbasis[1][i]])
        HTT.set_data([0,hbasis[0][i]],[0,hbasis[1][i]])
        
        XTT.set_3d_properties([0,xbasis[2][i]], 'z')
        YTT.set_3d_properties([0,ybasis[2][i]], 'z')
        ZTT.set_3d_properties([0,zbasis[2][i]], 'z')
        HTT.set_3d_properties([0,hbasis[2][i]], 'z')
        
        return Qi,Qj,Qk,Qw,WX,WY,WZ,TX,TY,TZ,XT,YT,ZT,XTT,YTT,ZTT,HT,HTT
    if(Anim==True):
        anim = animation.FuncAnimation(fig, animate, init_func=init,frames=len(traj), interval=100*TT[Axis][-1]/(len(traj)*sp), blit=True)
    #anim.save('basic_animation.html', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.tight_layout(True)
    #plt.savefig(fname)
    plt.show()
    

def PlotSlew(traj,sp=6,Qsig=[],fname = 'Video',Anim=True,Elev=30,Azim =0,Ivec=[1,1,1], scaleH=True):
    BT =[]
    Ivec = np.array(Ivec)
    TT = np.array(traj).T
    xbasis = []
    ybasis = []
    zbasis = []
    hbasis = []
    for T in traj:
        Q = T[0:4]
        qv =T[0:3]/np.linalg.norm(T[0:3])
        H = (Ivec*T[4:7])/np.linalg.norm((Ivec*T[4:7]))
        #H = (T[4:7])/np.linalg.norm((T[4:7]))

        RR = R.from_quat(Q).as_matrix()
        BT.append(T[8:11])
        xbasis.append(RR.T[0])
        #xbasis.append(qv)
        ybasis.append(RR.T[1])
        #ybasis.append(np.dot(RR,H1))

        zbasis.append(RR.T[2])
        hbasis.append(np.dot(RR,H))
        #hbasis.append(H)
        
    
    xbasis = np.array(xbasis).T
    ybasis = np.array(ybasis).T
    zbasis = np.array(zbasis).T
    hbasis = np.array(hbasis).T
    BT=np.array(BT).T

    Axis = 7
    fig = plt.figure()
    
    
    
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(211)
    ax3 = fig.add_subplot(212)
    
    axs=[ax1,ax2,ax3]
    #fig.set_size_inches(15, 8)
    
    
    WX,=axs[1].plot(TT[Axis],(TT[4]),label=r'$\omega_1$', color="r")
    WY,=axs[1].plot(TT[Axis],(TT[5]),label=r'$\omega_2$', color="g")
    WZ,=axs[1].plot(TT[Axis],(TT[6]),label=r'$\omega_3$', color="b")
    axs[1].grid(True)
    axs[1].set_ylabel(r'$\vec{\omega}$')
    axs[1].legend(loc=1)
    
    TX, = axs[2].plot(TT[Axis],BT[0],label=r'$T_1$', color="r")
    TY, =axs[2].plot(TT[Axis],BT[1],label=r'$T_2$', color="g")
    TZ, =axs[2].plot(TT[Axis],BT[2],label=r'$T_3$', color="b")
    axs[2].grid(True)
    axs[2].set_ylabel(r'$\vec{T}$')
    axs[2].legend(loc=1)
    plt.tight_layout(True)
    plt.show()
    
    
    
def CompSlew(trajs,sp=6,Qsig=[],fname = 'Video',Anim=True,Elev=30,Azim =0,Ivec=[1,1,1],nvec=[1,1,1], scaleH=True):
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=Elev, azim=Azim)
    
    lss = ['solid','dashed','dotted']
    
    Ivec = np.array(Ivec)
    for i,traj in enumerate(trajs):
        TT = np.array(traj).T
        xbasis = []
        ybasis = []
        zbasis = []
        hbasis = []
        for T in traj:
            Q = T[0:4]
            qv =T[0:3]/np.linalg.norm(T[0:3])
            H = (Ivec*T[4:7])/np.linalg.norm((Ivec*T[4:7]))
            #H = (T[4:7])/np.linalg.norm((T[4:7]))
    
            RR = R.from_quat(Q).as_matrix()
            xbasis.append(RR.T[0])
            #xbasis.append(qv)
            ybasis.append(RR.T[1])
            #ybasis.append(np.dot(RR,H1))
    
            zbasis.append(RR.T[2])
            hbasis.append(np.dot(RR,H))
            #hbasis.append(H)
            
        
        xbasis = np.array(xbasis).T
        ybasis = np.array(ybasis).T
        zbasis = np.array(zbasis).T
        hbasis = np.array(hbasis).T
       
    
        

        XT,=ax.plot(xbasis[0],xbasis[1],xbasis[2], color="r",linestyle = lss[i])
        YT,=ax.plot(ybasis[0],ybasis[1],ybasis[2], color="g",linestyle = lss[i])
        ZT,=ax.plot(zbasis[0],zbasis[1],zbasis[2], color="b",linestyle = lss[i])
        #HT,=ax.plot(hbasis[0],hbasis[1],hbasis[2], color="k",linestyle = lss[i])
        
        if(i==len(trajs)):
            w=0
            XTT,=ax.plot([0,xbasis[0][w]],[0,xbasis[1][w]],[0,xbasis[2][w]], color="r",marker='o',markeredgecolor='black',markerfacecolor='r',zorder=10)
            YTT,=ax.plot([0,ybasis[0][w]],[0,ybasis[1][w]],[0,ybasis[2][w]], color="g",marker='o',markeredgecolor='black',markerfacecolor='g',zorder=10)
            ZTT,=ax.plot([0,zbasis[0][w]],[0,zbasis[1][w]],[0,zbasis[2][w]], color="b",marker='o',markeredgecolor='black',markerfacecolor='b',zorder=10)
            NTT,=ax.plot([0,nvec[0]],[0,nvec[1]],[0,nvec[2]], color="k",marker='o',markeredgecolor='black',markerfacecolor='black',zorder=10)
            
            #HTT,=ax.plot([0,hbasis[0][-1]],[0,hbasis[1][-1]],[0,hbasis[2][-1]], color="k",marker='o',markeredgecolor='black',markerfacecolor='black')

    #ax.set_aspect("equal")
    lines = ax.get_lines()

    legend1 = plt.legend([lines[i] for i in [0,3]], ["OPT", "EAM"], loc=1)
    legend2 = plt.legend([lines[i] for i in [9,10,11,12]], [r'$\hat{X}$',r'$\hat{Y}$',r'$\hat{Z}$',r'$\hat{n}$'], loc=2)

    ax.add_artist(legend1)
    ax.add_artist(legend2)
    
    ax.set_xlabel(r'${X}$')
    ax.set_ylabel(r'${Y}$')
    ax.set_zlabel(r'${Z}$')
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    #ax.legend()
    ax.plot_surface(x, y, z, color="cyan",alpha=0.152)
    
    
    u, v = np.mgrid[0:np.pi/2:50j, 0:np.pi/2:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    #ax.legend()
    ax.plot_surface(x, y, z, color="k",alpha=0.13)
    
    plt.show()
    
    
def CompSlew2(ax,trajs,sp=6,Qsig=[],fname = 'Video',Anim=True,Elev=30,Azim =0,Ivec=[1,1,1],nvec=[1,1,1], scaleH=True):
    
    
    ax.view_init(elev=Elev, azim=Azim)
    
    lss = ['solid','dashed','dotted']
    
    Ivec = np.array(Ivec)
    for i,traj in enumerate(trajs):
        TT = np.array(traj).T
        xbasis = []
        ybasis = []
        zbasis = []
        hbasis = []
        for T in traj:
            Q = T[0:4]
            qv =T[0:3]/np.linalg.norm(T[0:3])
            H = (Ivec*T[4:7])/np.linalg.norm((Ivec*T[4:7]))
            #H = (T[4:7])/np.linalg.norm((T[4:7]))
    
            RR = R.from_quat(Q).as_matrix()
            xbasis.append(RR.T[0])
            #xbasis.append(qv)
            ybasis.append(RR.T[1])
            #ybasis.append(np.dot(RR,H1))
    
            zbasis.append(RR.T[2])
            hbasis.append(np.dot(RR,H))
            #hbasis.append(H)
            
        
        xbasis = np.array(xbasis).T
        ybasis = np.array(ybasis).T
        zbasis = np.array(zbasis).T
        hbasis = np.array(hbasis).T
       
    
        

        XT,=ax.plot(xbasis[0],xbasis[1],xbasis[2], color="r",linestyle = lss[i])
        YT,=ax.plot(ybasis[0],ybasis[1],ybasis[2], color="g",linestyle = lss[i])
        ZT,=ax.plot(zbasis[0],zbasis[1],zbasis[2], color="b",linestyle = lss[i])
        #HT,=ax.plot(hbasis[0],hbasis[1],hbasis[2], color="k",linestyle = lss[i])
        
        if(i==len(trajs)-1):
            w=0
            XTT,=ax.plot([0,xbasis[0][w]],[0,xbasis[1][w]],[0,xbasis[2][w]], color="r",marker='o',markeredgecolor='black',markerfacecolor='r',zorder=10)
            YTT,=ax.plot([0,ybasis[0][w]],[0,ybasis[1][w]],[0,ybasis[2][w]], color="g",marker='o',markeredgecolor='black',markerfacecolor='g',zorder=10)
            ZTT,=ax.plot([0,zbasis[0][w]],[0,zbasis[1][w]],[0,zbasis[2][w]], color="b",marker='o',markeredgecolor='black',markerfacecolor='b',zorder=10)
            NTT,=ax.plot([0,nvec[0]],[0,nvec[1]],[0,nvec[2]], color="k",marker='o',markeredgecolor='black',markerfacecolor='black',zorder=10)
            
            #HTT,=ax.plot([0,hbasis[0][-1]],[0,hbasis[1][-1]],[0,hbasis[2][-1]], color="k",marker='o',markeredgecolor='black',markerfacecolor='black')

    ax.plot([0],[0],[0], color="k",linestyle = "solid")
    ax.plot([0],[0],[0], color="k",linestyle = "dashed")
    #ax.plot(zbasis[0],zbasis[1],zbasis[2], color="b",linestyle = lss[i])
    #ax.set_aspect("equal")
    lines = ax.get_lines()
    

    legend1 = plt.legend([lines[i] for i in [13,14]], ["True Optimum", "Eigen-axis"], loc=1)
    legend2 = plt.legend([lines[i] for i in [9,10,11,12]], [r'$\hat{X}(0)$',r'$\hat{Y}(0)$',r'$\hat{Z}(0)$',r'$\hat{n}_i$'], loc=2)

    ax.add_artist(legend1)
    ax.add_artist(legend2)
    
    ax.set_xlabel(r'${X}$')
    ax.set_ylabel(r'${Y}$')
    ax.set_zlabel(r'${Z}$')
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    #ax.legend()
    ax.plot_surface(x, y, z, color="cyan",alpha=0.152)
    
    
    u, v = np.mgrid[0:np.pi/2:50j, 0:np.pi/2:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    #ax.legend()
    ax.plot_surface(x, y, z, color="k",alpha=0.13)
    
    plt.show()
    
    

    
    
