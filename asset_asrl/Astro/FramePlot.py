import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches
import seaborn as sns
import asset_asrl.Astro.Constants as c
import numpy as np
from seaborn import color_palette as colpal

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

class PlotBase:
    def __init__(self):
        self.Trajs = {}
        self.Points = {}
        self.AltObjs={}
        self.NormScales={}
        self.BBox={}
        self.POI ={}
        self.DefaultView = None
        
    def AddCircle(self,name,p,r,color='',marker='',markersize = 0,linestyle='--'):
        theta = np.linspace(0,2.0*np.pi,2000)
        xs = r*np.cos(theta) + np.full_like(theta,p[0])
        ys = r*np.sin(theta) + np.full_like(theta,p[1])
        zs = np.full_like(theta,0)
        self.Trajs[name]={'Traj':[xs,ys,zs],'Color':color,'Marker':marker,'linestyle':linestyle,'Markersize':markersize,'type':'circle','Name':name}
        
    def addTraj(self,traj,name,color='',marker='',markersize =0,linestyle='-'):
        self.Trajs[name]={'Traj':np.array(traj).T,'Color':color,'Marker':marker,'linestyle':linestyle,'Markersize':markersize,'Name':name}
    def addTrajSeq(self,Trajs,header ='',tags=None,colp = 'plasma'):
        if(tags==None):tags=range(0,len(Trajs))
        cols = colpal(colp,len(Trajs))
        for i in range(0,len(Trajs)):
            self.addTraj(Trajs[i],header + str(tags[i]),cols[i])
        
    def addPropTraj(self,traj,name,color='',marker='',markersize =0,linestyle='-'):
        self.addTraj(name,traj,color,marker,markersize,linestyle)
    def addPoint(self,point,name,color,marker='*',markersize =100,markeredgewidth =0.5 , edgcolor = 'black'):
         self.Points[name]={'Pos':point,'Color':color,'Marker':marker,'Markersize':markersize,'Markeredgewidth':markeredgewidth,'Name':name,'EdgeColor':edgcolor}
    def addPointSeq(self,Trajs,header ='',tags=None,colp = 'plasma',marker='*'):
        if(tags==None):tags=range(0,len(Trajs))
        cols = colpal(colp,len(Trajs))
        for i in range(0,len(Trajs)):
            self.addPoint(Trajs[i],header + str(tags[i]),cols[i],marker)

    def PlotSTraj(self, mesh ,col='b',name = '', norm =1 ,plot =True,Sl = .06):
        m1 = np.copy(mesh).T
        for s in m1[0:-1:norm]:
            fdir = s[0:3] + Sl*s[7:10]/np.linalg.norm(s[7:10])
            a = Arrow3D([s[0], fdir[0]], [s[1], fdir[1]], 
                [s[2], fdir[2]], mutation_scale=8, 
                lw=1.2, arrowstyle="-|>",label = 'Sail Normal',color='black')
            plot.add_artist(a)
        return plot
    def PlotSTraj2(self, mesh ,col='b',name = '', norm =1 ,plot =True,Sl = .06,view=[0,1]):
        m1 = np.copy(mesh).T
        for s in m1[0:-1:norm]:
            fdir =Sl*s[7:10]/np.linalg.norm(s[7:10])
            plot.arrow(s[view[0]],s[view[1]],fdir[view[0]],fdir[view[1]],color='black',width=.00005)
        return plot
    def PlotLTraj(self, mesh ,col='b',name = '', norm =1 ,plot =True,Sl = .06):
        m1 = np.copy(mesh).T
        for s in m1[0:-1:norm]:
            if(np.linalg.norm(s[8:11])>0.01):
                fdir = s[0:3] + Sl*s[8:11]
                a = Arrow3D([s[0], fdir[0]], [s[1], fdir[1]], 
                    [s[2], fdir[2]], mutation_scale=8, 
                    lw=1.2, arrowstyle="-|>",label = 'Sail Normal',color='red')
                plot.add_artist(a)
        return plot
    def PlotS2d(self, mesh ,col='b',name = '', norm =1,Sl = .06):
        m1 = np.copy(mesh).T
        for s in m1[0:-1:norm]:
            fdir = s[0:3] + Sl*s[7:10]/np.linalg.norm(s[7:10])
            plt.arrow(s[0],s[1], fdir[0], fdir[1],  fc='k', ec='k')
            
        
    
    def Plot2dAx(self,ax,pois='all',trajs='all',bbox = False,legend = False,view=[0,1],plegend=False,Arrows=False):
        if (pois == 'all'):
            pois = self.POI.keys()

        for p in pois:
            obj = self.POI[p]
            objp= obj["Pos"]
            ax.scatter(objp[view[0]],objp[view[1]],label = obj["Name"],c=obj["Color"],marker = obj["Marker"],s=obj["Size"],zorder=7)
        if(plegend):ax.legend()
        if (trajs == 'all'):
            trajs = self.Trajs.keys()
        for t in trajs:
            obj = self.Trajs[t]
            objp= obj["Traj"]
            ax.plot(objp[view[0]],objp[view[1]],label = obj["Name"],color=obj["Color"],marker = obj["Marker"],markersize=obj["Markersize"],linestyle=obj["linestyle"])
            if(len(objp)==10)and(Arrows==True):
                self.PlotSTraj2(objp,obj["Color"],obj["Name"],self.NormDensity,ax,self.NormScales['YZ'],view=view)

        for obj in self.Points.values():
            objp= obj["Pos"]
            ax.scatter(objp[view[0]],objp[view[1]],label = obj["Name"],c=obj["Color"],marker = obj["Marker"],s=obj["Markersize"],
                       edgecolors  = obj["EdgeColor"],linewidths =obj["Markeredgewidth"],zorder=3)
        if(legend):ax.legend()
        if(bbox !=False):
            ax.axis('equal')
            ax.set_xlim(self.BBox[bbox][0])
            ax.set_ylim(self.BBox[bbox][1])
        labs = ['X (ND)','Y (ND)','Z (ND)']
        ax.set_xlabel(labs[view[0]])
        ax.set_ylabel(labs[view[1]])
        
        return ax
    def Plot2d(self,*args):
        
        self.Plot2dAx(plt, *args)
        plt.show()
    
    
    def Plot3dAx(self,ax,pois='all',trajs='all',bbox = False,legend = True,Arrows=False,plegend = True):
        if (pois == 'all'):
            pois = self.POI.keys()
        for p in pois:
            obj = self.POI[p]
            objp= obj["Pos"]
            ax.scatter(objp[0],objp[1],objp[2],label = obj["Name"],c=obj["Color"],marker = obj["Marker"],s=obj["Size"])
        if(plegend):ax.legend()

        if (trajs == 'all'):
            trajs = self.Trajs.keys()
        for t in trajs:
            obj = self.Trajs[t]
            objp= obj["Traj"]
            ax.plot(objp[0],objp[1],objp[2],label = obj["Name"],color=obj["Color"],marker = obj["Marker"],markersize=obj["Markersize"])
            if(len(objp)==10 and Arrows==True):
             ax=  self.PlotSTraj(objp,obj["Color"],obj["Name"],self.NormDensity,ax,.004)
        for obj in self.Points.items():
            objp= obj["Pos"]
            ax.scatter(objp[0],objp[1],objp[2],label = obj["Name"],c=obj["Color"],marker = obj["Marker"],s=obj["Markersize"],
                       edgecolors  = 'black',linewidths =obj["Markeredgewidth"],zorder=3)
        ax.set_xlabel('X (ND)')
        ax.set_ylabel('Y (ND)')
        ax.set_zlabel('Z (ND)')
        if(bbox !=False):
            ax.set_xlim(self.BBox[bbox][0])
            ax.set_ylim(self.BBox[bbox][1])
            ax.set_zlim(self.BBox[bbox][2])
        if(legend):ax.legend()
        return ax
    def Plot3d(self,*args):
        fig = plt.figure()
        ax = fig.gca(projection='3d') 
        
        self.Plot3dAx(ax, *args)
        plt.show()
    

class CRPlot(PlotBase):
    def __init__(self, ode,p1name='Sun',p2name='Earth',p1col='gold',p2col='green'):
        PlotBase.__init__(self)
        
        self.ode = ode
        [[.985,.995],[-.005,.005],[-.005,.005]]
        p=40
        L1 = {"Pos":ode.L1,"Color":'magenta','Marker':'o','Size':p/2,'Name':"L1"}
        L2 = {"Pos":ode.L2,"Color":'cyan','Marker':'o','Size':p/2,'Name':'L2'}
        L3 = {"Pos":ode.L3,"Color":'orange','Marker':'o','Size':p/2,'Name':'L3'}
        L4 = {"Pos":ode.L4,"Color":'purple','Marker':'o','Size':p/2,'Name':'L4'}
        L5 = {"Pos":ode.L5,"Color":'magenta','Marker':'o','Size':p,'Name':'L5'}
        P1 = {"Pos":ode.P1,"Color":p1col,'Marker':'o','Size':p*1,'Name':p1name}
        P2 = {"Pos":ode.P2,"Color":p2col,'Marker':'o','Size':p*1.0,'Name':p2name}
        self.p12actual = False
        self.POI ={'L1':L1,"L2":L2,"L3":L3,"L4":L4,"L5":L5,"P1":P1,"P2":P2}
        self.Trajs = {}
        self.Points = {}
        self.AltObjs={}
        self.NormScales={}
        self.BBox={}
        sphoi = (ode.L2[0] - ode.L1[0])/2.0
        l1pos = ode.L1[0]
        l2pos = ode.L2[0]
        p2pos = ode.P2[0]
        self.BBox["L1P2L2"] = [[p2pos-sphoi,p2pos+sphoi],[-sphoi,sphoi],[-sphoi,sphoi]]
        self.BBox["L1P2L2_wide"] = [[p2pos-sphoi*2,p2pos+sphoi*2],[-sphoi*2,sphoi*2],[-sphoi*2,sphoi*2]]

        self.BBox["YZ"] = [[-sphoi,+sphoi],[-sphoi,sphoi],[-sphoi,sphoi]]

        self.BBox["L1P2"] = [[p2pos-sphoi,p2pos],[-sphoi/2,sphoi/2],[-sphoi/2,sphoi/2]]
        self.BBox["P2L2"] = [[p2pos,p2pos+sphoi],[-sphoi/2,sphoi/2],[-sphoi/4,sphoi*3/4]]
        self.BBox["L1"] = [[l1pos-sphoi/2,l1pos+sphoi/2],[-sphoi/2,sphoi/2],[-sphoi/3,sphoi/3]]
        self.BBox["L2"] = [[l2pos-sphoi/2,l2pos+sphoi/2],[-sphoi/2,sphoi/2],[-sphoi/2,sphoi/2]]

        pp = np.zeros((3))
        pp[0]=np.cos(-2.0*c.dtr)
        pp[1]=np.sin(-2.0*c.dtr)
        self.BBox["2deg"] = [[pp[0]-sphoi,pp[0]+sphoi],[pp[1]-sphoi,pp[1]+sphoi],[-sphoi,sphoi]]

        self.BBox["P1P2L5"] = [[0,1],[-1,0],[-.5,.5]]
        self.BBox["L5"] = [[.38,.62],[-.986,-.746],[-.1,.1]]
        self.BBox["All"] = [[-1,1],[-1,1],[-1,1]]
        self.BBox["offs"] = [[.9848-sphoi,.9848+sphoi],[-sphoi-.17,sphoi-.17],[-sphoi,sphoi]]
        self.NormDensity = 12;
        self.NormScales["L1P2L2"] = .005
        self.NormScales["2deg"] = .03

        self.NormScales["offs"] = .005
        self.NormScales["L1P2"] = .05
        self.NormScales["P2L2"] = .02
        self.NormScales["L1"]   = .0015
        self.NormScales["L2"]     = .002
        self.NormScales["P1P2L5"] = .07
        self.NormScales["L5"]     = .002
        self.NormScales["All"]    = .02
        
        self.DefaultView = 'L1P2'
        
class TBPlot(PlotBase):
    def __init__(self, ode,p1name='Sun',p1col='gold'):
        PlotBase.__init__(self)
        p=40
        P1 = {"Pos":[0,0,0],"Color":p1col,'Marker':'o','Size':p*1,'Name':p1name}
        self.p12actual = False
        self.POI ={"P1":P1}
        
        onebox =[-1.2,1.2]
        self.BBox["One"] = [onebox,onebox,onebox]
        twobox =[-2.2,2.2]
        self.BBox["Two"] = [twobox,twobox,twobox]
        threebox =[-3.2,3.2]
        self.BBox["Two"] = [threebox,threebox,threebox]
        
        