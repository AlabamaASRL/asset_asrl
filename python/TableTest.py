import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import MKgSecConstants as c
from SpiceRead import GetEphemTraj2
import Date as dt
from scipy.spatial.transform import Rotation as Rot
norm = np.linalg.norm

def normalize(x): 
    return np.copy(x)/norm(x)
    
DiffTab = ast.OptimalControl.FiniteDiffTable
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Tmodes = oc.TranscriptionModes

def MccinnesSail(r,n,beta,mu,n1,n2,t1):
    ndr  = vf.dot(r, n)
    rn   = r.norm()*n.norm()
    ncr  = vf.cross(n,r)
    ncrn = vf.cross(ncr,n)
    N3DR4 = vf.dot(n.normalized_power3(),r.normalized_power4())
    sc= (beta*mu/2.0)
    acc = N3DR4*(((n1*sc)*ndr + (n2*sc)*rn)*n  + (t1*sc)*ncrn)
    return acc #+ acc*0 #+ acc*0 + acc*0 #+ acc*0 + acc*0 + acc*0 + acc*0 #+ acc*0 + acc*0 + acc*0

def MccinnesSailC(r,n,beta,mu,rbar=.91,sbar=.89,Bf=.79,Bb=.67,ef=.025,eb=.27):
    n1 = 1 + rbar*sbar
    n2 = Bf*(1-sbar)*rbar + (1-rbar)*(ef*Bf - eb*Bb)/(ef+eb)
    t1 = 1 - sbar*rbar
    return MccinnesSail(r,n,beta,mu,n1,n2,t1)

class EPPRFrame:
    def __init__(self,P1name,P1mu,P2name,P2mu,Lstar,JD0,JDF,N = 3000):
        
        self.P1name = P1name
        self.P2name = P2name
        
        self.P1mu  = P1mu
        self.P2mu  = P2mu
        self.Lstar = Lstar
        self.Vstar = np.sqrt(P1mu/Lstar)
        self.Tstar = self.Lstar/self.Vstar
        print(self.Vstar)
        self.P1Data = GetEphemTraj2(P1name,JD0,JDF,N,self.Lstar,self.Tstar)
        self.P2Data = GetEphemTraj2(P2name,JD0,JDF,N,self.Lstar,self.Tstar)
        
        self.P2DotData = DiffTab(6,self.P2Data).all_derivs(1,4)
        self.RelData = np.copy(self.P2Data)
        for i,T in enumerate(self.RelData):
            T[0:6] = T[0:6] - self.P1Data[i][0:6]
            
        Rdata = [[norm(T[0:3]),T[6]] for T in self.RelData]
        
        Rdiff = DiffTab(1,Rdata)
        
        self.RData = []
        
        for i,Rt in enumerate(Rdata):
    
            R = Rt[0]
            t= Rt[1]
            Rdot  = Rdiff.deriv(i,1,4)[0]
            Rddot = Rdiff.deriv(i,2,4)[0]
            self.RData.append([R,Rdot,Rddot,t])
        ###########################################
        self.RotData = []
        for i,T in enumerate(self.RelData):
            r = T[0:3]
            v = T[3:6]
            xhat = normalize(r).tolist()
            zhat = normalize(np.cross(r,v)).tolist()
            yhat = normalize(np.cross(zhat,xhat)).tolist()
            
            self.RotData.append(xhat + yhat + zhat + [T[6]])
        
        self.D1RotData = DiffTab(9,self.RotData).all_derivs(1,4)
        
        self.WData = []
        self.BCaccData =[]
        self.GscaleData =[]
        self.VscaleData =[]
        self.RscaleData =[]
        
        for i,rot in enumerate(self.RotData):
            drot =  self.D1RotData[i]
            t= rot[9]
            r = self.RData[i][0]
            rdot = self.RData[i][1]
            rddot = self.RData[i][2]
            
            xhat = rot[0:3]
            yhat = rot[3:6]
            zhat = rot[6:9]
            
            dxhat = drot[0:3]
            dyhat = drot[3:6]
            dzhat = drot[6:9]
            
            DCM = np.array([xhat,yhat,zhat]).T
            dDCM = np.array([dxhat,dyhat,dzhat]).T
            
            Omat = np.matmul(dDCM,DCM.T)
            
            wx = Omat[2,1]
            wy = Omat[0,2]
            wz = Omat[1,0]
            w = np.array([wx,wy,wz])
            
            W = np.zeros((4))
            W[3] = rot[9]
            W[0:3] = np.dot(DCM.T,w)
            
            BCacc = np.zeros((4))
            BCacc[3] = rot[9]
            BCacc[0:3] =  np.dot(DCM.T,self.P2DotData[i][3:6]/r)
            
            self.WData.append(W)
            self.BCaccData.append(BCacc)
            self.GscaleData.append([r**(-3),t])
            self.VscaleData.append([rdot/r,t])
            self.RscaleData.append([rddot/r,t])
            
            
            
        self.WdotData = DiffTab(3,self.WData).all_derivs(1,4)
        
        #####################################################
        self.WTable      = ast.OptimalControl.LGLInterpTable(3,self.WData,len(self.WData))
        self.WdotTable   = ast.OptimalControl.LGLInterpTable(3,self.WdotData,len(self.WdotData))
        self.BCaccTable  = ast.OptimalControl.LGLInterpTable(3,self.BCaccData,len(self.BCaccData))
        self.GscaleTable = ast.OptimalControl.LGLInterpTable(1,self.GscaleData,len(self.GscaleData))
        self.VscaleTable = ast.OptimalControl.LGLInterpTable(1,self.VscaleData,len(self.VscaleData))
        self.RscaleTable = ast.OptimalControl.LGLInterpTable(1,self.RscaleData,len(self.RscaleData))
        
        fun = ast.OptimalControl.InterpFunction_3(self.WTable).vf()
        fun.rpt([1],1000000)
        
        ######################################################
    
    
    def plotRData(self):
        
        fig, axs = plt.subplots(3,1)
        
        RDT = np.copy(np.array(self.RData).T)
        
        axs[0].plot(RDT[3],RDT[0])
        axs[1].plot(RDT[3],RDT[1])
        axs[2].plot(RDT[3],RDT[2])
        plt.show()
    def plotWData(self):
        
        fig, axs = plt.subplots(2,1)
        
        WDT = np.copy(np.array(self.WData).T)
        
        axs[0].plot(WDT[3],WDT[0])
        axs[0].plot(WDT[3],WDT[1])
        axs[0].plot(WDT[3],WDT[2])
        
        WDT = np.copy(np.array(self.WdotData).T)
        
        axs[1].plot(WDT[3],WDT[0])
        axs[1].plot(WDT[3],WDT[1])
        axs[1].plot(WDT[3],WDT[2])

        plt.show()
    def plotBCData(self):
        
        
        WDT = np.copy(np.array(self.BCaccData).T)
        
        plt.plot(WDT[3],WDT[0])
        plt.plot(WDT[3],WDT[1])
        plt.plot(WDT[3],WDT[2])
        plt.show()
        
        
        

JD0 = 2459599.0
JDF = JD0 + 5.0*365.0        
Frame = EPPRFrame("SUN",c.MuSun,"EARTH",c.MuEarth,c.AU,JD0,JDF)        

#Frame.plotWData()       
        

def EPPRDynamics(mu,beta,Frame):
    args = Args(10)
    r = args.head3()
    v = args.segment3(3)
    n = args.tail3()
   
    
    
    p1loc = np.array([-mu,0,0])
    p2loc = np.array([1.0-mu,0,0])
    
    g1 = r.normalized_power3(-p1loc,(mu-1.0))
    g2 = r.normalized_power3(-p2loc,(-mu))
    
    rvec  = r 
    rvec.rpt([1,1,1],1000000)
    
    ### Boiler Plate that can be shortened
    Gscale = oc.InterpFunction_1(Frame.GscaleTable).sf().eval(args[6])
    Rscale = oc.InterpFunction_1(Frame.RscaleTable).sf().eval(args[6])
    Vscale = oc.InterpFunction_1(Frame.VscaleTable).sf().eval(args[6])
    BCacc = -oc.InterpFunction_3(Frame.BCaccTable).vf().eval(args[6])
    W = oc.InterpFunction_3(Frame.WTable).vf().eval(args[6])
    Wdot = oc.InterpFunction_3(Frame.WdotTable).vf().eval(args[6])
    
    

    Sail  =MccinnesSailC(r-p1loc,n,beta,mu)
    
    Grav    = (vf.Sum([g1,g2,Sail]))*Gscale
    wtemp   = vf.Sum([(-2.0)*v , Vscale*rvec, - vf.cross(W,rvec)])
    Wacc    = vf.cross(W, wtemp) 
    Wdotacc = - vf.cross(Wdot,rvec)
    Pulseacc1 = (rvec)*Rscale 
    Pulseacc2 = (v)*Vscale
    
    acc = vf.Sum([Grav,Wacc,Wdotacc,BCacc,Pulseacc1,Pulseacc2])
    
    return vf.Stack([v,acc])
    
def CR3BP(mu,DoLT = False,ltacc=.05):
    irows = 7
    if(DoLT==True): irows+=3
    
    args = vf.Arguments(irows)
    r = args.head3()
    v = args.segment3(3)
    
    x    = args[0]
    y    = args[1]
    xdot = args[3]
    ydot = args[4]
    
    
    t1     = vf.SumElems([ydot,x],[ 2.0,1.0])
    t2     = vf.SumElems([xdot,y],[-2.0,1.0])
    
    rterms = vf.StackScalar([t1,t2]).padded_lower(1)
    
    p1loc = np.array([-mu,0,0])
    p2loc = np.array([1.0-mu,0,0])
    
    g1 = r.normalized_power3(-p1loc,(mu-1.0))
    g2 = r.normalized_power3(-p2loc,(-mu))
    
    if(DoLT==True):
        thrust = args.tail_3()*ltacc
        acc = vf.Sum([rterms,g1,g2,thrust])
    else:
        acc = vf.Sum([g1,g2,rterms])
    return vf.Stack([v,acc])
    
mu = c.MuEarth/c.MuSun


ode =EPPRDynamics(mu,.01,Frame).SuperTest([.99,0,0,0,0,0,.1,1,0,0],100000)
ode =CR3BP(mu,True,.00).SuperTest([.99,0,0,0,0,0,.1,1,0,0],100000)

        
        
            
        

        
        
        
        
        




