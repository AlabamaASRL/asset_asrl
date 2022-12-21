import numpy as np
import asset as ast
import asset_asrl.Astro.Constants as c
from   asset_asrl.Astro.SpiceRead import GetEphemTraj2,PoleVector,SpiceFrameTransform
import asset_asrl.Astro.Date as dt
from   asset_asrl.Astro.DataReadWrite import ReadData,WriteData,ReadCopernicusFile
from   asset_asrl.Astro.Extensions.TwoBodyFrame import TwoBodyFrame

BProps=c.SpiceBodyProps


norm = np.linalg.norm
def normalize(x): return np.copy(x)/norm(x)
    

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
DiffTab = ast.OptimalControl.FiniteDiffTable
InterpTab = ast.OptimalControl.LGLInterpTable

class NBodyFrame(TwoBodyFrame):
    def __init__(self,P1name,P1mu,Lstar,JD0,JDF,N = 3000,SpiceFrame ='J2000'):
        
        TwoBodyFrame.__init__(self,P1mu,Lstar)
        self.P1name = P1name
        self.JD0 = JD0
        self.JDF = JDF
        self.SpiceFrame =SpiceFrame
        self.P1Data = GetEphemTraj2(P1name,JD0,JDF,N,self.lstar,self.tstar,Frame=self.SpiceFrame)
        
        self.AltBodyGTables  = {}
        self.AltBodyLocFuncs   = {}
        self.AltBodyMuVals   = {}
        self.AltBodyNames    = []
        
        self.CalcFrameData()
        self.P1_J2 = False
        
    def JD_to_NDTime(self, JD):
        if(isinstance(JD,float)):
            if(JD<self.JD0 or JD>self.JDF):
                print("Warning: Requested time falls outside frame time range")
        DeltaJD = JD -self.JD0
        return DeltaJD*24.0*3600.0/self.tstar
    def NDTime_to_JD(self, ND):
        DeltaJD = ND*self.tstar/(24.0*3600.0)
        return self.JD0 + DeltaJD
    
    def NDInertial_to_Frame(self,Traj,axis=6):
        NTraj = []
        for T in Traj:
            t = T[axis]
            XN = np.copy(T)
            XN[0:6] = T[0:6] - self.P1Table.Interpolate(t)[0:6]
            NTraj.append(XN)
        return NTraj
    def Frame_to_NDInertial(self,Traj,axis=6):
        NTraj = []
        for T in Traj:
            t = T[axis]
            XN = np.copy(T)
            XN[0:6] = T[0:6] + self.P1Table.Interpolate(t)[0:6]
            NTraj.append(XN)
        return NTraj
    
    def NDInertial_to_Frame_Func(self):
        args = Args(7)
        t = args[6]
        XN = args.head(6).vf() - self.P1Func.eval(t)
        return vf.stack([XN,t])
    def Frame_to_NDInertial_Func(self):
        args = Args(7)
        t = args[6]
        XN = args.head(6).vf() + self.P1Func.eval(t)
        return vf.stack([XN,t])
    def Transform_Func(self,OtherFrame):
        FrameToND1 = self.Frame_to_NDInertial_Func()
        ND2ToFrame = OtherFrame.NDInertial_to_Frame_Func()
        
        xscale = self.lstar/OtherFrame.lstar
        vscale = self.vstar/OtherFrame.vstar
        
        args = Args(7)
        Xnd1 = args.head3()
        Vnd1 = args.segment3(3)
        tnd1 = args[6]
        
        Xnd2 = Xnd1*xscale
        Vnd2 = Vnd1*vscale
        tnd2 = OtherFrame.JD_to_NDTime(self.NDTime_to_JD(tnd1))
        
        ND1toND2 = vf.stack([Xnd2,Vnd2,tnd2])
        
        return (ND2ToFrame.eval(ND1toND2)).eval(FrameToND1)
    
    def GetSpiceBodyTraj(self,Name,N):
        ITraj = GetEphemTraj2(Name,self.JD0,self.JDF,N,self.lstar,self.tstar,Frame=self.SpiceFrame)
        return self.NDInertial_to_Frame(ITraj)
    def GetSpiceBodyTable(self,Name,N):
        Traj = self.GetSpiceBodyTraj(Name,N)
        return InterpTab(6,Traj,N+1)
    def AddSpiceBody(self,Name,mu=None,N = 5000):
        if(mu==None):mu=BProps[Name]["Mu"]
        ETraj = self.GetSpiceBodyTraj(Name,N)
        GTraj = []
        for T in ETraj:
            X = np.zeros(4)
            X[3] = T[6]
            X[0:3]= T[0:3]
            GTraj.append(X)
        self.AltBodyGTables[Name]   = InterpTab(3,GTraj,len(GTraj))
        self.AltBodyLocFuncs[Name]  = oc.InterpFunction_3(self.AltBodyGTables[Name]).vf()
        self.AltBodyMuVals[Name]    = mu/self.mustar
        self.AltBodyNames.append(Name)
        
    def AddSpiceBodies(self,Names,N = 5000):
        for Name in Names:self.AddSpiceBody(Name,N=N)
        
    def Copernicus_to_Frame(self,Filename,SpiceFrame='J2000'):
        data = ReadCopernicusFile(Filename)
        Tab = self.P1Table
        Traj = []
        for T in data:
            X = np.zeros((7))
            TC = np.copy(T)
            if(SpiceFrame!=self.SpiceFrame):
                TC[0:6] =SpiceFrameTransform(SpiceFrame,self.SpiceFrame,TC[0:6],TC[7])
            tnd = self.JD_to_NDTime(TC[7])
            #X[0:6] = Tab.Interpolate(tnd)[0:6]
            X[0:3] += TC[0:3]/self.lstar
            X[3:6] += TC[3:6]/self.vstar
            X[6] = tnd
            Traj.append(X)
        return Traj
    
    def GetPoleVectors(self,Name,N=6000):
        Poles = PoleVector(Name,self.SpiceFrame,self.JD0,self.JDF,N,self.tstar)
        return Poles
    def Add_P1_J2Effect(self,J2c=None,RadP1=None):
        if(J2c==None):J2c=BProps[self.P1name]["J2"]
        if(RadP1==None):RadP1=BProps[self.P1name]["Radius"]

        self.P1_Rad = RadP1/self.lstar
        self.P1_J2  = J2c
        self.P1_PoleVectors = self.GetPoleVectors("IAU_"+self.P1name,len(self.P1Data))
        self.P1_PoleFunc = oc.InterpFunction_3(oc.LGLInterpTable(3,self.P1_PoleVectors,len(self.P1Data))).vf()
    def CalcFrameData(self):
        self.P1Table = InterpTab(6,self.P1Data,len(self.P1Data))
        self.P1Func  = oc.InterpFunction_6(self.P1Table).vf()
        P1AccData = DiffTab(6,self.P1Data).all_derivs(1,4)
        P1AccD = []
        for P in P1AccData:
            X = np.zeros((4))
            X[3]=P[6]
            X[0:3]=-P[3:6]
            P1AccD.append(X)
        self.P1AccTable = InterpTab(3,P1AccD,len(P1AccD))
        self.P1AccFunc  = oc.InterpFunction_3(self.P1AccTable).vf()

    def NBodyEOMs(self,r,v,t,otherAccs=[], otherEOMs=[], ActiveAltBodies = 'All', Enable_J2=False ,Enable_P1_Acc=True):
        if(ActiveAltBodies=='All'):Names = self.AltBodyNames
        else:Names = self.ActiveAltBodies
        for Name in Names:
             rBody  = self.AltBodyLocFuncs[Name].eval(t)
             muBody = self.AltBodyMuVals[Name]
             otherAccs.append((rBody-r).normalized_power3()*(muBody))
             
        if(self.P1_J2 != False and Enable_J2==True):
             p = self.P1_PoleFunc.eval(t)
             j2func = ast.Astro.J2Cartesian((self.mu),self.P1_J2,self.P1_Rad)
             otherAccs.append(j2func(r,p))
             
            
        if(Enable_P1_Acc==True):
            otherAccs.append(self.P1AccFunc.eval(t))
        
        return self.TwoBodyEOMs(r,v,otherAccs,otherEOMs)
    
    


