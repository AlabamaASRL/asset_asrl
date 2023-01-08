import numpy as np
import asset as ast
import asset_asrl.Astro.Constants as c
from   asset_asrl.Astro.SpiceRead import GetEphemTraj2,PoleVector,SpiceFrameTransform
import asset_asrl.Astro.Date as dt
from   asset_asrl.Astro.Extensions.CR3BPFrame import CR3BPFrame
from   asset_asrl.Astro.DataReadWrite import ReadData,WriteData,ReadCopernicusFile
BProps=c.SpiceBodyProps



norm = np.linalg.norm
def normalize(x): return np.copy(x)/norm(x)
    

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
DiffTab = ast.OptimalControl.FiniteDiffTable
InterpTab = ast.OptimalControl.LGLInterpTable

class EPPRFrame(CR3BPFrame):
    def __init__(self,P1name,P1mu,P2name,P2mu,Lstar,JD0,JDF,N = 3000,SpiceFrame ='J2000'):
        
        
        CR3BPFrame.__init__(self,P1mu,P2mu,Lstar)
        
        self.P1name = P1name
        self.P2name = P2name
        self.JD0 = JD0
        self.JDF = JDF
        self.SpiceFrame =SpiceFrame
        self.P1Data = GetEphemTraj2(P1name,JD0,JDF,N,self.lstar,self.tstar,Frame=self.SpiceFrame)
        self.P2Data = GetEphemTraj2(P2name,JD0,JDF,N,self.lstar,self.tstar,Frame=self.SpiceFrame)
        
        self.AltBodyGTables  = {}
        self.AltBodyLocFuncs   = {}
        self.AltBodyMuVals   = {}
        self.AltBodyNames    = []
        
        self.CalcFrameData()
        self.P1_J2 = False
        self.P2_J2 = False
        
        
    
    def JD_to_NDTime(self, JD):
        if(isinstance(JD,float)):
            if(JD<self.JD0 or JD>self.JDF):
                print("Warning: Requested time falls outside frame time range")
        DeltaJD = JD -self.JD0
        return DeltaJD*24.0*3600.0/self.tstar
    def NDTime_to_JD(self, ND):
        DeltaJD = ND*self.tstar/(24.0*3600.0)
        return self.JD0 + DeltaJD
    
    def M_S(self,tnd):
        return 1.0/(self.RTable.Interpolate(tnd)[0]*self.vstar)
        
    def GetDCM(self,t):
        rot = self.RotTable.Interpolate(t)
        xhat = normalize(rot[0:3])
        yhat = normalize(rot[3:6])
        zhat = normalize(rot[6:9])
        DCM = np.array([xhat,yhat,zhat]).T
        return DCM
    def NDInertial_to_EPPR(self, ITraj,axis = 6): ## assumes already ND and reffed to JD0
        
        Ptraj = []
        F=self.NDInertial_to_Frame_Func()
        for T in ITraj:
            Ptraj.append(F.compute(T[0:7]))

        return Ptraj
    
    def NDInertial_to_Frame_Func(self):
        args = Args(16)
        X=args.head(3)
        V=args.segment3(3)
        t=args[6]
        Xbc = args.segment3(7)
        Vbc = args.segment3(10)
        r   = args[13]
        rdot = args[14]
        
        DCMT = vf.RowMatrix(self.RotFunc.eval(t),3,3)
        W = self.WFunc.eval(t)
        
        Xnd = (X-Xbc)/r
        Vnd = (V-Vbc)/r
        Xrot = DCMT*Xnd 
        Vrot = DCMT*Vnd + vf.cross(Xrot,W) - Xrot*rdot/r
        
        state = vf.Stack([Xrot,Vrot,t])
        
        realargs = Args(7)
        t = realargs[6]
        dataargs = vf.Stack([realargs,self.BCFunc.eval(t),self.RFunc.eval(t)])
        return state.eval(dataargs)
        
        
        

    def EPPR_to_NDInertial(self, ITraj,axis = 6): ## assumes already ND and reffed to JD0
        Ptraj = []
        for T in ITraj:
            t =T[axis]
            
            p1xv = self.BCTable.Interpolate(t)[0:6]
            p1 =p1xv[0:3]
            v1 =p1xv[3:6]
            rdat  = self.RTable.Interpolate(t)
            r=rdat[0]
            rdot = rdat[1]
            DCM = self.GetDCM(t)
            W = self.WTable.Interpolate(t)[0:3]
             
            Xrot = T[0:3]
            Vrot = T[3:6]
            Xnd = np.matmul(DCM,Xrot)
            Vnd = np.matmul(DCM,Vrot - np.cross(Xrot,W) + Xrot*rdot/r )
            Xnd = Xnd*r + p1
            Vnd = Vnd*r + v1
        
            State = np.zeros((7))
            State[0:3] = Xnd
            State[3:6] = Vnd
            State[6]=t
            Ptraj.append(State)
        
        return Ptraj
    
    def Frame_to_NDInertial_Func(self):
        args = Args(16)
        Xrot=args.head(3)
        Vrot=args.segment3(3)
        t=args[6]
        Xbc = args.segment3(7)
        Vbc = args.segment3(10)
        r   = args[13]
        rdot = args[14]
        
        DCM = vf.ColMatrix(self.RotFunc.eval(t),3,3)
        W = self.WFunc.eval(t)
        
        Xnd = (DCM*Xrot)*r + Xbc
        Vnd = (DCM*(Vrot - vf.cross(Xrot,W) + Xrot*rdot/r ))*r + Vbc
       
        state = vf.stack([Xnd,Vnd,t])
    
        realargs = Args(7)
        t = realargs[6]
        dataargs = vf.stack([realargs,self.BCFunc.eval(t),self.RFunc.eval(t)])
        return state.eval(dataargs)
        
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
        

            
    def GetSpiceBodyEPPRTraj(self,Name,N):
        ITraj = GetEphemTraj2(Name,self.JD0,self.JDF,N,self.lstar,self.tstar,Frame=self.SpiceFrame)
        return self.NDInertial_to_EPPR(ITraj)
    
    def GetSpiceBodyTraj(self,Name,N):
        ITraj = GetEphemTraj2(Name,self.JD0,self.JDF,N,self.lstar,self.tstar,Frame=self.SpiceFrame)
        return self.NDInertial_to_EPPR(ITraj)
    def GetSpiceBodyTable(self,Name,N):
        Traj = self.GetSpiceBodyTraj(Name,N)
        return InterpTab(6,Traj,N+1)
    
    def Test(self,Name,N):
        ITraj = GetEphemTraj2(Name,self.JD0,self.JDF,N,self.lstar,self.tstar,Frame=self.SpiceFrame)
        RTraj = self.NDInertial_to_EPPR(ITraj)
        ITraj2 = self.EPPR_to_NDInertial(RTraj)
        epf = self.NDInertial_to_Frame_Func()
        ipf = self.Frame_to_NDInertial_Func()
        for i in range(0,len(ITraj)):
            print(max(abs(ITraj[i] - ITraj2[i])))
            RT = epf.compute(ITraj[i])
            IT = ipf.compute(RTraj[i])
            print(max(abs(RTraj[i] - RT)))
            print(max(abs(ITraj[i] - IT)))
        return self.NDInertial_to_EPPR(ITraj)
    
    def AddSpiceBody(self,Name,mu=None,N = 5000):
        
        if(mu==None):Mu=BProps[Name]["Mu"]
        else:Mu = np.copy(mu)
        ETraj = self.GetSpiceBodyEPPRTraj(Name,N)
        GTraj = []
        for T in ETraj:
            X = np.zeros(4)
            X[3] = T[6]
            X[0:3]= T[0:3]
            GTraj.append(X)
        self.AltBodyGTables[Name] = InterpTab(3,GTraj,len(GTraj))
        self.AltBodyLocFuncs[Name]  = oc.InterpFunction_3(self.AltBodyGTables[Name]).vf()
        
        self.AltBodyMuVals[Name]  = Mu/self.mustar
        self.AltBodyNames.append(Name)
    def AddSpiceBodies(self,Names,N = 5000):
        for Name in Names:self.AddSpiceBody(Name,N=N)
        
    def GetPoleVectors(self,Name,N=6000):
        Poles = PoleVector(Name,self.SpiceFrame,self.JD0,self.JDF,N,self.tstar)
        for P in Poles:
            t =P[3]
            DCM = self.GetDCM(t)
            P[0:3] = np.matmul(DCM.T,P[0:3])
        return Poles
    def Add_P2_J2Effect(self,J2c=None,RadP2=None):
        if(J2c==None):J2c=BProps[self.P2name]["J2"]
        if(RadP2==None):RadP2=BProps[self.P2name]["Radius"]
        
        self.P2_Rad = RadP2/self.lstar
        self.P2_J2  = J2c
        self.P2_PoleVectors = self.GetPoleVectors("IAU_"+self.P2name,len(self.P2Data))
        self.P2_PoleFunc = oc.InterpFunction_3(oc.LGLInterpTable(3,self.P2_PoleVectors,len(self.P2Data))).vf()
        
    def Add_P1_J2Effect(self,J2c=None,RadP1=None):
        if(J2c==None):J2c=BProps[self.P1name]["J2"]
        if(RadP1==None):RadP1=BProps[self.P1name]["Radius"]
        self.P1_Rad = RadP1/self.lstar
        self.P1_J2  = J2c
        self.P1_PoleVectors = self.GetPoleVectors("IAU_"+self.P1name,len(self.P1Data))
        self.P1_PoleFunc = oc.InterpFunction_3(oc.LGLInterpTable(3,self.P1_PoleVectors,len(self.P1Data))).vf()
        
        
        
    def EPPREOMs(self,r,v,t, otherGaccs=[], otherAccs=[], otherEOMs=[], ActiveAltBodies = 'All', Enable_J2=False):
         
         
         Gscale =  self.GscaleFunc.eval(t)
         Rscale =  self.RscaleFunc.eval(t)
         Vscale =  self.VscaleFunc.eval(t)
         BCacc  =  self.BCaccFunc.eval(t)
         W      =  self.WFunc.eval(t)
         Wdot   =  self.WdotFunc.eval(t)
         
         
         g1 = r.normalized_power3(-self.P1,(self.mu-1.0))
         g2 = r.normalized_power3(-self.P2,(-self.mu))
         
         GravTerms = [g1,g2] + otherGaccs
        
         if(ActiveAltBodies=='All'):Names = self.AltBodyNames
         else:Names = self.ActiveAltBodies
         for Name in Names:
             rBody  = self.AltBodyLocFuncs[Name].eval(t)
             muBody = self.AltBodyMuVals[Name]
             GravTerms.append((rBody-r).normalized_power3()*(muBody))

         Grav  = vf.Sum(GravTerms)*Gscale
         
         if(Enable_J2==True):otherAccs+=self.J2_ACC(r,t)
         
         RelVec = r
         wtemp   = vf.Sum([(-2.0)*v , Vscale*RelVec, vf.cross(RelVec,W)])
         Wacc    = vf.cross(W, wtemp) 
         Wdotacc = vf.cross(RelVec,Wdot)
         Pulseacc1 = (RelVec)*Rscale 
         Pulseacc2 = (v)*Vscale
         acc = vf.sum([Grav,Wacc,Wdotacc,BCacc,Pulseacc1,Pulseacc2] + otherAccs)
         func= vf.stack([v,acc]+otherEOMs)
         return func
        
    def J2_ACC(self,r,t):
        
        J2Accs = []
        j2sc = self.AccscaleFunc.eval(t)**5
        
        if(self.P2_J2 != False):
             NP2 = self.P2_PoleFunc.eval(t)#.normalized()
             RP2 = r-self.P2
             Scale = 0.5*(self.mu)*self.P2_J2*(self.P2_Rad)**2
             dotterm = (vf.dot(RP2.normalized(),NP2))**2
             
             t1 = (15.0*dotterm - 3.0)*RP2.normalized_power5() 
             t2 = -6.0*vf.dot(RP2.normalized_power5(),NP2)*NP2
             #J2Accs.append(Scale*( t1 + t2))
             
             j2func = ast.Astro.J2Cartesian((self.mu),self.P2_J2,self.P2_Rad)
             
             J2Accs.append(j2func(RP2,NP2))
             
        if(self.P1_J2 != False):
             NP2 = self.P1_PoleFunc.eval(t)#.normalized()
             RP2 = r-self.P1
             Scale = 0.5*(1-self.mu)*self.P1_J1*(self.P1_Rad)**2
             dotterm = (vf.dot(RP2.normalized(),NP2))**2
             
             t1 = (15.0*dotterm - 3.0)*RP2.normalized_power5() 
             t2 = -6.0*vf.dot(RP2.normalized_power5(),NP2)*NP2
             
             #J2Accs.append(Scale*( t1 + t2))
             
             j2func = ast.Astro.J2Cartesian((1-self.mu),self.P1_J2,self.P1_Rad)
             
             J2Accs.append(j2func(RP2,NP2))
             
             
        if(len(J2Accs)>0):
            return [vf.Sum(J2Accs)*j2sc]
        else: return []
         
             
        
    def Copernicus_to_Frame(self,Filename,center="P2",SpiceFrame='J2000',Folder='Data'):
        data = ReadCopernicusFile(Filename,Folder)
        if(center=="P2"):
            Tab = self.P2Table
        elif(center=="P1"):
            Tab = self.P1Table
        Traj = []
        for T in data:
            X = np.zeros((7))
            TC = np.copy(T)
            if(SpiceFrame!=self.SpiceFrame):
                TC[0:6] =SpiceFrameTransform(SpiceFrame,self.SpiceFrame,TC[0:6],TC[7])
                
            tnd = self.JD_to_NDTime(TC[7])
            X[0:6] = Tab.Interpolate(tnd)[0:6]
            X[0:3] += TC[0:3]/self.lstar
            X[3:6] += TC[3:6]/self.vstar
            X[6] = tnd
            Traj.append(X)
        return self.NDInertial_to_EPPR(Traj)
    
    
    
    def Vector_to_Frame(self,posvel, JD,center="P2",SpiceFrame='J2000'):
        if(center=="P2"):
            Tab = self.P2Table
        elif(center=="P1"):
            Tab = self.P1Table
        Traj = []
        X = np.zeros((7))
        TC = np.copy(posvel)
        if(SpiceFrame!=self.SpiceFrame):
            TC[0:6] =SpiceFrameTransform(SpiceFrame,self.SpiceFrame,posvel,JD)
            
        tnd = self.JD_to_NDTime(JD)
        X[0:6] = Tab.Interpolate(tnd)[0:6]
        X[0:3] += TC[0:3]/self.lstar
        X[3:6] += TC[3:6]/self.vstar
        X[6] = tnd
        Traj.append(X)
        Traj = [Traj]
        return self.NDInertial_to_EPPR(Traj[0])[0]
    
    
        
        
    def CalcFrameData(self):
        
        self.BCData = np.copy(self.P1Data)
        for i in range(0,len(self.BCData)):
            self.BCData[i][0:6] = (self.P1mu*self.P1Data[i][0:6] + self.P2mu*self.P2Data[i][0:6])/(self.P1mu+self.P2mu)
            
        BCDotData = DiffTab(6,self.BCData).all_derivs(1,4)

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
        self.AccscaleData =[]
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
            
            #print(w)
            
            W = np.zeros((4))
            W[3] = rot[9]
            W[0:3] = np.dot(DCM.T,w)
            
            BCacc = np.zeros((4))
            BCacc[3] = rot[9]
            BCacc[0:3] =  -np.dot(DCM.T,BCDotData[i][3:6]/r)
            
            self.WData.append(W)
            self.BCaccData.append(BCacc)
            self.GscaleData.append([r**(-3),t])
            self.AccscaleData.append([1/r,t])
            self.VscaleData.append([-2.0*rdot/r,t])
            self.RscaleData.append([-rddot/r,t])
            
            
            
        

        self.WdotData = DiffTab(3,self.WData).all_derivs(1,4)
        
        ###################################################################
        self.P1Table      = InterpTab(6,self.P1Data,len(self.P1Data))
        self.P2Table      = InterpTab(6,self.P2Data,len(self.P2Data))
        self.BCTable      = InterpTab(6,self.BCData,len(self.BCData))

        self.RotTable      = InterpTab(9,self.RotData,len(self.RotData))
        self.RTable        = InterpTab(3,self.RData,len(self.RData))

        
        self.WTable      = InterpTab(3,self.WData,len(self.WData))
        self.WdotTable   = InterpTab(3,self.WdotData,len(self.WdotData))
        self.BCaccTable  = InterpTab(3,self.BCaccData,len(self.BCaccData))
        
        self.GscaleTable = InterpTab(1,self.GscaleData,len(self.GscaleData))
        self.VscaleTable = InterpTab(1,self.VscaleData,len(self.VscaleData))
        self.RscaleTable = InterpTab(1,self.RscaleData,len(self.RscaleData))
        self.AccscaleTable = InterpTab(1,self.AccscaleData,len(self.AccscaleData))

        
        #####################################################################
        self.RotFunc   = oc.InterpFunction(self.RotTable,range(0,9)).vf()
        self.BCFunc    = oc.InterpFunction_6(self.BCTable).vf()
        self.RFunc     = oc.InterpFunction_3(self.RTable ).vf()
        ####################################################################

        self.WFunc      = oc.InterpFunction_3(self.WTable).vf()
        self.WdotFunc   = oc.InterpFunction_3(self.WdotTable ).vf()
        self.BCaccFunc  = oc.InterpFunction_3(self.BCaccTable).vf()

        self.GscaleFunc = oc.InterpFunction_1(self.GscaleTable).sf()
        self.VscaleFunc = oc.InterpFunction_1(self.VscaleTable).sf()
        self.RscaleFunc = oc.InterpFunction_1(self.RscaleTable).sf()
        self.AccscaleFunc = oc.InterpFunction_1(self.AccscaleTable).sf()
        


