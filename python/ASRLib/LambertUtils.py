import numpy as np
import asset as ast
import random
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments




def LambertTreeSearchImpl(mu,integ, Tables, Events, T0s,LegTOFs, Best=10, ConcatLegs=False):
        
        
        Lens =[len(T0s)] +[len(LegT) for LegT in LegTOFs]
        Idxs = np.indices(Lens).T.reshape(-1,len(Lens))
        
        Tseqs=[]
        
        for Id in Idxs:
            ts = [T0s[Id[0]]]
            for i in range(1,len(Id)):
                ts.append(ts[-1] + LegTOFs[i-1][Id[i]])
            Tseqs.append(ts)
        
        def LegDV(Tab1,t1,Tab2,t2,lw,Integ=False):
            
            tof = t2-t1
            
            X1 =  Tab1.Interpolate(t1)
            X2 =  Tab2.Interpolate(t2)
            
            V1,V2 = ast.Astro.lambert_izzo(X1[0:3], X2[0:3],tof,mu,lw)
            DV = np.linalg.norm(V1-X1[3:6]) + np.linalg.norm(V2-X2[3:6])
            if(Integ==False):
                return DV
            else:
                IGS = np.copy(X1)
                IGS[3:6] = V1
                Traj = integ.integrate_dense(IGS,t2)
                return Traj
                
        
       
        BestTraj=[]
        DVs=[]
        SLseqs=[]
        
        for i,Seq in enumerate(Tseqs):
            print(float(i)/float(len(Tseqs)))
            DV = 0
            sl =[]
            for i in range(0,len(Seq)-1):
                Tab1 = Tables[i]
                Tab2 = Tables[i+1]
                t1  = Seq[i]
                t2  = Seq[i+1]
                DVL = LegDV(Tab1,t1,Tab2,t2,True)
                DVS = LegDV(Tab1,t1,Tab2,t2,False)
                flip = bool(random.randint(0, 1))

                if(flip==False):
                    DV+=DVS
                    sl.append(False)
                else:
                    DV+=DVL
                    sl.append(True)
                
            DVs.append(DV)
            SLseqs.append(sl)
            
            
        idx = np.argmin(DVs)
        Seq = Tseqs[idx]
        sl  = SLseqs[idx]
        BestTraj=[]
        for i in range(0,len(Seq)-1):
            Tab1 = Tables[i]
            Tab2 = Tables[i+1]
            t1  = Seq[i]
            t2  = Seq[i+1]
            BestTraj.append(LegDV(Tab1,t1,Tab2,t2,sl[i],integ))
        
        
        print(len(Tseqs))
        return BestTraj