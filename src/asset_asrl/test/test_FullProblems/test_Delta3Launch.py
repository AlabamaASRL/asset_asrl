import numpy as np
import asset as ast
import unittest


vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes    = oc.ControlModes


############################################################################

g0      =  9.80665 
Lstar   =  6378145           ## m   Radius of Earth
Tstar   =  961.0             ## sec Engine Burn Time
Mstar   =  301454.0          ## kgs Inital Mass of Rocket


Astar   =  Lstar/Tstar**2
Vstar   =  Lstar/Tstar
Rhostar =  Mstar/Lstar**3
Estar   =  Mstar*(Vstar**2)
Mustar  =  (Lstar**3)/(Tstar**2)
Fstar   =  Astar*Mstar
#############################################################################

mu      = 3.986012e14     /Mustar
Re      = 6378145      /Lstar
We      = 7.29211585e-5          *Tstar

RhoAir  = 1.225        /Rhostar
h_scale = 7200         /Lstar
g       = g0           /Astar


CD = .5
S  = 4*np.pi   /Lstar**2



TS = 628500      /Fstar
T1 = 1083100     /Fstar
T2 = 110094      /Fstar

IS = 283.33364   /Tstar
I1 = 301.68      /Tstar
I2 = 467.21       /Tstar

tS = 75.2        /Tstar
t1 = 261         /Tstar
t2 = 700         /Tstar


TMS = 19290      /Mstar
TM1 = 104380     /Mstar
TM2 = 19300      /Mstar
TMPay = 4164     /Mstar


PMS = 17010     /Mstar
PM1 = 95550     /Mstar
PM2 = 16820     /Mstar

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


##############################################################################
class Delta3(oc.ode_x_u.ode):
    def __init__(self,T,mdot):
        ####################################################
        args  = oc.ODEArguments(7,3)
        
        r = args.XVec().head3()
        v = args.XVec().segment3(3)
        m = args.XVar(6)
        u = args.tail3().normalized()
        
        h       = r.norm() - Re
        rho     = RhoAir * vf.exp(-h / h_scale)
        vr      = v + r.cross(np.array([0,0,We]))
        D       = (-0.5*CD*S)*rho*(vr*vr.norm())
        
        rdot    =  v
        vdot    =  (-mu)*r.normalized_power3() + (T*u + D)/m
        
        ode = vf.stack(rdot,vdot,-mdot)
        ####################################################
        super().__init__(ode,7,3)

def TargetOrbit(at,et,it, Ot,Wt):
    rvec,vvec = Args(6).tolist([(0,3),(3,3)])
    
    hvec = rvec.cross(vvec)
    nvec = vf.cross([0,0,1],hvec)
    
    r    = rvec.norm()
    v    = vvec.norm()
    
    eps = 0.5*(v**2) - mu/r
    
    a =  -0.5*mu/eps
    
    evec = vvec.cross(hvec)/mu - rvec.normalized()
    
    i = vf.arccos(hvec.normalized()[2]) 
    
    Omega = vf.arccos(nvec.normalized()[0])
    Omega = vf.ifelse(nvec[1]>0,Omega,2*np.pi -Omega)
    W = vf.arccos(nvec.normalized().dot(evec.normalized()))
    W = vf.ifelse(evec[2]>0,W,2*np.pi-W)
    return vf.stack([a-at,evec.norm()-et,i-it,Omega-Ot,W-Wt])
    
##############################################################################
class test_Delta3Launch(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        
        self.FinalObj = 7529.749892668763
        self.MaximumIters = 100
    
    def problem_impl(self,tmode,cmode,nsegs):
        at = 24361140 /Lstar
        et = .7308
        Ot = np.deg2rad(269.8)
        Wt = np.deg2rad(130.5)
        istart = np.deg2rad(28.5)
        
        
        y0      = np.zeros((6))
        y0[0:3] = np.array([np.cos(istart),0,np.sin(istart)])*Re
        y0[3:6] =-np.cross(y0[0:3],np.array([0,0,We]))
        y0[3]  += 0.0001/Vstar
        
        M0   =-.05
        OEF  = [at,et,istart,Ot,Wt,M0]
        yf   = ast.Astro.classic_to_cartesian(OEF,mu)
        
        ts   = np.linspace(0,tf_phase4,nsegs)
        
        IG1 =[]
        IG2 =[]
        IG3 =[]
        IG4 =[] 
        
        
        for t in ts:
            X = np.zeros((11))
            X[0:6]= y0 + (yf-y0)*(t/ts[-1])
            X[7]  = t
            
            if(t<tf_phase1):
                m= m0_phase1 + (mf_phase1-m0_phase1)*(t/tf_phase1)
                X[6]=m
                X[8:11]= vf.normalize([1,0,0])
                IG1.append(X)
            elif(t<tf_phase2):
                m= m0_phase2 + (mf_phase2-m0_phase2)*(( t-tf_phase1) / (tf_phase2 - tf_phase1))
                X[6]=m
                X[8:11]= vf.normalize([1,0,0])
                IG2.append(X)
            elif(t<tf_phase3):
                m= m0_phase3 + (mf_phase3-m0_phase3)*(( t-tf_phase2) / (tf_phase3 - tf_phase2))
                X[6]=m
                X[8:11]= vf.normalize([1,0,0])
                IG3.append(X)
            elif(t<tf_phase4):
                m= m0_phase4 + (mf_phase4-m0_phase4)*(( t-tf_phase3) / (tf_phase4 - tf_phase3))
                X[6]=m
                X[8:11]= vf.normalize([1,0,0])
                IG4.append(X)
            
        
        
        ode1 = Delta3(T_phase1,mdot_phase1)
        ode2 = Delta3(T_phase2,mdot_phase2)
        ode3 = Delta3(T_phase3,mdot_phase3)
        ode4 = Delta3(T_phase4,mdot_phase4)
        
       
        phase1 = ode1.phase(tmode,IG1,len(IG1)-1)
        phase1.setControlMode(cmode)
        phase1.addLUNormBound("Path",[8,9,10],.5,1.5)
        
        phase1.addBoundaryValue("Front",range(0,8),IG1[0][0:8])
        phase1.addBoundaryValue("Back",[7],[tf_phase1])
        
        phase2 = ode2.phase(tmode,IG2,len(IG2)-1)
        phase2.setControlMode(cmode)

        phase2.addLUNormBound("Path",[8,9,10],.5,1.5)
        phase2.addBoundaryValue("Front",[6], [m0_phase2])
        phase2.addBoundaryValue("Back", [7] ,[tf_phase2])
        
        phase3 = ode3.phase(tmode,IG3,len(IG3)-1)
        phase3.setControlMode(cmode)

        phase3.addLUNormBound("Path",[8,9,10],.5,1.5)
        phase3.addBoundaryValue("Front",[6], [m0_phase3])
        phase3.addBoundaryValue("Back", [7] ,[tf_phase3])
        
        phase4 = ode4.phase(tmode,IG4,len(IG4)-1)
        phase4.setControlMode(cmode)

        phase4.addLUNormBound("Path",[8,9,10],.5,1.5)
        phase4.addBoundaryValue("Front",[6], [m0_phase4])
        phase4.addValueObjective("Back",6,-1.0)
        phase4.addUpperVarBound("Back",7,tf_phase4,1.0)
        phase4.addEqualCon("Back",TargetOrbit(at,et,istart,Ot,Wt),range(0,6))
        
        phase1.addLowerNormBound("Path",[0,1,2],Re*.999999)
        phase2.addLowerNormBound("Path",[0,1,2],Re*.999999)
        phase3.addLowerNormBound("Path",[0,1,2],Re*.999999)
        phase4.addLowerNormBound("Path",[0,1,2],Re*.999999)
        
        ocp = oc.OptimalControlProblem()
        ocp.addPhase(phase1)
        ocp.addPhase(phase2)
        ocp.addPhase(phase3)
        ocp.addPhase(phase4)
        
        ocp.addForwardLinkEqualCon(phase1,phase4,[0,1,2,3,4,5,7,8,9,10])
        ocp.optimizer.set_OptLSMode("L1")
        ocp.optimizer.MaxLSIters = 2
        ocp.optimizer.PrintLevel=3

        ocp.Threads=8
        ocp.optimizer.QPThreads=8

        Flag = ocp.optimize()
        
        #Phase1Traj = phase1.returnTraj()  
        #Phase2Traj = phase2.returnTraj()
        #Phase3Traj = phase3.returnTraj()
        Phase4Traj = phase4.returnTraj()
        
        FinalMassKg = Phase4Traj[-1][6]*Mstar
        
        MassError = abs(FinalMassKg-self.FinalObj)
        
        self.assertEqual(Flag,ast.Solvers.ConvergenceFlags.CONVERGED, 
                         "Problem did not converge")
        
        self.assertLess(ocp.optimizer.LastIterNum, self.MaximumIters,
                         "Optimizer iterations exceeded expected maximum")
        
        self.assertLess(MassError, 1.0,
                         "Final mass differs from known answer by greater than 1 kg")
        
        
    
    def test_FullProblem(self):
        
        tmodes = ["LGL3","LGL5","LGL7","Trapezoidal"]
        nsegs  = [200   ,200   ,200   ,500]
        for tmode,nseg in zip(tmodes,nsegs):
            with self.subTest(TranscriptionMode=tmode):
                with self.subTest(cmode="HighestOrderSpline"):
                    self.problem_impl(tmode,"HighestOrderSpline",nseg)
                with self.subTest(cmode="BlockConstant"):
                    self.problem_impl(tmode,"BlockConstant",nseg)

        

        
        

###############################################################################

if __name__ == "__main__":

    unittest.main(exit=False)    
    

