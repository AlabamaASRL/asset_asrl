import numpy as np
import matplotlib.pyplot as plt
import asset_asrl as ast
import asset_asrl.VectorFunctions as vf
import asset_asrl.OptimalControl as oc
from asset_asrl.VectorFunctions import Arguments as Args
import unittest



###################  ##################################

Lstar = 100000.0     ## feet
Tstar = 60.0         ## sec
Vstar = Lstar/Tstar


g0 = 32.2 
W  = 203000


tmax = 2500           
Re = 20902900          
S  = 2690.0            
m  = (W/g0)            
mu = (0.140765e17)    
rho0 =.002378          
h_ref = 23800        

a0 = -.20704
a1 = .029244

b0 = .07854
b1 = -.61592e-2
b2 = .621408e-3

c0 =  1.0672181
c1 = -.19213774e-1
c2 = .21286289e-3
c3 = -.10117e-5

Qlimit = 70.0

##############################################################################
class ShuttleReentry(oc.ODEBase):
    def __init__(self):
        ############################################################
        args  = oc.ODEArguments(5,2)
        
        h       = args.XVar(0)
        theta   = args.XVar(1)
        v       = args.XVar(2)
        gamma   = args.XVar(3)
        psi     = args.XVar(4)
        
        h,theta,v,gamma,psi = args.XVec().tolist()
        
        alpha   = args.UVar(0)
        beta    = args.UVar(1)
        
        alphadeg = (180.0/np.pi)*alpha
        
        CL  = a0 + a1*alphadeg
        CD  = b0 + b1*alphadeg + b2*(alphadeg**2)
        rho = rho0*vf.exp(-h/h_ref)
        r   = h + Re
        
        L   = 0.5*CL*S*rho*(v**2)
        D   = 0.5*CD*S*rho*(v**2)
        g   = mu/(r**2)
        
        sgam = vf.sin(gamma)
        cgam = vf.cos(gamma)
        
        sbet = vf.sin(beta)
        cbet = vf.cos(beta)
        
        spsi = vf.sin(psi)
        cpsi = vf.cos(psi)
        tantheta = vf.tan(theta)
        
        hdot     = v*sgam
        thetadot = (v/r)*cgam*cpsi
        vdot     = -D/m - g*sgam
        gammadot = (L/(m*v))*cbet +cgam*(v/r - g/v)
        psidot   = L*sbet/(m*v*cgam) + (v/(r))*cgam*spsi*tantheta
        
    
        ode = vf.stack([hdot,thetadot,vdot,gammadot,psidot])
        
        Vgroups = {}
        Vgroups[('h','altitude')] =  h
        Vgroups[("v","velocity")] =  v
        Vgroups["theta"] =  theta
        Vgroups["gamma"] =  gamma
        Vgroups["psi"]   =  psi
        Vgroups[("alpha","AoA")] =  alpha
        Vgroups["beta"]  =  beta
        Vgroups[("t","time")] =  args.TVar()
        
        ##############################################################
        super().__init__(ode,5,2,Vgroups = Vgroups)

def QFunc():
    h,v,alpha = Args(3).tolist()
    alphadeg = (180.0/np.pi)*alpha
    rhodim = rho0*vf.exp(-h/h_ref)
    vdim = v
    
    qr = 17700*vf.sqrt(rhodim)*((.0001*vdim)**3.07)
    qa = c0 + c1*alphadeg + c2*(alphadeg**2)+ c3*(alphadeg**3)
    
    return qa*qr
 

class test_Reentry(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.NumSegments1 = 40
        
        
        self.FinalObj1 = 0.5958800738629952 
        self.MaxObjError1 = .01
        
        self.FinalObj2 = 0.534620087611498
        self.MaxObjError2 = .01
        
    
    def problem_impl(self,tmode,cmode,mtol):
        
        ##########################################################################
        tf  = 2000

        ht0  = 260000
        htf  = 80000 
        vt0  = 25600
        vtf  = 2500 


        gammat0 = np.deg2rad(-1.0)
        gammatf = np.deg2rad(-5.0)
        psit0   = np.deg2rad(90.0)


        ts = np.linspace(0,tf,200)

        TrajIG = []
        for t in ts:
            X = np.zeros((8))
            X[0] = ht0*(1-t/tf) + htf*t/tf
            X[1] = 0
            X[2] = vt0*(1-t/tf) + vtf*t/tf
            X[3] = gammat0*(1-t/tf) + gammatf*t/tf
            X[4] = psit0
            X[5] = t
            X[6] =.00
            X[7] =.00
            TrajIG.append(np.copy(X))
            
            
        ################################################################


        ode = ShuttleReentry()
        
        phase = ode.phase(tmode,TrajIG,self.NumSegments1)
        
        phase.setAutoScaling(True)  
        phase.setUnits(h = Lstar,
                       v = Vstar,
                       t = Tstar)
        
        phase.setControlMode(cmode)
        phase.setAdaptiveMesh(True)

        phase.addBoundaryValue("Front",range(0,6),TrajIG[0][0:6])
        phase.addLUVarBound("Path","theta",np.deg2rad(-89.0),np.deg2rad(89.0))
        phase.addLUVarBound("Path","gamma",np.deg2rad(-89.0),np.deg2rad(89.0))
        phase.addLUVarBound("Path","AoA",np.deg2rad(-90.0),np.deg2rad(90.0))
        phase.addLUVarBound("Path","beta" ,np.deg2rad(-90.0),np.deg2rad(1.0))
        phase.addUpperDeltaTimeBound(tmax,1.0)
        
        phase.addBoundaryValue("Back" 
                                  ,["h","v","gamma"]
                                  ,[htf,vtf,gammatf])
        
        phase.addDeltaVarObjective("theta",-1.0)
        
        
        phase.optimizer.set_SoeLSMode("L1")
        phase.optimizer.set_OptLSMode("L1")
        
        phase.optimizer.MaxLSIters = 2
        phase.optimizer.MaxAccIters = 100
        phase.optimizer.PrintLevel = 3
        phase.setThreads(1,1)
        phase.optimizer.CNRMode =True
        phase.PrintMeshInfo = False
        phase.setMeshTol(mtol)

        Flag1 = phase.solve_optimize()
        
        self.assertEqual(Flag1,ast.Solvers.ConvergenceFlags.CONVERGED, 
                         "Problem did not converge")
        
        Traj1 = phase.returnTraj()

        Obj1 = Traj1[-1][1]
        ObjError1 = abs(Obj1-self.FinalObj1)
        self.assertLess(ObjError1, self.MaxObjError1,
                 "Final objective significantly differs from known answer")

        
        phase.addUpperFuncBound("Path",QFunc(),["h","v","alpha"],Qlimit,1/Qlimit)
        Flag2 = phase.optimize()
        Traj2 = phase.returnTraj()
       
        self.assertEqual(Flag2,ast.Solvers.ConvergenceFlags.CONVERGED, 
                         "Problem did not converge")
       
        
        Obj2 = Traj2[-1][1]
        ObjError2 = abs(Obj2-self.FinalObj2)
        self.assertLess(ObjError2, self.MaxObjError2,
                 "Final objective significantly differs from known answer")

        
    
    def test_FullProblem(self):
        
        for tmode in ["LGL3","LGL5","LGL7","Trapezoidal"]:
            with self.subTest(TranscriptionMode=tmode):
                
                mtol = 1.0e-4 if tmode=="Trapezoidal" else 1.0e-7
                
                with self.subTest(cmode="HighestOrderSpline"):
                    self.problem_impl(tmode,"HighestOrderSpline",mtol)
                with self.subTest(cmode="BlockConstant"):
                    self.problem_impl(tmode,"BlockConstant",mtol)




if __name__ == "__main__":
    unittest.main(exit=False)
