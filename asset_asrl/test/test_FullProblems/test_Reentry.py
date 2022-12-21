import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import unittest

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes    = oc.ControlModes


################### Non Dimensionalize ##################################
g0 = 32.2 
W  = 203000

Lstar = 100000.0     ## feet
Tstar = 60.0         ## sec
Mstar = W/g0         ## slugs

Vstar   = Lstar/Tstar
Fstar   = Mstar*Lstar/(Tstar**2)
Astar   = Lstar/(Tstar**2)
Rhostar = Mstar/(Lstar**3)
BTUstar = 778.0*Lstar*Fstar
Mustar  = (Lstar**3)/(Tstar**2)

tmax = 2500/Tstar
Re = 20902900          /Lstar
S  = 2690.0            /(Lstar**2)
m  = (W/g0)            /Mstar
mu = (0.140765e17)     /Mustar
rho0 =.002378          /Rhostar
h_ref = 23800          /Lstar

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
class ShuttleReentry(oc.ode_x_u.ode):
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
        ##############################################################
        super().__init__(ode,5,2)

def QFunc():
    h,v,alpha = Args(3).tolist()
    alphadeg = (180.0/np.pi)*alpha
    rhodim = rho0*vf.exp(-h/h_ref)*Rhostar
    vdim = v*Vstar
    
    qr = 17700*vf.sqrt(rhodim)*((.0001*vdim)**3.07)
    qa = c0 + c1*alphadeg + c2*(alphadeg**2)+ c3*(alphadeg**3)
    
    return qa*qr
 

class test_Reentry(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.NumSegments1 = 64
        self.NumSegments2 = 256
        
        
        self.FinalObj1 = -0.5958800738629952 
        self.MaxObjError1 = .01
        
        self.FinalObj2 = -0.534620087611498
        self.MaxObjError2 = .01
        
        self.MaximumIters1 = 250    
        self.MaximumIters2 = 50   
    
    def problem_impl(self,tmode,cmode):
        
        ##########################################################################
        tf  = 2000/Tstar

        ht0  = 260000/Lstar
        htf  = 80000 /Lstar
        vt0  = 25600/Vstar
        vtf  = 2500 /Vstar

        thetaf =  (vt0*tf + 0.5*(vtf-vt0)*tf)/Re

        gammat0 = np.deg2rad(-1.0)
        gammatf = np.deg2rad(-5.0)
        psit0   = np.deg2rad(90.0)


        ts = np.linspace(0,tf,200)

        TrajIG = []
        for t in ts:
            X = np.zeros((8))
            X[0] = ht0*(1-t/tf) + htf*t/tf
            X[1] = thetaf*t/tf
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
        phase.setControlMode(cmode)
        
        phase.addBoundaryValue("Front",range(0,6),TrajIG[0][0:6])
        phase.addLUVarBounds("Path",[1,3],np.deg2rad(-89.0),np.deg2rad(89.0),1.0)
        phase.addLUVarBound("Path",6,np.deg2rad(-90.0),np.deg2rad(90.0),1.0)
        phase.addLUVarBound("Path",7,np.deg2rad(-90.0),np.deg2rad(1.0) ,1.0)
        phase.addUpperDeltaTimeBound(tmax,1.0)
        phase.addBoundaryValue("Back" ,[0,2,3],[htf,vtf,gammatf])
        phase.addDeltaVarObjective(1,-1.0)
        
        
        phase.optimizer.set_OptLSMode("AUGLANG")
        phase.optimizer.MaxLSIters = 2
        phase.optimizer.MaxAccIters = 100
        phase.optimizer.PrintLevel = 3
        
        Flag1 = phase.solve_optimize()
        
        self.assertEqual(Flag1,ast.Solvers.ConvergenceFlags.CONVERGED, 
                         "Problem did not converge")
        
        self.assertLess(phase.optimizer.LastIterNum, self.MaximumIters1,
                         "Optimizer iterations exceeded expected maximum")
        
        phase.refineTrajManual(self.NumSegments2)
        phase.optimize()
        
        Obj1 = phase.optimizer.LastObjVal
        ObjError1 = abs(Obj1-self.FinalObj1)
        self.assertLess(ObjError1, self.MaxObjError1,
                 "Final objective significantly differs from known answer")

        
        phase.addUpperFuncBound("Path",QFunc(),[0,2,6],Qlimit,1/Qlimit)
        Flag2 = phase.optimize()
       
        self.assertEqual(Flag2,ast.Solvers.ConvergenceFlags.CONVERGED, 
                         "Problem did not converge")
        self.assertLess(phase.optimizer.LastIterNum, self.MaximumIters2,
                         "Optimizer iterations exceeded expected maximum")
        
        Obj2 = phase.optimizer.LastObjVal
        ObjError2 = abs(Obj2-self.FinalObj2)
        self.assertLess(ObjError2, self.MaxObjError2,
                 "Final objective significantly differs from known answer")

        
    
    def test_FullProblem(self):
        
        for tmode in ["LGL3","LGL5","LGL7","Trapezoidal"]:
            with self.subTest(TranscriptionMode=tmode):
                with self.subTest(cmode="HighestOrderSpline"):
                    self.problem_impl(tmode,"HighestOrderSpline")
                with self.subTest(cmode="BlockConstant"):
                    self.problem_impl(tmode,"BlockConstant")




if __name__ == "__main__":
    unittest.main(exit=False)

    
  
   
